"""Inference utilities: top-K next-step prediction, completion, anomaly
scoring — usable with any checkpoint produced by `src.train.trainer`.

Designed to run on Leonardo compute nodes where torch + the checkpoints
live. The functions here are imported by `src.eval.run_eval` (the CLI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.tokenizer import BaseTokenizer, build_tokenizer
from src.data.validator import (
    RULE_IDS,
    validate_sequence,
)
from src.model.registry import build_model


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: BaseTokenizer
    cfg: dict
    device: torch.device


def load_model(
    checkpoint_path: Path, device: torch.device | None = None, eval_mode: bool = True
) -> LoadedModel:
    """Load a model + tokenizer from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = build_tokenizer(cfg["tokenization"]["mode"])
    enable_heads = cfg["loss"]["validity_weight"] > 0 or cfg["loss"]["rule_id_weight"] > 0
    model = build_model(
        cfg["arch"], cfg["model"], tokenizer.vocab_size, enable_multitask_heads=enable_heads
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    if eval_mode:
        model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, cfg=cfg, device=device)


def encode_prefix(
    tokenizer: BaseTokenizer, family: str, prefix_steps: list[str], max_len: int = 256
) -> torch.Tensor:
    """Encode a prefix as [BOS, FAMILY, step_tokens...] for next-step prediction."""
    step_ids = tokenizer.encode_steps(prefix_steps)
    wrapped = [tokenizer.bos_id, tokenizer.family_id(family), *step_ids]
    # No EOS at inference time — the next-step we want to predict goes here.
    if len(wrapped) > max_len:
        head = wrapped[:2]
        tail = wrapped[-(max_len - 2) :]
        wrapped = head + tail
    return torch.tensor(wrapped, dtype=torch.long)


def _step_topk_logits(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    family: str,
    prefix_steps: list[str],
    k_pool: int,
    device: torch.device,
) -> list[tuple[str, float]]:
    """Return [(step_string, logit/score)] ranked from highest to lowest."""
    input_ids = encode_prefix(tokenizer, family, prefix_steps).to(device)
    with torch.no_grad():
        out = model(input_ids.unsqueeze(0))
    next_logits = out["lm_logits"][0, -1]  # last position
    if tokenizer.mode == "step":
        topk = torch.topk(next_logits, k=min(k_pool, next_logits.numel()))
        candidates: list[tuple[str, float]] = []
        for tok_id, score in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
            step = tokenizer.id_to_token[tok_id]
            if step.startswith("<") and step.endswith(">"):
                continue
            candidates.append((step, score))
        return candidates
    # Compositional: a step is W word-tokens followed by <STEP>. We greedy-decode
    # candidates by sampling beams of word tokens until <STEP> appears.
    return _compositional_topk(model, tokenizer, family, prefix_steps, k_pool, device)


def _compositional_topk(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    family: str,
    prefix_steps: list[str],
    k_pool: int,
    device: torch.device,
    max_words: int = 8,
    beam_width: int = 12,
) -> list[tuple[str, float]]:
    """For compositional models: beam search until <STEP> delimiter to assemble
    next-step candidates as strings."""
    step_id = tokenizer.step_id
    eos_id = tokenizer.eos_id
    base = encode_prefix(tokenizer, family, prefix_steps)
    base_list = base.tolist()
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    completed: list[tuple[list[int], float]] = []
    for _ in range(max_words):
        if not beams:
            break
        seqs = []
        for ids, _ in beams:
            seq = base_list + ids
            seqs.append(seq)
        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_id
        x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn_mask = (x != pad_id).long()
        with torch.no_grad():
            out = model(x, attn_mask=attn_mask)
        lengths = attn_mask.sum(dim=1)
        all_logits = out["lm_logits"]
        idx = (lengths - 1).clamp(min=0)
        logits = all_logits[torch.arange(x.size(0)), idx]  # [B, V]
        logp = F.log_softmax(logits, dim=-1)
        top = torch.topk(logp, k=beam_width, dim=-1)
        new_beams: list[tuple[list[int], float]] = []
        for i, (ids, score) in enumerate(beams):
            for j in range(beam_width):
                tok = int(top.indices[i, j])
                added = float(top.values[i, j])
                new_ids = ids + [tok]
                new_score = score + added
                if tok == step_id:
                    completed.append((new_ids, new_score))
                elif tok == eos_id:
                    completed.append((new_ids, new_score))
                else:
                    new_beams.append((new_ids, new_score))
        new_beams.sort(key=lambda b: -b[1])
        beams = new_beams[:beam_width]
        if len(completed) >= k_pool * 2:
            break
    candidates: list[tuple[str, float]] = []
    for ids, score in completed:
        words = []
        for t in ids:
            if t in (tokenizer.step_id, tokenizer.eos_id, tokenizer.pad_id):
                break
            tok = tokenizer.id_to_token[t]
            if tok.startswith("<") and tok.endswith(">"):
                continue
            words.append(tok)
        step_str = " ".join(words)
        if step_str:
            candidates.append((step_str, score))
    best: dict[str, float] = {}
    for s, sc in candidates:
        if s not in best or sc > best[s]:
            best[s] = sc
    out = sorted(best.items(), key=lambda kv: -kv[1])[:k_pool]
    return out


def candidate_violates(prefix: list[str], candidate: str) -> bool:
    new_prefix = prefix + [candidate]
    new_idx = len(prefix)
    for v in validate_sequence(new_prefix):
        if v.step_index == new_idx:
            return True
    return False


def topk_next_step(
    lm: LoadedModel,
    family: str,
    prefix_steps: list[str],
    k: int = 5,
    k_pool: int = 30,
    grammar: bool = True,
) -> list[str]:
    """Top-K next-step prediction with optional grammar mask."""
    pool = _step_topk_logits(lm.model, lm.tokenizer, family, prefix_steps, k_pool, lm.device)
    if not grammar:
        return [s for s, _ in pool[:k]]
    kept: list[str] = []
    for s, _ in pool:
        if candidate_violates(prefix_steps, s):
            continue
        kept.append(s)
        if len(kept) >= k:
            break
    if not kept:
        return [s for s, _ in pool[:k]]
    while len(kept) < k:
        for s, _ in pool:
            if s not in kept:
                kept.append(s)
                break
        else:
            break
    return kept[:k]


def complete_sequence(
    lm: LoadedModel, family: str, prefix_steps: list[str], max_len: int = 200, grammar: bool = True
) -> list[str]:
    """Greedy completion with optional grammar mask."""
    cur = list(prefix_steps)
    out: list[str] = []
    for _ in range(max_len):
        top = topk_next_step(lm, family, cur, k=1, k_pool=20, grammar=grammar)
        if not top:
            break
        nxt = top[0]
        out.append(nxt)
        cur.append(nxt)
        if nxt == "SHIP LOT":
            break
    return out


def anomaly_ensemble(
    lm: LoadedModel, family: str, full_sequence: list[str]
) -> dict[str, float | int | str]:
    """Score a full sequence for validity.

    Ensemble:
      1. Symbolic validator (oracle for the 10 known rules).
      2. If multi-task heads available: validity_logit + rule_id_logits.
      3. Fallback: LM perplexity z-score (placeholder; computed but not the
         primary signal).

    Returns a dict matching the submission CSV columns:
      IS_VALID (0/1), SCORE (probability valid, 0..1), PREDICTED_RULE (str or "")
    """
    violations = validate_sequence(full_sequence)
    if violations:
        return {
            "IS_VALID": 0,
            "SCORE": 0.05,  # near-zero P(valid)
            "PREDICTED_RULE": violations[0].rule,
        }
    # Symbolic says valid. Cross-check with the learned head if present.
    input_ids = encode_prefix(lm.tokenizer, family, full_sequence).to(lm.device)
    # Append <EOS> so the pooled rep sees the end.
    input_ids = torch.cat([input_ids, torch.tensor([lm.tokenizer.eos_id], device=lm.device)])
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = lm.model(input_ids.unsqueeze(0), attn_mask=attn.unsqueeze(0))
    if "validity_logit" in out:
        p_valid = float(torch.sigmoid(out["validity_logit"])[0])
        if p_valid < 0.5:
            # Heads disagree with the symbolic check — choose the heads' guess.
            rule = ""
            if "rule_id_logits" in out:
                rid = int(out["rule_id_logits"][0].argmax())
                if 0 <= rid < len(RULE_IDS):
                    rule = RULE_IDS[rid]
            return {"IS_VALID": 0, "SCORE": p_valid, "PREDICTED_RULE": rule}
        return {"IS_VALID": 1, "SCORE": p_valid, "PREDICTED_RULE": ""}
    # No head — go with symbolic.
    return {"IS_VALID": 1, "SCORE": 0.95, "PREDICTED_RULE": ""}
