"""Small, role-factored, constraint-aware neural ranker (the "neuro" passenger).

This is deliberately tiny (track evidence: >5M params is wasted on this data).
It is a next-step language model over process sequences, factored through the
family-agnostic role ontology so the part that *can* transfer across families is
maximized and the part that can't is minimized:

  * per-position input  = step-embedding + role-embedding + family-embedding + positional
  * backbone            = a SMALL causal ``nn.TransformerEncoder`` (batch_first)
  * two heads           = ``step_head`` (next step) and ``role_head`` (next role)

It implements the RANKER PROTOCOL (``predict`` / ``predict_roles`` /
``perplexity``) via :class:`NeuralRanker`, so ``decode`` / ``predict`` / ``eval``
treat it identically to ``nspe.ppm.PPM``. Symbolic legality is *not* baked into
the default training objective — the symbolic engine masks at inference time in
``decode`` (the constrained support is what guarantees rule-validity). An optional
semantic/constraint loss can be switched on via the config (``sem_w`` /
``mask_train``); building the per-position legal-step mask is expensive so it is
only constructed (and cached to ``NSPE_OUT``) when requested.

Only this module and ``nspe.losses`` import ``torch``. The symbolic core never
does; ``predict``/``eval`` import this module lazily, only to reload a checkpoint.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from nspe.data import build_step_index, load_family, role_encode
from nspe.losses import total_loss
from nspe.official import FAMILIES
from nspe.roles import NUM_ROLES, role_idx

__all__ = [
    "ConstrainedRanker",
    "NeuralRanker",
    "load_ranker",
    "train_ranker",
    "DEFAULT_CONFIG",
]

# Special step ids (mirrors nspe.data.build_step_index).
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Role-stream padding id: roles occupy 0..NUM_ROLES-1, so NUM_ROLES is a free pad.
ROLE_PAD_ID = NUM_ROLES

DEFAULT_CONFIG: Dict = {
    "d_model": 128,
    "layers": 3,
    "heads": 4,
    "steps": 2000,      # max optimizer steps
    "epochs": 8,
    "batch": 32,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "role_w": 0.3,
    "sem_w": 0.0,       # default: plain CE+role; rely on inference-time masking
    "mask_train": False,
    "max_len": 256,
    "seed": 0,
    "warmup": 50,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConstrainedRanker(nn.Module):
    """A small causal transformer next-step ranker with step/role/family inputs.

    Parameters
    ----------
    vocab   : step vocabulary size (including specials).
    n_roles : number of role embeddings (NUM_ROLES + 1 to include a role-pad slot).
    n_fam   : number of families.
    d       : model width (default 128).
    layers  : transformer encoder layers (default 3).
    heads   : attention heads (default 4).
    max_len : maximum positional index supported.
    """

    def __init__(
        self,
        vocab: int,
        n_roles: int,
        n_fam: int,
        d: int = 128,
        layers: int = 3,
        heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d = d
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD_ID)
        self.role = nn.Embedding(n_roles, d, padding_idx=ROLE_PAD_ID)
        self.fam = nn.Embedding(n_fam, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: incompatible with norm_first and with a
        # custom attention mask; keeps behaviour stable across torch 2.4-2.10.
        self.body = nn.TransformerEncoder(layer, num_layers=layers,
                                          enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.step_head = nn.Linear(d, vocab)
        self.role_head = nn.Linear(d, n_roles)

    def forward(
        self,
        step_ids: torch.Tensor,
        role_ids: torch.Tensor,
        fam_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute step and role logits for every position.

        Parameters
        ----------
        step_ids : ``[B, T]`` long, step ids.
        role_ids : ``[B, T]`` long, role ids (ROLE_PAD_ID at pads).
        fam_ids  : ``[B]`` long, one family id per sequence.
        key_padding_mask : optional ``[B, T]`` bool, True where padded (ignored by
            attention). Matches ``nn.TransformerEncoder`` convention.

        Returns
        -------
        (step_logits ``[B, T, V]``, role_logits ``[B, T, R]``).
        """
        B, T = step_ids.shape
        device = step_ids.device
        pos = torch.arange(T, device=device).clamp_max(self.max_len - 1)
        h = (
            self.tok(step_ids)
            + self.role(role_ids)
            + self.fam(fam_ids)[:, None, :]
            + self.pos(pos)[None, :, :]
        )
        # Bool causal mask (True = position is NOT allowed to attend), matching the
        # bool src_key_padding_mask convention so torch does not warn about mixed
        # mask dtypes. Position i may attend to j<=i.
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )
        h = self.body(h, mask=causal, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.step_head(h), self.role_head(h)


# ---------------------------------------------------------------------------
# Inference wrapper implementing the RANKER PROTOCOL
# ---------------------------------------------------------------------------
class NeuralRanker:
    """Wraps a trained :class:`ConstrainedRanker` to the ranker protocol.

    ``predict`` runs the model on ``[BOS] + prefix`` and reads the last-position
    step logits, masked to ``candset`` (or the full vocab when ``candset`` is
    empty). Grammar legality is applied by the *caller* (decode passes a
    symbolically-legal candset). ``predict_roles`` reads the last-position role
    logits. ``perplexity`` is the token-level perplexity of a full sequence under
    the next-step model (used by the optional anomaly residual).
    """

    def __init__(
        self,
        model: ConstrainedRanker,
        id_to_step: Sequence[str],
        step_to_id: Dict[str, int],
        family_to_id: Dict[str, int],
        roles: Sequence[str],
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.id_to_step = list(id_to_step)
        self.step_to_id = dict(step_to_id)
        self.family_to_id = dict(family_to_id)
        self.roles = list(roles)
        # Default family id when an unseen (OOD) family is requested: family 0.
        self._default_fam = 0
        self.ppl_p95: float = float("inf")  # set by callers if a threshold is fit

    # -- helpers -----------------------------------------------------------
    def _fam_id(self, family: str) -> int:
        return self.family_to_id.get(str(family).lower(), self._default_fam)

    def _encode_prefix(self, prefix: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``[BOS] + prefix`` into step/role id tensors of shape ``[1, T]``."""
        step_ids = [BOS_ID] + [self.step_to_id.get(s, UNK_ID) for s in prefix]
        role_ids = [ROLE_PAD_ID] + [role_idx(s) for s in prefix]
        st = torch.tensor([step_ids], dtype=torch.long, device=self.device)
        ro = torch.tensor([role_ids], dtype=torch.long, device=self.device)
        return st, ro

    @torch.no_grad()
    def _last_step_logits(self, prefix: Sequence[str], family: str) -> torch.Tensor:
        st, ro = self._encode_prefix(prefix)
        fam = torch.tensor([self._fam_id(family)], dtype=torch.long, device=self.device)
        step_logits, _ = self.model(st, ro, fam)
        return step_logits[0, -1]  # [V]

    @torch.no_grad()
    def _last_role_logits(self, prefix: Sequence[str], family: str) -> torch.Tensor:
        st, ro = self._encode_prefix(prefix)
        fam = torch.tensor([self._fam_id(family)], dtype=torch.long, device=self.device)
        _, role_logits = self.model(st, ro, fam)
        return role_logits[0, -1]  # [R]

    # -- protocol ----------------------------------------------------------
    def predict(self, prefix: List[str], family: str, candset) -> Dict[str, float]:
        """Probabilities over ``candset`` (full vocab if ``candset`` is empty)."""
        logits = self._last_step_logits(prefix, family)

        cands = list(candset) if candset else None
        if cands is None:
            # Full real vocabulary (exclude specials).
            ids = [i for i in range(len(self.id_to_step)) if i > UNK_ID]
            steps = [self.id_to_step[i] for i in ids]
        else:
            # Map candidate strings to ids; novel (OOD) strings get UNK logit.
            steps, ids = [], []
            for c in cands:
                steps.append(c)
                ids.append(self.step_to_id.get(c, UNK_ID))

        if not ids:
            return {}
        sel = logits[torch.tensor(ids, dtype=torch.long, device=self.device)]
        probs = torch.softmax(sel, dim=-1).tolist()
        return {s: float(p) for s, p in zip(steps, probs)}

    def predict_roles(self, prefix: List[str], family: str, top_r: int = 3) -> List[str]:
        """Top-``top_r`` next roles under the role head."""
        logits = self._last_role_logits(prefix, family)
        # Only consider real roles (0..NUM_ROLES-1), never the role-pad slot.
        real = logits[:NUM_ROLES]
        order = torch.argsort(real, descending=True).tolist()
        return [self.roles[i] for i in order[: max(1, top_r)]]

    @torch.no_grad()
    def perplexity(self, seq: Sequence[str]) -> float:
        """Token-level perplexity of a full sequence under the next-step model."""
        steps = list(seq)
        if not steps:
            return float("inf")
        family = "*"  # family-agnostic structural signal; falls back to default id
        step_ids = [BOS_ID] + [self.step_to_id.get(s, UNK_ID) for s in steps]
        role_ids = [ROLE_PAD_ID] + [role_idx(s) for s in steps]
        st = torch.tensor([step_ids], dtype=torch.long, device=self.device)
        ro = torch.tensor([role_ids], dtype=torch.long, device=self.device)
        fam = torch.tensor([self._fam_id(family)], dtype=torch.long, device=self.device)
        step_logits, _ = self.model(st, ro, fam)  # [1, T, V]
        logp = torch.log_softmax(step_logits[0], dim=-1)  # [T, V]
        # Predict position t+1 from position t: targets are steps[0..], inputs BOS..steps[-1].
        targets = [self.step_to_id.get(s, UNK_ID) for s in steps]
        nll = 0.0
        for t, tgt in enumerate(targets):
            nll += -float(logp[t, tgt])
        return math.exp(nll / len(targets))

    # -- persistence -------------------------------------------------------
    def save(self, ckpt_path: str) -> str:
        cfg = {
            "vocab": len(self.id_to_step),
            "n_roles": self.model.role.num_embeddings,
            "n_fam": self.model.fam.num_embeddings,
            "d": self.model.d,
            "layers": len(self.model.body.layers),
            "heads": self.model.body.layers[0].self_attn.num_heads,
            "max_len": self.model.max_len,
        }
        payload = {
            "model_config": cfg,
            "id_to_step": self.id_to_step,
            "family_to_id": self.family_to_id,
            "roles": self.roles,
            "state_dict": self.model.state_dict(),
            "ppl_p95": self.ppl_p95,
        }
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        torch.save(payload, ckpt_path)
        return ckpt_path


def load_ranker(ckpt: str, device: Optional[str] = None) -> NeuralRanker:
    """Reload a :class:`NeuralRanker` from a checkpoint written by training."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = payload["model_config"]
    model = ConstrainedRanker(
        vocab=cfg["vocab"], n_roles=cfg["n_roles"], n_fam=cfg["n_fam"],
        d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], max_len=cfg["max_len"],
    )
    model.load_state_dict(payload["state_dict"])
    id_to_step = payload["id_to_step"]
    step_to_id = {s: i for i, s in enumerate(id_to_step)}
    ranker = NeuralRanker(
        model, id_to_step, step_to_id, payload["family_to_id"], payload["roles"], device=dev,
    )
    ranker.ppl_p95 = float(payload.get("ppl_p95", float("inf")))
    return ranker


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _out_dir() -> Path:
    return Path(os.environ.get("NSPE_OUT", "models/neurosymbolic/outputs"))


def _build_examples(
    train_families: Sequence[str],
    step_to_id: Dict[str, int],
    family_to_id: Dict[str, int],
    smoke: bool,
) -> List[Tuple[List[int], List[int], List[int], int]]:
    """Build per-sequence (input_step, input_role, target_step, fam_id) lists.

    Inputs are ``[BOS] + steps`` and targets are ``steps + [EOS]`` (teacher
    forcing). Role inputs parallel the step inputs (ROLE_PAD_ID for BOS).
    """
    examples = []
    for fam in train_families:
        fam = fam.lower()
        seqs = load_family(fam)
        if smoke:
            seqs = seqs[:40]
        fid = family_to_id[fam]
        for seq in seqs:
            steps = list(seq)
            if not steps:
                continue
            in_steps = [BOS_ID] + [step_to_id.get(s, UNK_ID) for s in steps]
            in_roles = [ROLE_PAD_ID] + role_encode(steps)
            tgt_steps = [step_to_id.get(s, UNK_ID) for s in steps] + [EOS_ID]
            examples.append((in_steps, in_roles, tgt_steps, fid))
    return examples


def _role_targets_from_step_targets(
    tgt_steps: List[int], id_to_step: Sequence[str]
) -> List[int]:
    """Role id for each target step (EOS / specials -> ROLE_PAD_ID, ignored)."""
    out = []
    for sid in tgt_steps:
        if sid <= UNK_ID:  # PAD/BOS/EOS/UNK -> no meaningful role target
            out.append(ROLE_PAD_ID)
        else:
            out.append(role_idx(id_to_step[sid]))
    return out


def _collate(
    batch, id_to_step: Sequence[str], device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch and return tensors:
    (step_in, role_in, fam_ids, step_tgt, role_tgt, key_padding_mask)."""
    maxT = max(len(b[0]) for b in batch)
    step_in, role_in, step_tgt, role_tgt, fam_ids, kpm = [], [], [], [], [], []
    for in_steps, in_roles, tgt_steps, fid in batch:
        pad = maxT - len(in_steps)
        step_in.append(in_steps + [PAD_ID] * pad)
        role_in.append(in_roles + [ROLE_PAD_ID] * pad)
        step_tgt.append(tgt_steps + [PAD_ID] * pad)
        role_tgt.append(_role_targets_from_step_targets(tgt_steps, id_to_step) + [ROLE_PAD_ID] * pad)
        fam_ids.append(fid)
        kpm.append([False] * len(in_steps) + [True] * pad)
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)  # noqa: E731
    return (
        t(step_in), t(role_in), t(fam_ids),
        t(step_tgt), t(role_tgt),
        torch.tensor(kpm, dtype=torch.bool, device=device),
    )


def _build_valid_mask(
    examples, id_to_step: Sequence[str], step_to_id: Dict[str, int], cache_path: Path,
    smoke: bool,
) -> List[List[List[int]]]:
    """Per-example, per-position list of legal step ids (for the semantic loss).

    EXPENSIVE: one symbolic ``valid_next_set`` per position. Only called when
    ``sem_w>0`` or ``mask_train``. Cached to ``cache_path`` (json) and reused
    across epochs; under ``smoke`` the per-position scan is capped for speed.
    """
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Lazy import: the symbolic engine stays torch-free. legal_next_sets is the
    # fast incremental legal-set computer (~500x faster than per-candidate
    # re-validation; verified to match the official validator exactly), so the
    # full-corpus mask builds in seconds instead of tens of minutes — no cap needed.
    from nspe.grammar import legal_next_sets

    cand_vocab = [s for i, s in enumerate(id_to_step) if i > UNK_ID]
    masks: List[List[List[int]]] = []
    for in_steps, _in_roles, _tgt, _fid in examples:
        prefix_steps = [id_to_step[i] for i in in_steps[1:]]  # actual steps (drop BOS)
        sets = legal_next_sets(prefix_steps, cand_vocab)       # len == len(in_steps)
        per_pos: List[List[int]] = []
        for t in range(len(in_steps)):
            legal = set(sets[t]) if t < len(sets) else set()
            if t < len(prefix_steps):
                legal.add(prefix_steps[t])  # gold always legal (defensive)
            per_pos.append(sorted(step_to_id[s] for s in legal if s in step_to_id))
        masks.append(per_pos)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(masks, fh)
    return masks


def _mask_tensor_for_batch(
    batch_idx, mask_lists, maxT: int, vocab: int, device: str
) -> torch.Tensor:
    """Build a ``[B, T, V]`` bool legal mask for one batch from cached id-lists.

    An empty id-list at a position means "all legal" (True everywhere) so it
    contributes ~0 semantic loss.
    """
    B = len(batch_idx)
    mask = torch.zeros(B, maxT, vocab, dtype=torch.bool, device=device)
    for b, ex_i in enumerate(batch_idx):
        per_pos = mask_lists[ex_i]
        for t in range(min(maxT, len(per_pos))):
            ids = per_pos[t]
            if not ids:
                mask[b, t, :] = True
            else:
                mask[b, t, torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask


def train_ranker(
    train_families: List[str],
    config: Optional[Dict] = None,
    out_dir: Optional[str] = None,
    holdout: Optional[str] = None,
    device: Optional[str] = None,
    smoke: bool = False,
) -> Dict:
    """Train the constrained neural ranker as a next-step LM with role+family
    features. Loss = ``step_ce + role_w*role_ce + sem_w*semantic_loss``.

    Parameters
    ----------
    train_families : families to train on (vocab is built from these only, which
        is the OOD-correct choice for LoFO — the held-out family's steps are UNK).
    config   : hyper-parameters (see ``DEFAULT_CONFIG``); merged over defaults.
    out_dir  : output directory for the checkpoint + mask cache. Defaults to
        ``$NSPE_OUT`` or ``models/neurosymbolic/outputs``.
    holdout  : the held-out family name, recorded in metrics (not trained on).
    device   : ``'cuda'``/``'cpu'``; defaults to cuda if available.
    smoke    : tiny fast run (few seqs/steps) for CI/self-test.

    Returns
    -------
    dict with ``metrics`` (loss curve, param count, config, families) and
    ``ckpt_path``.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir) if out_dir else _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.get("seed", 0)))

    train_families = [f.lower() for f in train_families]
    id_to_step, step_to_id = build_step_index(tuple(train_families))
    family_to_id = {f: i for i, f in enumerate(train_families)}
    vocab = len(id_to_step)
    n_roles = NUM_ROLES + 1  # +1 for the role-pad slot
    n_fam = max(1, len(train_families))

    from nspe.roles import ROLES  # local import keeps module import order clean

    model = ConstrainedRanker(
        vocab=vocab, n_roles=n_roles, n_fam=n_fam,
        d=int(cfg["d_model"]), layers=int(cfg["layers"]), heads=int(cfg["heads"]),
        max_len=int(cfg["max_len"]),
    )
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(dev)

    examples = _build_examples(train_families, step_to_id, family_to_id, smoke)
    if not examples:
        raise RuntimeError("no training examples built")

    # Optional expensive legal-step mask (only when requested).
    sem_w = float(cfg.get("sem_w", 0.0))
    mask_train = bool(cfg.get("mask_train", False))
    mask_lists = None
    if sem_w > 0.0 or mask_train:
        tag = "+".join(train_families) + (f"-ho_{holdout}" if holdout else "")
        cache_path = out / f"validmask_{tag}{'_smoke' if smoke else ''}.json"
        mask_lists = _build_valid_mask(examples, id_to_step, step_to_id, cache_path, smoke)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                            weight_decay=float(cfg.get("weight_decay", 0.0)))
    warmup = int(cfg.get("warmup", 0))

    def lr_at(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(cfg["lr"]) * (step + 1) / warmup
        return float(cfg["lr"])

    batch = int(cfg["batch"])
    max_steps = int(cfg["steps"])
    epochs = 1 if smoke else int(cfg.get("epochs", 1))
    g = torch.Generator().manual_seed(int(cfg.get("seed", 0)))

    model.train()
    loss_curve: List[float] = []
    step = 0
    t0 = time.time()
    done = False
    for _epoch in range(epochs):
        if done:
            break
        perm = torch.randperm(len(examples), generator=g).tolist()
        for bstart in range(0, len(perm), batch):
            batch_idx = perm[bstart:bstart + batch]
            sub = [examples[i] for i in batch_idx]
            step_in, role_in, fam_ids, step_tgt, role_tgt, kpm = _collate(sub, id_to_step, dev)
            valid_mask = None
            if mask_lists is not None and sem_w > 0.0:
                valid_mask = _mask_tensor_for_batch(
                    batch_idx, mask_lists, step_in.size(1), vocab, dev)

            for grp in opt.param_groups:
                grp["lr"] = lr_at(step)
            opt.zero_grad()
            step_logits, role_logits = model(step_in, role_in, fam_ids, key_padding_mask=kpm)
            loss, comp = total_loss(
                step_logits, role_logits, step_tgt, role_tgt, PAD_ID,
                valid_id_mask=valid_mask, role_w=float(cfg["role_w"]), sem_w=sem_w,
                role_pad_id=ROLE_PAD_ID,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_curve.append(comp["total"])
            step += 1
            if step >= max_steps:
                done = True
                break

    # Final-batch train accuracy as a cheap sanity metric.
    model.eval()
    with torch.no_grad():
        step_logits, _ = model(step_in, role_in, fam_ids, key_padding_mask=kpm)
        pred = step_logits.argmax(-1)
        valid_pos = step_tgt != PAD_ID
        acc = float((pred[valid_pos] == step_tgt[valid_pos]).float().mean())

    ranker = NeuralRanker(model, id_to_step, step_to_id, family_to_id, list(ROLES), device=dev)
    tag = "+".join(train_families) + (f"-ho_{holdout}" if holdout else "")
    ckpt_path = str(out / f"ranker_{tag}{'_smoke' if smoke else ''}.pt")
    ranker.save(ckpt_path)

    metrics = {
        "n_params": n_params,
        "train_families": train_families,
        "holdout": holdout,
        "device": dev,
        "steps_run": step,
        "epochs": epochs,
        "n_examples": len(examples),
        "final_loss": loss_curve[-1] if loss_curve else None,
        "first_loss": loss_curve[0] if loss_curve else None,
        "final_batch_step_acc": acc,
        "wall_sec": round(time.time() - t0, 2),
        "sem_w": sem_w,
        "mask_train": mask_train,
        "config": cfg,
    }
    return {"metrics": metrics, "ckpt_path": ckpt_path}


# ---------------------------------------------------------------------------
# Self-test (CPU)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    from nspe.roles import ROLES

    print("== ConstrainedRanker forward + total_loss + one optim step ==")
    torch.manual_seed(0)
    vocab, n_roles, n_fam = 50, NUM_ROLES + 1, 3
    m = ConstrainedRanker(vocab, n_roles, n_fam, d=64, layers=2, heads=2, max_len=64)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"param count (d=64,layers=2): {n_params:,}")

    B, T = 4, 10
    step_ids = torch.randint(0, vocab, (B, T))
    role_ids = torch.randint(0, n_roles, (B, T))
    fam_ids = torch.randint(0, n_fam, (B,))
    kpm = torch.zeros(B, T, dtype=torch.bool)
    kpm[0, -2:] = True  # pad last two positions of seq 0
    sl, rl = m(step_ids, role_ids, fam_ids, key_padding_mask=kpm)
    print("step_logits", tuple(sl.shape), "role_logits", tuple(rl.shape))
    assert sl.shape == (B, T, vocab) and rl.shape == (B, T, n_roles)

    step_tgt = torch.randint(1, vocab, (B, T))
    role_tgt = torch.randint(0, NUM_ROLES, (B, T))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss, comp = total_loss(sl, rl, step_tgt, role_tgt, PAD_ID, role_w=0.3, sem_w=0.0,
                            role_pad_id=ROLE_PAD_ID)
    opt.zero_grad(); loss.backward(); opt.step()
    print("loss value:", round(comp["total"], 4), comp)
    assert torch.isfinite(loss)

    print("\n== train_ranker(['mosfet'], smoke=True) on CPU ==")
    tmp = tempfile.mkdtemp(prefix="nspe_model_")
    res = train_ranker(
        ["mosfet"],
        {"d_model": 64, "layers": 2, "heads": 2, "steps": 30, "batch": 8, "lr": 1e-3},
        out_dir=tmp, smoke=True, device="cpu",
    )
    print("metrics:", {k: res["metrics"][k] for k in
                       ("n_params", "steps_run", "n_examples", "first_loss",
                        "final_loss", "final_batch_step_acc", "wall_sec")})
    ckpt = res["ckpt_path"]
    print("ckpt_path:", ckpt)
    assert os.path.exists(ckpt)
    assert res["metrics"]["n_params"] < 5_000_000, "model must stay small (<5M)"

    print("\n== load_ranker + .predict / .predict_roles / .perplexity ==")
    rk = load_ranker(ckpt, device="cpu")
    seq = list(load_family("mosfet")[0])
    prefix = seq[:8]
    # Restricted candset (a handful of real steps + the gold).
    candset = set(seq[8:14]) | {seq[8]}
    probs = rk.predict(prefix, "mosfet", candset)
    print("predict over candset (|c|=%d):" % len(candset),
          {k: round(v, 3) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]})
    s = sum(probs.values())
    print("sum probs:", round(s, 6))
    assert abs(s - 1.0) < 1e-4
    # Full-vocab predict also sums to ~1.
    full = rk.predict(prefix, "mosfet", set())
    assert abs(sum(full.values()) - 1.0) < 1e-4
    roles = rk.predict_roles(prefix, "mosfet", top_r=3)
    print("predict_roles top-3:", roles)
    assert len(roles) == 3 and all(r in ROLES for r in roles)
    ppl = rk.perplexity(seq)
    print("perplexity(full seq):", round(ppl, 2))
    assert ppl > 0 and math.isfinite(ppl)

    print("\nSELF-TEST PASSED")
