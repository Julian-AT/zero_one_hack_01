"""End-to-end evaluation orchestrator.

Loads a trained checkpoint and runs:
  - Held-out next-step + completion metrics (Top-K, MRR, Exact Match, NED)
  - Anomaly detection on a held-out mix of valid + corrupted sequences
  - Optional LoFO breakdown when given --lofo

Outputs a JSON + Markdown report under `extras/results/eval/<run_name>/`.

Usage on Leonardo:
  pixi run python -m src.eval.run_eval \
      --checkpoint extras/checkpoints/cell1-transformer_medium-compositional/final.pt \
      --output-dir extras/results/eval/cell1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

from src.data.canonicalize import canonicalize_sequence
from src.data.corrupt import corrupt_random
from src.data.load import load_all_families
from src.data.validator import RULE_IDS, validate_sequence
from src.eval.predict import (
    anomaly_ensemble,
    complete_sequence,
    load_model,
    topk_next_step,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("eval")


def normalized_edit_distance(a: list[str], b: list[str]) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0: return 0.0
    if la == 0 or lb == 0: return 1.0
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb] / max(la, lb)


def eval_nextstep_and_completion(lm, examples, frac: float,
                                    grammar: bool, canonicalize: bool,
                                    max_examples: int = 200) -> dict:
    n = c1 = c3 = c5 = 0
    rr_sum = 0.0
    em = 0
    ned_sum = 0.0
    rng = random.Random(0)
    sample = rng.sample(examples, min(max_examples, len(examples)))
    for ex in sample:
        s = ex.steps
        if len(s) < 6: continue
        cut = max(2, int(len(s) * frac))
        if cut >= len(s): continue
        prefix = s[:cut]
        gold_next = s[cut]
        ranked = topk_next_step(lm, ex.family, prefix, k=5, grammar=grammar)
        n += 1
        if canonicalize:
            ranked = canonicalize_sequence(ranked)
            gold_next = canonicalize_sequence([gold_next])[0]
        if ranked and ranked[0] == gold_next: c1 += 1
        if gold_next in ranked[:3]: c3 += 1
        if gold_next in ranked[:5]: c5 += 1
        if gold_next in ranked:
            rr_sum += 1.0 / (ranked.index(gold_next) + 1)
        # Completion
        gold = s[cut:]
        pred = complete_sequence(lm, ex.family, prefix,
                                  max_len=len(gold) + 20, grammar=grammar)
        if canonicalize:
            pred = canonicalize_sequence(pred)
            gold = canonicalize_sequence(gold)
        if pred == gold: em += 1
        ned_sum += normalized_edit_distance(pred, gold)
    return {
        "n": n,
        "top1_at_cut": c1 / n if n else 0,
        "top3_at_cut": c3 / n if n else 0,
        "top5_at_cut": c5 / n if n else 0,
        "mrr_at_cut":  rr_sum / n if n else 0,
        "completion_exact_match": em / n if n else 0,
        "completion_ned":         ned_sum / n if n else 0,
    }


def eval_anomaly(lm, examples, corrupt_frac: float = 0.5,
                  max_examples: int = 200) -> dict:
    """Build a mixed valid/corrupted held-out, score with the ensemble."""
    rng = random.Random(0)
    sample = rng.sample(examples, min(max_examples, len(examples)))
    items: list[tuple[list[str], int, str | None, str]] = []  # (seq, gold_is_valid, gold_rule, family)
    for ex in sample:
        if rng.random() < corrupt_frac:
            c = corrupt_random(list(ex.steps), rng, verify=True)
            if c is not None:
                items.append((c.corrupted_steps, 0, c.rule, ex.family))
                continue
        items.append((ex.steps, 1, None, ex.family))
    tp = fp = tn = fn = 0
    rule_correct = rule_total = 0
    for seq, gold, gold_rule, fam in items:
        result = anomaly_ensemble(lm, fam, seq)
        pred = result["IS_VALID"]
        pred_rule = result["PREDICTED_RULE"]
        if gold == 1 and pred == 1: tp += 1
        elif gold == 1 and pred == 0: fp += 1
        elif gold == 0 and pred == 0: tn += 1
        elif gold == 0 and pred == 1: fn += 1
        if gold == 0 and gold_rule is not None:
            rule_total += 1
            if pred_rule == gold_rule:
                rule_correct += 1
    n = len(items)
    acc = (tp + tn) / n if n else 0
    precision_valid = tp / (tp + fp) if (tp + fp) else 0  # P(true_valid | predicted_valid)
    recall_valid    = tp / (tp + fn) if (tp + fn) else 0
    return {
        "n": n,
        "binary_accuracy": acc,
        "precision_valid": precision_valid,
        "recall_valid": recall_valid,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "rule_attribution_accuracy": (rule_correct / rule_total) if rule_total else 0,
        "rule_attribution_n": rule_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-grammar", action="store_true")
    parser.add_argument("--canonicalize", action="store_true")
    parser.add_argument("--max-examples", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.checkpoint}…")
    lm = load_model(Path(args.checkpoint))
    logger.info(f"  arch={lm.cfg['arch']}  tok={lm.cfg['tokenization']['mode']}  "
                f"params={sum(p.numel() for p in lm.model.parameters()):,}")

    # Held-out: last 100 sequences per family.
    examples = load_all_families()
    held: list = []
    by_fam: dict[str, list] = {}
    for ex in examples:
        by_fam.setdefault(ex.family, []).append(ex)
    for fam, lst in by_fam.items():
        held.extend(lst[-100:])

    use_grammar = not args.no_grammar
    results: dict = {
        "checkpoint": args.checkpoint,
        "config": lm.cfg,
        "grammar": use_grammar,
        "canonicalize": args.canonicalize,
    }

    logger.info("Eval: next-step + completion @ frac=0.6 + 0.8 (held-out per family)")
    results["per_family"] = {}
    for fam, lst in by_fam.items():
        per_frac = {}
        for frac in (0.6, 0.8):
            t0 = time.time()
            m = eval_nextstep_and_completion(lm, lst[-100:], frac=frac,
                                               grammar=use_grammar,
                                               canonicalize=args.canonicalize,
                                               max_examples=args.max_examples)
            m["wall_seconds"] = round(time.time() - t0, 1)
            logger.info(f"  {fam.upper()}  frac={frac}: "
                        f"Top1={m['top1_at_cut']:.4f}  Top5={m['top5_at_cut']:.4f}  "
                        f"EM={m['completion_exact_match']:.4f}  NED={m['completion_ned']:.4f}  "
                        f"({m['wall_seconds']}s)")
            per_frac[str(frac)] = m
        results["per_family"][fam] = per_frac

    logger.info("Eval: anomaly detection (mixed held-out, 50% corrupted)")
    a = eval_anomaly(lm, held, corrupt_frac=0.5, max_examples=args.max_examples)
    logger.info(f"  acc={a['binary_accuracy']:.4f}  "
                f"TP/FP/TN/FN={a['tp']}/{a['fp']}/{a['tn']}/{a['fn']}  "
                f"rule_attrib={a['rule_attribution_accuracy']:.4f}")
    results["anomaly"] = a

    # Save
    with (out / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    md = ["# Eval — " + args.checkpoint, ""]
    md.append("## Next-step + completion (held-out per family)")
    md.append("| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |")
    md.append("|---|---|--:|--:|--:|--:|")
    for fam, fracs in results["per_family"].items():
        for frac, m in fracs.items():
            md.append(f"| {fam.upper()} | {frac} | {m['top1_at_cut']:.4f} | "
                      f"{m['top5_at_cut']:.4f} | {m['completion_exact_match']:.4f} | "
                      f"{m['completion_ned']:.4f} |")
    md.append("")
    md.append("## Anomaly detection")
    md.append(f"- n = {a['n']}, binary acc = {a['binary_accuracy']:.4f}")
    md.append(f"- precision(valid) = {a['precision_valid']:.4f}, "
              f"recall(valid) = {a['recall_valid']:.4f}")
    md.append(f"- TP/FP/TN/FN = {a['tp']}/{a['fp']}/{a['tn']}/{a['fn']}")
    md.append(f"- rule attribution accuracy = "
              f"{a['rule_attribution_accuracy']:.4f} (n_invalid={a['rule_attribution_n']})")
    (out / "metrics.md").write_text("\n".join(md))
    logger.info(f"Wrote {out / 'metrics.json'} and {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
