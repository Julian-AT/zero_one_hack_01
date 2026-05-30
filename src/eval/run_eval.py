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

from src.data.canonicalize import canonicalize_sequence
from src.data.corrupt import corrupt_random
from src.data.load import load_all_families
from src.eval.metrics import normalized_edit_distance
from src.eval.predict import (
    anomaly_ensemble,
    complete_sequence,
    load_model,
    topk_next_step,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s", stream=sys.stdout
)
logger = logging.getLogger("eval")


def token_accuracy(pred: list[str], ref: list[str]) -> float:
    """Fraction of positions (up to min length) where pred == ref.

    Matches eval_metrics.py:token_accuracy."""
    n = min(len(pred), len(ref))
    if n == 0:
        return 0.0
    return sum(p == r for p, r in zip(pred, ref, strict=False)) / n


# Major-process-block taxonomy — duplicated from eval_metrics.py:_major_block
# so we report the same Block-level Accuracy the organizers' script computes.
def _major_block(step: str) -> str:
    s = step.upper()
    if "LITHO" in s or s.startswith("SPIN COAT PHOTORESIST") or "MASK LEVEL" in s:
        return "LITHO"
    if "ETCH" in s or s.startswith("OPEN PAD WINDOW"):
        return "ETCH"
    if "IMPLANT" in s or "ANNEAL" in s or "DIFFUSION" in s:
        return "DOPING_THERMAL"
    if s.startswith("DEPOSIT") or "OXIDATION" in s or "GROWTH" in s:
        return "DEPOSITION"
    if s.startswith("CMP") or "PLANAR" in s:
        return "PLANARIZATION"
    if "VIA" in s:
        return "VIA"
    if "PASSIVATION" in s:
        return "PASSIVATION"
    if "BACKSIDE" in s or "GRIND" in s:
        return "BACKSIDE"
    if "TEST" in s or "MEASURE" in s or "INSPECT" in s or "ANALYSIS" in s:
        return "METROLOGY_TEST"
    if "LOT" in s or "RELEASE" in s or "SHIP" in s:
        return "LOGISTICS"
    return "OTHER"


def _block_signature(seq: list[str]) -> list[str]:
    sig: list[str] = []
    prev: str | None = None
    for step in seq:
        b = _major_block(step)
        if b != prev:
            sig.append(b)
            prev = b
    return sig


def block_level_accuracy(pred: list[str], ref: list[str]) -> float:
    """Position-wise accuracy over the coarse 10-block signatures.

    Matches eval_metrics.py:block_level_accuracy."""
    return token_accuracy(_block_signature(pred), _block_signature(ref))


def eval_nextstep_and_completion(
    lm,
    examples,
    frac: float,
    grammar: bool,
    canonicalize: bool,
    max_examples: int = 40,
    do_completion: bool = True,
    max_completion_steps: int = 60,
) -> dict:
    n = c1 = c3 = c5 = 0
    rr_sum = 0.0
    em = 0
    ned_sum = 0.0
    tacc_sum = 0.0
    bacc_sum = 0.0
    rng = random.Random(0)
    sample = rng.sample(examples, min(max_examples, len(examples)))
    for ex in sample:
        s = ex.steps
        if len(s) < 6:
            continue
        cut = max(2, int(len(s) * frac))
        if cut >= len(s):
            continue
        prefix = s[:cut]
        gold_next = s[cut]
        ranked = topk_next_step(lm, ex.family, prefix, k=5, grammar=grammar)
        n += 1
        if canonicalize:
            ranked = canonicalize_sequence(ranked)
            gold_next = canonicalize_sequence([gold_next])[0]
        if ranked and ranked[0] == gold_next:
            c1 += 1
        if gold_next in ranked[:3]:
            c3 += 1
        if gold_next in ranked[:5]:
            c5 += 1
        if gold_next in ranked:
            rr_sum += 1.0 / (ranked.index(gold_next) + 1)
        if do_completion:
            gold = s[cut:]
            cap = min(max_completion_steps, len(gold) + 5)
            pred = complete_sequence(lm, ex.family, prefix, max_len=cap, grammar=grammar)
            if canonicalize:
                pred = canonicalize_sequence(pred)
                gold = canonicalize_sequence(gold)
            if pred == gold:
                em += 1
            ned_sum += normalized_edit_distance(pred, gold)
            tacc_sum += token_accuracy(pred, gold)
            bacc_sum += block_level_accuracy(pred, gold)
    return {
        "n": n,
        "top1_at_cut": c1 / n if n else 0,
        "top3_at_cut": c3 / n if n else 0,
        "top5_at_cut": c5 / n if n else 0,
        "mrr_at_cut": rr_sum / n if n else 0,
        "completion_exact_match": em / n if n else 0,
        "completion_ned": ned_sum / n if n else 0,
        "completion_token_acc": tacc_sum / n if n else 0,
        "completion_block_acc": bacc_sum / n if n else 0,
    }


def _anomaly_metrics_from_items(lm, items) -> dict:
    """Score a list of (seq, gold_is_valid, gold_rule, family) tuples.

    Adds ROC-AUC computed from the ensemble's SCORE field — the brief's
    rubric for Task 3 includes AUC, which is wasted unless SCORE varies."""
    tp = fp = tn = fn = 0
    rule_correct = rule_total = 0
    scores: list[float] = []
    golds: list[int] = []
    for seq, gold, gold_rule, fam in items:
        result = anomaly_ensemble(lm, fam, seq)
        pred = result["IS_VALID"]
        pred_rule = result["PREDICTED_RULE"]
        scores.append(float(result["SCORE"]))
        golds.append(int(gold))
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 1 and pred == 0:
            fp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fn += 1
        if gold == 0 and gold_rule is not None:
            rule_total += 1
            if pred_rule == gold_rule:
                rule_correct += 1
    n = len(items)
    acc = (tp + tn) / n if n else 0
    # Convention matches eval_metrics.py: "invalid" is the positive class.
    # In our internal variables we have invalid_tp = tn here (and vice versa),
    # so we expose BOTH framings.
    precision_valid = tp / (tp + fp) if (tp + fp) else 0
    recall_valid = tp / (tp + fn) if (tp + fn) else 0
    f1_valid = (
        (2 * precision_valid * recall_valid / (precision_valid + recall_valid))
        if (precision_valid + recall_valid)
        else 0
    )
    # Invalid-as-positive (official Task 3 reporting class)
    inv_tp = tn  # correctly detected invalid
    inv_fp = fn  # predicted invalid but actually valid
    inv_fn = fp  # missed invalid (said valid)
    precision_invalid = inv_tp / (inv_tp + inv_fp) if (inv_tp + inv_fp) else 0
    recall_invalid = inv_tp / (inv_tp + inv_fn) if (inv_tp + inv_fn) else 0
    f1_invalid = (
        (2 * precision_invalid * recall_invalid / (precision_invalid + recall_invalid))
        if (precision_invalid + recall_invalid)
        else 0
    )
    pos = [s for s, g in zip(scores, golds, strict=True) if g == 1]
    neg = [s for s, g in zip(scores, golds, strict=True) if g == 0]
    auc = _roc_auc(pos, neg)
    return {
        "n": n,
        "binary_accuracy": acc,
        "precision_valid": precision_valid,
        "recall_valid": recall_valid,
        "f1_valid": f1_valid,
        "precision_invalid": precision_invalid,
        "recall_invalid": recall_invalid,
        "f1_invalid": f1_invalid,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "roc_auc": auc,
        "rule_attribution_accuracy": (rule_correct / rule_total) if rule_total else 0,
        "rule_attribution_n": rule_total,
    }


def _roc_auc(pos: list[float], neg: list[float]) -> float:
    """AUC = P(score(pos) > score(neg)). Rank-based, ties broken by 0.5."""
    if not pos or not neg:
        return 0.0
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def eval_anomaly(lm, examples, corrupt_frac: float = 0.5, max_examples: int = 200) -> dict:
    """Build a mixed valid/corrupted held-out, score with the ensemble.

    Computes overall metrics plus a per-family breakdown so LoFO runs can
    isolate the held-out family's OOD anomaly numbers."""
    rng = random.Random(0)
    sample = rng.sample(examples, min(max_examples, len(examples)))
    items: list[tuple[list[str], int, str | None, str]] = []
    for ex in sample:
        if rng.random() < corrupt_frac:
            c = corrupt_random(list(ex.steps), rng, verify=True)
            if c is not None:
                items.append((c.corrupted_steps, 0, c.rule, ex.family))
                continue
        items.append((ex.steps, 1, None, ex.family))
    overall = _anomaly_metrics_from_items(lm, items)
    per_family: dict[str, dict] = {}
    by_fam: dict[str, list] = {}
    for it in items:
        by_fam.setdefault(it[3], []).append(it)
    for fam, fam_items in by_fam.items():
        per_family[fam] = _anomaly_metrics_from_items(lm, fam_items)
    overall["per_family"] = per_family
    return overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-grammar", action="store_true")
    parser.add_argument("--canonicalize", action="store_true")
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--max-completion-steps", type=int, default=60)
    parser.add_argument("--skip-completion", action="store_true")
    parser.add_argument(
        "--held-out-family",
        default=None,
        help="Tag the metrics with the held-out family for LoFO runs. "
        "Pure label — the eval already runs on all 3 families.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {args.checkpoint}…")
    lm = load_model(Path(args.checkpoint))
    logger.info(
        f"  arch={lm.cfg['arch']}  tok={lm.cfg['tokenization']['mode']}  "
        f"params={sum(p.numel() for p in lm.model.parameters()):,}"
    )

    # Held-out: last 100 sequences per family.
    examples = load_all_families()
    held: list = []
    by_fam: dict[str, list] = {}
    for ex in examples:
        by_fam.setdefault(ex.family, []).append(ex)
    for _fam, lst in by_fam.items():
        held.extend(lst[-100:])

    use_grammar = not args.no_grammar
    results: dict = {
        "checkpoint": args.checkpoint,
        "config": lm.cfg,
        "grammar": use_grammar,
        "canonicalize": args.canonicalize,
        "held_out_family": args.held_out_family,
    }

    logger.info("Eval: next-step + completion @ frac=0.6 + 0.8 (held-out per family)")
    results["per_family"] = {}
    for fam, lst in by_fam.items():
        per_frac = {}
        for frac in (0.6, 0.8):
            t0 = time.time()
            m = eval_nextstep_and_completion(
                lm,
                lst[-100:],
                frac=frac,
                grammar=use_grammar,
                canonicalize=args.canonicalize,
                max_examples=args.max_examples,
                do_completion=not args.skip_completion,
                max_completion_steps=args.max_completion_steps,
            )
            m["wall_seconds"] = round(time.time() - t0, 1)
            logger.info(
                f"  {fam.upper()}  frac={frac}: "
                f"Top1={m['top1_at_cut']:.4f}  Top5={m['top5_at_cut']:.4f}  "
                f"EM={m['completion_exact_match']:.4f}  NED={m['completion_ned']:.4f}  "
                f"({m['wall_seconds']}s)"
            )
            per_frac[str(frac)] = m
        results["per_family"][fam] = per_frac

    logger.info("Eval: anomaly detection (mixed held-out, 50% corrupted)")
    # Scale anomaly sample with families so each family gets ~max_examples items.
    a = eval_anomaly(lm, held, corrupt_frac=0.5, max_examples=max(args.max_examples * 3, len(held)))
    logger.info(
        f"  overall acc={a['binary_accuracy']:.4f}  "
        f"AUC={a['roc_auc']:.4f}  "
        f"TP/FP/TN/FN={a['tp']}/{a['fp']}/{a['tn']}/{a['fn']}  "
        f"rule_attrib={a['rule_attribution_accuracy']:.4f}"
    )
    for fam, fam_a in a.get("per_family", {}).items():
        tag = "(HELD-OUT)" if fam == args.held_out_family else ""
        logger.info(
            f"    {fam.upper():>7} {tag:>11} n={fam_a['n']:>3}  "
            f"acc={fam_a['binary_accuracy']:.4f}  "
            f"AUC={fam_a['roc_auc']:.4f}  "
            f"rule_attrib={fam_a['rule_attribution_accuracy']:.4f}"
        )
    results["anomaly"] = a

    # Save
    with (out / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    md = ["# Eval — " + args.checkpoint, ""]
    md.append("## Next-step + completion (held-out per family)")
    md.append(
        "| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |"
    )
    md.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for fam, fracs in results["per_family"].items():
        for frac, m in fracs.items():
            md.append(
                f"| {fam.upper()} | {frac} | "
                f"{m['top1_at_cut']:.4f} | {m['top3_at_cut']:.4f} | "
                f"{m['top5_at_cut']:.4f} | {m['mrr_at_cut']:.4f} | "
                f"{m['completion_exact_match']:.4f} | "
                f"{m['completion_ned']:.4f} | "
                f"{m.get('completion_token_acc', 0):.4f} | "
                f"{m.get('completion_block_acc', 0):.4f} |"
            )
    md.append("")
    md.append("## Anomaly detection")
    md.append(f"- n = {a['n']}, binary acc = {a['binary_accuracy']:.4f}, AUC = {a['roc_auc']:.4f}")
    md.append(
        f"- invalid class (Task-3 reporting): "
        f"P = {a.get('precision_invalid', 0):.4f}, "
        f"R = {a.get('recall_invalid', 0):.4f}, "
        f"F1 = {a.get('f1_invalid', 0):.4f}"
    )
    md.append(
        f"- valid class: P = {a['precision_valid']:.4f}, "
        f"R = {a['recall_valid']:.4f}, "
        f"F1 = {a.get('f1_valid', 0):.4f}"
    )
    md.append("- confusion matrix (invalid = positive):")
    md.append("    | | pred invalid | pred valid |")
    md.append("    |---|--:|--:|")
    md.append(f"    | actual invalid | {a['tn']} | {a['fp']} |")
    md.append(f"    | actual valid   | {a['fn']} | {a['tp']} |")
    md.append(
        f"- rule attribution accuracy = "
        f"{a['rule_attribution_accuracy']:.4f} (n_invalid={a['rule_attribution_n']})"
    )
    if a.get("per_family"):
        md.append("")
        md.append(
            "### Per-family breakdown"
            + (f" (held-out: **{args.held_out_family.upper()}**)" if args.held_out_family else "")
        )
        md.append("| family | n | acc | AUC | rule_attrib |")
        md.append("|---|--:|--:|--:|--:|")
        for fam, fam_a in a["per_family"].items():
            star = " ⭐" if fam == args.held_out_family else ""
            md.append(
                f"| {fam.upper()}{star} | {fam_a['n']} | "
                f"{fam_a['binary_accuracy']:.4f} | {fam_a['roc_auc']:.4f} | "
                f"{fam_a['rule_attribution_accuracy']:.4f} |"
            )
    (out / "metrics.md").write_text("\n".join(md))
    logger.info(f"Wrote {out / 'metrics.json'} and {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
