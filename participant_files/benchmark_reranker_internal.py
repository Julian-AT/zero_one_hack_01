#!/usr/bin/env python3
"""
Benchmark model-only vs rule-aware reranked next-step predictions
on the internal labeled next_step_prediction.csv test split.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NEXT_STEP_DATA = ROOT / "tracks" / "industrial-infineon" / "data" / "task_datasets_v1" / "next_step_prediction.csv"
PRED_MOD_PATH = ROOT / "participant_files" / "make_eval_predictions.py"
RERANK_MOD_PATH = ROOT / "participant_files" / "rerank_nextstep_with_rules.py"

OUT_DIR = ROOT / "participant_files" / "predictions"
OUT_CSV = OUT_DIR / "internal_reranker_benchmark_examples.csv"
OUT_MD = OUT_DIR / "internal_reranker_benchmark_report.md"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pred_mod = load_module(PRED_MOD_PATH, "make_eval_predictions_mod")
rerank_mod = load_module(RERANK_MOD_PATH, "rerank_nextstep_mod")


def norm(x: str) -> str:
    return str(x or "").strip().upper()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def reciprocal_rank(truth: str, ranks: list[str]) -> float:
    truth = norm(truth)
    ranks = [norm(r) for r in ranks]
    if truth in ranks:
        return 1.0 / (ranks.index(truth) + 1)
    return 0.0


def metric_summary(rows, key_prefix: str):
    n = len(rows)
    if n == 0:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "mrr": 0.0,
        }

    return {
        "top1": sum(r[f"{key_prefix}_hit1"] for r in rows) / n,
        "top3": sum(r[f"{key_prefix}_hit3"] for r in rows) / n,
        "top5": sum(r[f"{key_prefix}_hit5"] for r in rows) / n,
        "mrr": sum(r[f"{key_prefix}_rr"] for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not NEXT_STEP_DATA.exists():
        raise FileNotFoundError(NEXT_STEP_DATA)

    print(f"Loading labeled next-step data: {NEXT_STEP_DATA}")
    all_rows = read_rows(NEXT_STEP_DATA)
    rows = [r for r in all_rows if r.get("SPLIT", "").strip().lower() == args.split]

    print(f"Rows in split={args.split}: {len(rows):,}")

    rng = random.Random(args.seed)
    if args.max_examples and len(rows) > args.max_examples:
        rows = rng.sample(rows, args.max_examples)

    print(f"Benchmark examples: {len(rows):,}")

    print("Loading model predictor...")
    predictor = pred_mod.HybridPredictor(pred_mod.CHECKPOINT, pred_mod.VOCAB_JSON)

    print("Building valid n-gram counts...")
    valid_counts = rerank_mod.build_valid_ngram_counts(rerank_mod.VALID_TRAIN)

    print("Building invalid-context penalties...")
    bad_counts = rerank_mod.build_invalid_bad_context_counts([
        rerank_mod.EASY_INVALID,
        rerank_mod.HARD_INVALID,
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    changed_top1 = 0
    changed_any_order = 0

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXAMPLE_ID",
            "FAMILY",
            "TRUTH",
            "MODEL_RANK_1",
            "MODEL_RANK_2",
            "MODEL_RANK_3",
            "MODEL_RANK_4",
            "MODEL_RANK_5",
            "RERANK_RANK_1",
            "RERANK_RANK_2",
            "RERANK_RANK_3",
            "RERANK_RANK_4",
            "RERANK_RANK_5",
            "MODEL_HIT1",
            "RERANK_HIT1",
            "MODEL_RR",
            "RERANK_RR",
        ])

        for i, r in enumerate(rows, start=1):
            eid = r["EXAMPLE_ID"]
            family = r["FAMILY"]
            partial = pred_mod.split_steps(r["PREFIX_CONTEXT"])
            truth = norm(r["NEXT_STEP"])

            model_ranks = [norm(x) for x in predictor.topk_next(family, partial, k=5)]

            scored = []
            for rank_idx, cand in enumerate(model_ranks):
                score, _details = rerank_mod.score_candidate(
                    family=family,
                    partial=partial,
                    candidate=cand,
                    rank_index=rank_idx,
                    valid_counts=valid_counts,
                    bad_counts=bad_counts,
                )
                scored.append((score, cand))

            scored.sort(key=lambda x: x[0], reverse=True)
            rerank_ranks = [cand for _, cand in scored]

            model_hit1 = int(model_ranks[0] == truth)
            rerank_hit1 = int(rerank_ranks[0] == truth)

            model_hit3 = int(truth in model_ranks[:3])
            rerank_hit3 = int(truth in rerank_ranks[:3])

            model_hit5 = int(truth in model_ranks[:5])
            rerank_hit5 = int(truth in rerank_ranks[:5])

            model_rr = reciprocal_rank(truth, model_ranks)
            rerank_rr = reciprocal_rank(truth, rerank_ranks)

            if model_ranks[0] != rerank_ranks[0]:
                changed_top1 += 1
            if model_ranks != rerank_ranks:
                changed_any_order += 1

            result = {
                "model_hit1": model_hit1,
                "rerank_hit1": rerank_hit1,
                "model_hit3": model_hit3,
                "rerank_hit3": rerank_hit3,
                "model_hit5": model_hit5,
                "rerank_hit5": rerank_hit5,
                "model_rr": model_rr,
                "rerank_rr": rerank_rr,
            }
            results.append(result)

            writer.writerow([
                eid,
                family,
                truth,
                *(model_ranks + [""] * 5)[:5],
                *(rerank_ranks + [""] * 5)[:5],
                model_hit1,
                rerank_hit1,
                f"{model_rr:.4f}",
                f"{rerank_rr:.4f}",
            ])

            if i % 1000 == 0:
                print(f"Processed {i:,}/{len(rows):,}")

    model = metric_summary(results, "model")
    rerank = metric_summary(results, "rerank")

    delta_top1 = rerank["top1"] - model["top1"]
    delta_top3 = rerank["top3"] - model["top3"]
    delta_top5 = rerank["top5"] - model["top5"]
    delta_mrr = rerank["mrr"] - model["mrr"]

    report = f"""# Internal Reranker Benchmark

Evaluated on internal labeled `{args.split}` split from:

`{NEXT_STEP_DATA}`

## Summary

| Metric | Model only | Reranked | Delta |
|---|---:|---:|---:|
| Top-1 Accuracy | {model["top1"]:.4f} | {rerank["top1"]:.4f} | {delta_top1:+.4f} |
| Top-3 Accuracy | {model["top3"]:.4f} | {rerank["top3"]:.4f} | {delta_top3:+.4f} |
| Top-5 Accuracy | {model["top5"]:.4f} | {rerank["top5"]:.4f} | {delta_top5:+.4f} |
| MRR | {model["mrr"]:.4f} | {rerank["mrr"]:.4f} | {delta_mrr:+.4f} |

## Reranking Activity

| Metric | Value |
|---|---:|
| Examples | {len(results)} |
| Top-1 changed | {changed_top1} |
| Top-1 changed % | {changed_top1 / max(1, len(results)):.2%} |
| Any order changed | {changed_any_order} |
| Any order changed % | {changed_any_order / max(1, len(results)):.2%} |

## Decision Rule

Use the reranked official prediction file only if internal Top-1 or MRR improves, or if it is neutral but improves process-rule plausibility.

Output examples:

`{OUT_CSV}`
"""

    OUT_MD.write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
