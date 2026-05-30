#!/usr/bin/env python3
"""
run_benchmark.py — score every model on the common eval set and build the leaderboard.

Expects each model's organizer-format predictions under a per-model folder:

  <pred-root>/<model_name>/
      predictions_nextstep.csv     (EXAMPLE_ID, RANK_1..RANK_5)         [optional]
      predictions_completion.csv   (EXAMPLE_ID, PREDICTED_SEQUENCE)     [optional]
      predictions_anomaly.csv      (EXAMPLE_ID, IS_VALID, SCORE, RULE)  [optional]

and the ground truth produced by prepare_eval_inputs.py under <gt-dir>.

Outputs:
  <out>/results.csv         tidy long-format results
  <out>/RESULTS_auto.md     leaderboard tables

Usage:
  python benchmark/run_benchmark.py \
      --pred-root benchmark/predictions \
      --gt-dir   benchmark/eval_set_v1/ground_truth \
      --out      benchmark
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import score as S

TASKS = {
    "next-step":  ("predictions_nextstep.csv",   "nextstep_gt.csv",   S.score_nextstep),
    "completion": ("predictions_completion.csv", "completion_gt.csv", S.score_completion),
    "anomaly":    ("predictions_anomaly.csv",    "anomaly_gt.csv",    S.score_anomaly),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default="benchmark/predictions")
    ap.add_argument("--gt-dir", default="benchmark/eval_set_v1/ground_truth")
    ap.add_argument("--out", default="benchmark")
    a = ap.parse_args()

    pred_root, gt_dir, out = Path(a.pred_root), Path(a.gt_dir), Path(a.out)
    models = sorted(p.name for p in pred_root.iterdir() if p.is_dir()) if pred_root.exists() else []
    if not models:
        raise SystemExit(f"No model folders under {pred_root}. See benchmark/README.md.")

    rows = []  # (model, task, metric, value)
    for m in models:
        for task, (pred_name, gt_name, fn) in TASKS.items():
            pred_csv = pred_root / m / pred_name
            gt_csv = gt_dir / gt_name
            if not pred_csv.exists() or not gt_csv.exists():
                continue
            res = fn(str(pred_csv), str(gt_csv))
            for metric, val in res["overall"].items():
                rows.append((m, task, metric, round(float(val), 4)))
            rows.append((m, task, "n", res["n"]))

    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "metric", "value"])
        w.writerows(rows)
    print(f"wrote {out / 'results.csv'}  ({len(rows)} rows)")

    # build markdown leaderboard
    def pivot(task, metrics):
        lines = [f"### {task}", "", "| Model | " + " | ".join(metrics) + " |",
                 "|---|" + "|".join(["--:"] * len(metrics)) + "|"]
        for m in models:
            cells = []
            for met in metrics:
                v = next((val for (mm, tt, mt, val) in rows if mm == m and tt == task and mt == met), "")
                cells.append(str(v))
            if any(c != "" for c in cells):
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    md = ["# Benchmark Results (auto-generated)", "",
          "Scored on the common eval set via `score.py` (reuses `eval_metrics.py`).", "",
          pivot("next-step", ["top1", "top3", "top5", "mrr", "n"]),
          pivot("completion", ["ned", "exact_match", "token_acc", "block_acc", "rule_valid_frac", "n"]),
          pivot("anomaly", ["f1", "rule_attr", "auc", "balanced_accuracy", "accuracy", "n"])]
    (out / "RESULTS_auto.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'RESULTS_auto.md'}")


if __name__ == "__main__":
    main()
