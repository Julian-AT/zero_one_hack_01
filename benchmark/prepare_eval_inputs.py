#!/usr/bin/env python3
"""
prepare_eval_inputs.py — build the common benchmark eval set from labeled task datasets.

Reads the labeled task datasets produced by
`tracks/industrial-infineon/data/build_task_datasets.py` and emits, for one chosen SPLIT
(default: test), BOTH:

  inputs/        organizer-format model inputs (feed these to every model)
    nextstep_input.csv     EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
    completion_input.csv   EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
    anomaly_input.csv      EXAMPLE_ID, FAMILY, SEQUENCE

  ground_truth/  labels for scoring (used by score.py / run_benchmark.py)
    nextstep_gt.csv        EXAMPLE_ID, FAMILY, NEXT_STEP
    completion_gt.csv      EXAMPLE_ID, FAMILY, PARTIAL_SEQUENCE, FULL_SEQUENCE
    anomaly_gt.csv         EXAMPLE_ID, IS_VALID, VIOLATED_RULE, FAMILY

Task datasets separate steps with ' ||| '; organizer format uses a single '|'. This script
converts between them.

Usage:
  python benchmark/prepare_eval_inputs.py \
      --task-dir tracks/industrial-infineon/data/task_datasets_v1 \
      --split test \
      --out-dir benchmark/eval_set_v1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

SEP_IN = "|||"   # task-dataset separator (with surrounding spaces, stripped on split)
SEP_OUT = "|"    # organizer separator


def steps_from(cell: str) -> list[str]:
    return [s.strip() for s in str(cell).split(SEP_IN) if s.strip()]


def join_steps(steps: list[str]) -> str:
    return SEP_OUT.join(steps)


def read_rows(path: Path, split: str | None) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if str(r.get("SPLIT", "")).strip().lower() == split.lower()]
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default="tracks/industrial-infineon/data/task_datasets_v1")
    ap.add_argument("--split", default="test", help="SPLIT value to keep (test/val/train); '' = all")
    ap.add_argument("--out-dir", default="benchmark/eval_set_v1")
    a = ap.parse_args()

    task = Path(a.task_dir)
    out = Path(a.out_dir)
    split = a.split or None

    # ---- next-step
    ns = read_rows(task / "next_step_prediction.csv", split)
    write_csv(out / "inputs" / "nextstep_input.csv",
              ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"],
              [[r["EXAMPLE_ID"], r.get("FAMILY", ""), "",
                join_steps(steps_from(r.get("PREFIX_CONTEXT", "")))] for r in ns])
    write_csv(out / "ground_truth" / "nextstep_gt.csv",
              ["EXAMPLE_ID", "FAMILY", "NEXT_STEP"],
              [[r["EXAMPLE_ID"], r.get("FAMILY", ""), r.get("NEXT_STEP", "")] for r in ns])

    # ---- completion (FULL = prefix + target suffix)
    cp = read_rows(task / "sequence_completion.csv", split)
    cp_input, cp_gt = [], []
    for r in cp:
        partial = steps_from(r.get("PREFIX_CONTEXT", ""))
        suffix = steps_from(r.get("TARGET_SUFFIX", ""))
        full = partial + suffix
        frac = r.get("CUT_FRACTION", "")
        cp_input.append([r["EXAMPLE_ID"], r.get("FAMILY", ""), frac, join_steps(partial)])
        cp_gt.append([r["EXAMPLE_ID"], r.get("FAMILY", ""), join_steps(partial), join_steps(full)])
    write_csv(out / "inputs" / "completion_input.csv",
              ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"], cp_input)
    write_csv(out / "ground_truth" / "completion_gt.csv",
              ["EXAMPLE_ID", "FAMILY", "PARTIAL_SEQUENCE", "FULL_SEQUENCE"], cp_gt)

    # ---- anomaly
    an = read_rows(task / "anomaly_detection.csv", split)
    write_csv(out / "inputs" / "anomaly_input.csv",
              ["EXAMPLE_ID", "FAMILY", "SEQUENCE"],
              [[r["EXAMPLE_ID"], r.get("FAMILY", ""),
                join_steps(steps_from(r.get("SEQUENCE", "")))] for r in an])
    write_csv(out / "ground_truth" / "anomaly_gt.csv",
              ["EXAMPLE_ID", "IS_VALID", "VIOLATED_RULE", "FAMILY"],
              [[r["EXAMPLE_ID"], r.get("IS_VALID", ""),
                r.get("VIOLATED_RULE", ""), r.get("FAMILY", "")] for r in an])

    print(f"\nDone. Common eval set in {out}/  (split={split or 'all'})")
    print("Feed inputs/ to every model, then score predictions with run_benchmark.py.")


if __name__ == "__main__":
    main()
