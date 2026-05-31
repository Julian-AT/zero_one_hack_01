#!/usr/bin/env python3
"""Shared helpers for benchmark adapters.

Every adapter reads a common eval set built by `make_eval_set.py`:

    <eval-set>/inputs/nextstep_input.csv     EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
    <eval-set>/inputs/completion_input.csv   EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
    <eval-set>/inputs/anomaly_input.csv      EXAMPLE_ID, FAMILY, SEQUENCE

and writes organizer-format predictions:

    <out>/predictions_nextstep.csv     EXAMPLE_ID, RANK_1..RANK_5
    <out>/predictions_completion.csv   EXAMPLE_ID, PREDICTED_SEQUENCE     ('|'-joined suffix)
    <out>/predictions_anomaly.csv      EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
"""
from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def split_steps(cell: str) -> list[str]:
    return [s.strip() for s in str(cell or "").split("|") if s.strip()]


def write_nextstep(out_dir: Path, rows: list[tuple[str, list[str]]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "predictions_nextstep.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for eid, ranks in rows:
            ranks = (list(ranks) + ["", "", "", "", ""])[:5]
            w.writerow([eid, *ranks])
    return p


def write_completion(out_dir: Path, rows: list[tuple[str, list[str]]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "predictions_completion.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for eid, steps in rows:
            w.writerow([eid, "|".join(steps)])
    return p


def write_anomaly(out_dir: Path, rows: list[tuple[str, int, float, str]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "predictions_anomaly.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for eid, is_valid, score, rule in rows:
            w.writerow([eid, int(is_valid), f"{float(score):.4f}", rule or ""])
    return p
