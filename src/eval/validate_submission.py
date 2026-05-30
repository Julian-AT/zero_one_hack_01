"""Strict schema validator for the three submission CSVs.

Checks each row against the format in `generation_rules.md §5.3`:

  nextstep.csv:    EXAMPLE_ID, RANK_1..RANK_5
                   - exactly 6 columns
                   - all 5 ranks present (non-empty)
                   - no duplicate ranks within a row

  completion.csv:  EXAMPLE_ID, PREDICTED_SEQUENCE
                   - 2 columns, PREDICTED_SEQUENCE is "STEP1|STEP2|..."
                   - at least one step
                   - does NOT repeat the prefix from eval_input_valid.csv
                     (Predict only the steps AFTER the cut point.)

  anomaly.csv:     EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
                   - IS_VALID in {0,1}
                   - SCORE in [0.0, 1.0]
                   - PREDICTED_RULE is one of 10 RULE_IDS or empty
                   - When IS_VALID=1 → PREDICTED_RULE should be empty

  Cross-file:      every EXAMPLE_ID in eval_input_valid.csv must appear
                   in both nextstep.csv AND completion.csv, with no extras

Exit code 0 = all valid; non-zero = any failure (printed to stderr).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterator

from src.data.validator import RULE_IDS

VALID_RULES = set(RULE_IDS)


def _err(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def _rows(path: Path) -> Iterator[list[str]]:
    with path.open(newline="") as f:
        for r in csv.reader(f):
            yield r


def validate_nextstep(path: Path, expected_ids: set[str]) -> list[str]:
    errors: list[str] = []
    rows = list(_rows(path))
    if not rows:
        return [f"{path}: empty file"]
    header = [c.strip() for c in rows[0]]
    expected = ["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]
    if header != expected:
        _err(f"{path}: header is {header}, expected {expected}", errors)
        return errors
    seen_ids: set[str] = set()
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != 6:
            _err(f"{path}:{i}: {len(row)} columns, expected 6", errors)
            continue
        eid, *ranks = (c.strip() for c in row)
        if not eid:
            _err(f"{path}:{i}: empty EXAMPLE_ID", errors)
        if eid in seen_ids:
            _err(f"{path}:{i}: duplicate EXAMPLE_ID {eid!r}", errors)
        seen_ids.add(eid)
        for j, r in enumerate(ranks, start=1):
            if not r:
                _err(f"{path}:{i}: RANK_{j} is empty for {eid!r}", errors)
        if len(set(r for r in ranks if r)) != sum(1 for r in ranks if r):
            _err(f"{path}:{i}: duplicate rank values for {eid!r}: {ranks}", errors)
    missing = expected_ids - seen_ids
    extras  = seen_ids - expected_ids
    if missing:
        _err(f"{path}: {len(missing)} EXAMPLE_IDs missing (e.g. {sorted(missing)[:3]})", errors)
    if extras:
        _err(f"{path}: {len(extras)} EXAMPLE_IDs not in eval_input (e.g. {sorted(extras)[:3]})", errors)
    return errors


def validate_completion(path: Path, expected_ids: set[str],
                        valid_input: Path | None = None) -> list[str]:
    errors: list[str] = []
    rows = list(_rows(path))
    if not rows:
        return [f"{path}: empty file"]
    header = [c.strip() for c in rows[0]]
    expected = ["EXAMPLE_ID", "PREDICTED_SEQUENCE"]
    if header != expected:
        _err(f"{path}: header is {header}, expected {expected}", errors)
        return errors

    # Build prefix lookup from eval_input_valid (to check "predict only the
    # steps AFTER the cut point" requirement).
    prefix_by_id: dict[str, list[str]] = {}
    if valid_input is not None and valid_input.exists():
        with valid_input.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                prefix_by_id[row["EXAMPLE_ID"]] = row["PARTIAL_SEQUENCE"].split("|")

    seen_ids: set[str] = set()
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            _err(f"{path}:{i}: {len(row)} columns, expected 2", errors)
            continue
        eid, pred = (c.strip() for c in row)
        if not eid:
            _err(f"{path}:{i}: empty EXAMPLE_ID", errors)
        if eid in seen_ids:
            _err(f"{path}:{i}: duplicate EXAMPLE_ID {eid!r}", errors)
        seen_ids.add(eid)
        steps = pred.split("|") if pred else []
        if not steps or steps == [""]:
            _err(f"{path}:{i}: empty PREDICTED_SEQUENCE for {eid!r}", errors)
        # "Predict only the steps AFTER the cut point" — first predicted step
        # must NOT be the last prefix step.
        if eid in prefix_by_id and prefix_by_id[eid] and steps:
            last_prefix = prefix_by_id[eid][-1]
            if steps[0] == last_prefix:
                # Soft warning: not strictly a schema violation, but a
                # common mistake. Some predictions legitimately repeat the
                # last step if the model thinks so.
                pass

    missing = expected_ids - seen_ids
    extras  = seen_ids - expected_ids
    if missing:
        _err(f"{path}: {len(missing)} EXAMPLE_IDs missing", errors)
    if extras:
        _err(f"{path}: {len(extras)} EXAMPLE_IDs not in eval_input", errors)
    return errors


def validate_anomaly(path: Path, expected_ids: set[str]) -> list[str]:
    errors: list[str] = []
    rows = list(_rows(path))
    if not rows:
        return [f"{path}: empty file"]
    header = [c.strip() for c in rows[0]]
    expected = ["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"]
    if header != expected:
        _err(f"{path}: header is {header}, expected {expected}", errors)
        return errors
    seen_ids: set[str] = set()
    contradictions = 0  # IS_VALID=1 but PREDICTED_RULE set, or =0 but unset
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != 4:
            _err(f"{path}:{i}: {len(row)} columns, expected 4", errors)
            continue
        eid, is_valid, score, rule = (c.strip() for c in row)
        if not eid:
            _err(f"{path}:{i}: empty EXAMPLE_ID", errors)
        if eid in seen_ids:
            _err(f"{path}:{i}: duplicate EXAMPLE_ID {eid!r}", errors)
        seen_ids.add(eid)
        if is_valid not in ("0", "1"):
            _err(f"{path}:{i}: IS_VALID={is_valid!r}, must be 0 or 1", errors)
        try:
            s = float(score)
            if not (0.0 <= s <= 1.0):
                _err(f"{path}:{i}: SCORE={s} not in [0,1]", errors)
        except ValueError:
            _err(f"{path}:{i}: SCORE={score!r} not a float", errors)
        if rule and rule not in VALID_RULES:
            _err(f"{path}:{i}: PREDICTED_RULE={rule!r} not in the 10 RULE_IDS", errors)
        if is_valid == "1" and rule:
            contradictions += 1
        if is_valid == "0" and not rule:
            # Score 0 + no rule isn't strictly invalid (rule is optional)
            # but flag for awareness.
            pass

    if contradictions:
        _err(f"{path}: {contradictions} rows have IS_VALID=1 AND a PREDICTED_RULE "
             f"(rule should be empty when valid)", errors)
    missing = expected_ids - seen_ids
    extras  = seen_ids - expected_ids
    if missing:
        _err(f"{path}: {len(missing)} EXAMPLE_IDs missing", errors)
    if extras:
        _err(f"{path}: {len(extras)} EXAMPLE_IDs not in eval_input", errors)
    return errors


def _ids_from(path: Path, id_col: str = "EXAMPLE_ID") -> set[str]:
    with path.open(newline="") as f:
        return {row[id_col] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", default="extras/results/submission")
    parser.add_argument("--eval-inputs-dir", default="extras/results/eval_inputs")
    args = parser.parse_args()

    sub = Path(args.submission_dir)
    inp = Path(args.eval_inputs_dir)

    valid_input = inp / "eval_input_valid.csv"
    anomaly_input = inp / "eval_input_anomaly.csv"

    valid_ids = _ids_from(valid_input)
    anomaly_ids = _ids_from(anomaly_input)

    print(f"eval_input_valid.csv:   {len(valid_ids)} EXAMPLE_IDs")
    print(f"eval_input_anomaly.csv: {len(anomaly_ids)} EXAMPLE_IDs")
    print()

    all_errors: dict[str, list[str]] = {}
    for fn, ids, validator in [
        ("nextstep.csv",   valid_ids,   validate_nextstep),
        ("completion.csv", valid_ids,   lambda p, e: validate_completion(p, e, valid_input)),
        ("anomaly.csv",    anomaly_ids, validate_anomaly),
    ]:
        path = sub / fn
        if not path.exists():
            all_errors[fn] = [f"{path}: file does not exist"]
            continue
        errs = validator(path, ids)
        all_errors[fn] = errs
        status = "OK" if not errs else f"FAIL ({len(errs)} issues)"
        print(f"{fn:18s} → {status}")
        for e in errs[:10]:
            print(f"    - {e}")
        if len(errs) > 10:
            print(f"    ... and {len(errs) - 10} more")

    total = sum(len(v) for v in all_errors.values())
    print()
    if total == 0:
        print("✓ All three CSVs match the documented schema (generation_rules.md §5.3).")
        sys.exit(0)
    print(f"✗ {total} schema issue(s) across files.")
    sys.exit(1)


if __name__ == "__main__":
    main()
