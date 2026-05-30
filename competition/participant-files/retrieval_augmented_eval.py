#!/usr/bin/env python3
"""
Retrieval-augmented eval prediction.

Uses a large generated valid sequence bank to improve:
  - Task 1 next-step ranking
  - Task 2 sequence completion

It does not change anomaly predictions.

Inputs:
  competition/participant-files/eval_input_valid.csv
  competition/participant-files/predictions/predictions_nextstep.csv
  competition/participant-files/predictions/predictions_completion.csv
  competition/track-details/data/coverage_guided_v1/coverage_guided_sequences.csv
  competition/track-details/data/retrieval_bank_v1/*.csv

Outputs:
  competition/participant-files/predictions/predictions_nextstep_retrieval.csv
  competition/participant-files/predictions/predictions_completion_retrieval.csv
  competition/participant-files/predictions/retrieval_augmented_report.md
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from transformer_xlstm.data.sequence_io import iter_grouped_sequences, norm, read_csv, split_steps
from transformer_xlstm.data.validator import validate_sequence

TRACK = ROOT / "competition" / "track-details"

EVAL_VALID = ROOT / "competition" / "participant-files" / "eval_input_valid.csv"

PRED_DIR = ROOT / "competition" / "participant-files" / "predictions"
MODEL_NEXT = PRED_DIR / "predictions_nextstep.csv"
MODEL_COMPLETION = PRED_DIR / "predictions_completion.csv"

OUT_NEXT = PRED_DIR / "predictions_nextstep_retrieval.csv"
OUT_COMPLETION = PRED_DIR / "predictions_completion_retrieval.csv"
REPORT = PRED_DIR / "retrieval_augmented_report.md"

VALID_SOURCES = [
    TRACK / "data" / "coverage_guided_v1" / "coverage_guided_sequences.csv",
    TRACK / "data" / "retrieval_bank_v1" / "MOSFET_retrieval.csv",
    TRACK / "data" / "retrieval_bank_v1" / "IGBT_retrieval.csv",
    TRACK / "data" / "retrieval_bank_v1" / "IC_retrieval.csv",
]


def model_next_rows():
    rows = read_csv(MODEL_NEXT)
    return {r["EXAMPLE_ID"].strip(): r for r in rows}


def model_completion_rows():
    rows = read_csv(MODEL_COMPLETION)
    return {r["EXAMPLE_ID"].strip(): r for r in rows}


def build_eval_prefix_index(eval_rows):
    """
    Returns:
      prefix_to_eids[(family, length, prefix_tuple)] = [eid, ...]
      lengths_by_family[family] = {lengths...}
    """
    prefix_to_eids = defaultdict(list)
    lengths_by_family = defaultdict(set)
    eval_info = {}

    for r in eval_rows:
        eid = r["EXAMPLE_ID"].strip()
        family = norm(r["FAMILY"])
        partial = split_steps(r["PARTIAL_SEQUENCE"])
        key = (family, len(partial), tuple(partial))

        prefix_to_eids[key].append(eid)
        lengths_by_family[family].add(len(partial))
        eval_info[eid] = {
            "family": family,
            "partial": partial,
            "fraction": r.get("COMPLETION_FRACTION", ""),
        }

    return prefix_to_eids, lengths_by_family, eval_info


def retrieve_matches(prefix_to_eids, lengths_by_family):
    """
    For every eval partial prefix, collect:
      - next-step counts
      - full suffix counts
    """
    match_counts = Counter()
    next_counts = defaultdict(Counter)
    suffix_counts = defaultdict(Counter)
    scanned = 0

    for source in VALID_SOURCES:
        if not source.exists():
            print(f"[WARN] source missing, skipping: {source}")
            continue

        print(f"Scanning valid source: {source}")

        for _sid, family, steps, _first in iter_grouped_sequences(source, warn_missing=True):
            scanned += 1
            family = norm(family)

            if family not in lengths_by_family:
                continue

            for L in lengths_by_family[family]:
                if L >= len(steps):
                    continue

                key = (family, L, tuple(steps[:L]))
                eids = prefix_to_eids.get(key)
                if not eids:
                    continue

                next_step = steps[L]
                suffix = tuple(steps[L:])

                for eid in eids:
                    match_counts[eid] += 1
                    next_counts[eid][next_step] += 1
                    suffix_counts[eid][suffix] += 1

        print(f"  scanned so far: {scanned:,}")

    return match_counts, next_counts, suffix_counts, scanned


def merge_ranks(retrieved_steps, model_ranks):
    out = []

    for step, _count in retrieved_steps:
        step = norm(step)
        if step and step not in out:
            out.append(step)

    for step in model_ranks:
        step = norm(step)
        if step and step not in out:
            out.append(step)

    while len(out) < 5:
        out.append("")

    return out[:5]


def choose_suffix(eid, suffix_counter):
    if not suffix_counter:
        return None

    # Most common retrieved suffix.
    # Tie-break: prefer longer complete-looking sequences ending in SHIP LOT.
    items = list(suffix_counter.items())

    def key_fn(item):
        suffix, count = item
        ends_ship = 1 if suffix and suffix[-1] == "SHIP LOT" else 0
        return (count, ends_ship, len(suffix))

    best_suffix, _ = max(items, key=key_fn)
    return list(best_suffix)


def main():
    for p in [EVAL_VALID, MODEL_NEXT, MODEL_COMPLETION]:
        if not p.exists():
            raise FileNotFoundError(p)

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    eval_rows = read_csv(EVAL_VALID)
    model_next = model_next_rows()
    model_completion = model_completion_rows()

    prefix_to_eids, lengths_by_family, eval_info = build_eval_prefix_index(eval_rows)

    print("Eval examples:", len(eval_rows))
    print("Prefix lengths by family:")
    for fam, lengths in sorted(lengths_by_family.items()):
        print(f"  {fam}: {sorted(lengths)}")

    match_counts, next_counts, suffix_counts, scanned = retrieve_matches(
        prefix_to_eids,
        lengths_by_family,
    )

    exact_matched = sum(1 for r in eval_rows if match_counts[r["EXAMPLE_ID"].strip()] > 0)

    changed_next_top1 = 0
    changed_next_any = 0

    with OUT_NEXT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])

        for r in eval_rows:
            eid = r["EXAMPLE_ID"].strip()
            old = model_next[eid]
            model_ranks = [
                old.get("RANK_1", ""),
                old.get("RANK_2", ""),
                old.get("RANK_3", ""),
                old.get("RANK_4", ""),
                old.get("RANK_5", ""),
            ]

            retrieved = next_counts[eid].most_common(5)
            new_ranks = merge_ranks(retrieved, model_ranks)

            old_ranks = [norm(x) for x in model_ranks]
            if new_ranks[0] != old_ranks[0]:
                changed_next_top1 += 1
            if new_ranks != old_ranks:
                changed_next_any += 1

            writer.writerow([eid] + new_ranks)

    changed_completion = 0
    validator_valid = 0
    validator_invalid = 0
    retrieved_completion_used = 0

    with OUT_COMPLETION.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])

        for r in eval_rows:
            eid = r["EXAMPLE_ID"].strip()
            partial = eval_info[eid]["partial"]

            old_suffix = split_steps(model_completion[eid].get("PREDICTED_SEQUENCE", ""))

            retrieved_suffix = choose_suffix(eid, suffix_counts[eid])

            if retrieved_suffix is not None:
                suffix = retrieved_suffix
                retrieved_completion_used += 1
            else:
                suffix = old_suffix

            if suffix != old_suffix:
                changed_completion += 1

            full_pred = partial + suffix
            violations = validate_sequence(full_pred)
            if violations:
                validator_invalid += 1
            else:
                validator_valid += 1

            writer.writerow([eid, "|".join(suffix)])

    match_values = [match_counts[r["EXAMPLE_ID"].strip()] for r in eval_rows]
    avg_matches = sum(match_values) / len(match_values) if match_values else 0
    max_matches = max(match_values) if match_values else 0

    report = (
        """# Retrieval-Augmented Eval Prediction Report

This run uses a large synthetic valid-sequence retrieval bank to improve official eval predictions.

## Data Sources

| Source | Exists |
|---|---:|
"""
        + "\n".join(f"| `{p}` | `{p.exists()}` |" for p in VALID_SOURCES)
        + f"""

## Retrieval Coverage

| Metric | Value |
|---|---:|
| Eval valid examples | {len(eval_rows)} |
| Valid sequences scanned | {scanned} |
| Exact prefix matched examples | {exact_matched} |
| Exact prefix match rate | {exact_matched / max(1, len(eval_rows)):.2%} |
| Mean exact matches per eval prefix | {avg_matches:.2f} |
| Max exact matches for one prefix | {max_matches} |

## Next-step Changes

| Metric | Value |
|---|---:|
| Top-1 changed vs model-only | {changed_next_top1} |
| Top-1 changed % | {changed_next_top1 / max(1, len(eval_rows)):.2%} |
| Any Top-5 order changed | {changed_next_any} |
| Any Top-5 order changed % | {changed_next_any / max(1, len(eval_rows)):.2%} |

## Completion Changes

| Metric | Value |
|---|---:|
| Retrieved completion used | {retrieved_completion_used} |
| Retrieved completion used % | {retrieved_completion_used / max(1, len(eval_rows)):.2%} |
| Completion changed vs model-only | {changed_completion} |
| Completion changed % | {changed_completion / max(1, len(eval_rows)):.2%} |
| Validator-valid completed sequences | {validator_valid} |
| Validator-invalid completed sequences | {validator_invalid} |

## Output Files

| File | Path |
|---|---|
| Retrieval next-step predictions | `{OUT_NEXT}` |
| Retrieval completion predictions | `{OUT_COMPLETION}` |

## Decision Rule

Use retrieval-augmented predictions if exact prefix match rate is high enough to trust the retrieval bank.

Recommended:
- If exact prefix match rate is above 25%, use both retrieval next-step and retrieval completion.
- If exact prefix match rate is below 10%, keep model-only predictions.
- If completion validator-invalid count increases substantially, keep model-only completion.
"""
    )

    REPORT.write_text(report, encoding="utf-8")

    print(report)
    print("Wrote:", OUT_NEXT)
    print("Wrote:", OUT_COMPLETION)
    print("Wrote:", REPORT)


if __name__ == "__main__":
    main()
