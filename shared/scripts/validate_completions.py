"""Self-evaluate our completion.csv by running the validator on
`partial + predicted` for each row.

This is the closest signal we have to the official Task 2 scoring without
having the ground-truth `eval_set_valid.csv`. A completion that produces
no validator violations is process-logic-valid; one that does is wrong
on Tasks 2 (NED) AND Task 3 (anomaly) simultaneously.

Reports per-family + overall:
  - fraction of completions that are validator-clean
  - breakdown by which rule each invalid completion violates

Usage:
    .venv/bin/python shared/scripts/validate_completions.py \\
        --eval-input  competition/participant-files/eval_input_valid.csv \\
        --predictions shared/extras/results/submission_v2_real/completion.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "models"))

from transformer_xlstm.data.validator import validate_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-input", default="competition/participant-files/eval_input_valid.csv")
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()

    # Index eval inputs by EXAMPLE_ID
    inputs: dict[str, dict[str, str]] = {}
    with Path(args.eval_input).open() as f:
        for r in csv.DictReader(f):
            inputs[r["EXAMPLE_ID"]] = r

    # Read predictions
    rows = []
    with Path(args.predictions).open() as f:
        for r in csv.DictReader(f):
            rows.append(r)

    family_total: Counter = Counter()
    family_valid: Counter = Counter()
    rule_hits: Counter = Counter()
    examples_per_rule: dict = defaultdict(list)

    for row in rows:
        eid = row["EXAMPLE_ID"]
        if eid not in inputs:
            continue
        inp = inputs[eid]
        partial = inp["PARTIAL_SEQUENCE"].split("|")
        predicted = row["PREDICTED_SEQUENCE"].split("|") if row["PREDICTED_SEQUENCE"] else []
        full = partial + predicted
        family = inp["FAMILY"].upper()

        viols = validate_sequence(full)
        family_total[family] += 1
        if not viols:
            family_valid[family] += 1
        else:
            for v in viols:
                rule_hits[v.rule] += 1
                if len(examples_per_rule[v.rule]) < 3:
                    examples_per_rule[v.rule].append((eid, v.step_index))

    print(f"\n{'='*60}")
    print(f"COMPLETION SELF-VALIDATION — predictions = {args.predictions}")
    print(f"{'='*60}")
    n_total = sum(family_total.values())
    n_valid = sum(family_valid.values())
    print(f"\nOverall: {n_valid}/{n_total} ({100*n_valid/max(1,n_total):.1f}%) "
          f"completions are validator-clean")
    print()
    print("Per-family:")
    for fam in sorted(family_total):
        tot = family_total[fam]
        ok = family_valid[fam]
        print(f"  {fam:<8s}  {ok:>3}/{tot:<3}  ({100*ok/tot:.1f}%)")

    if rule_hits:
        print()
        print("Rules most commonly violated (lower = better; ideally 0):")
        for rule, n in rule_hits.most_common():
            print(f"  {rule:<40s} {n:>4}")
            for eid, idx in examples_per_rule[rule][:1]:
                print(f"      e.g. {eid} at step_index={idx}")


if __name__ == "__main__":
    main()
