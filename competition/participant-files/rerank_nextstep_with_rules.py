#!/usr/bin/env python3
"""
Rerank official next-step Top-5 predictions using:
1. original model rank,
2. valid-data n-gram continuation counts,
3. invalid-data mutation penalties,
4. process-rule validator penalties.

Input:
  competition/participant-files/eval_input_valid.csv
  competition/participant-files/predictions/predictions_nextstep.csv
  competition/track-details/data/coverage_guided_v1/coverage_guided_sequences.csv
  competition/track-details/data/easy_invalid_v1/invalid_sequences.csv
  competition/track-details/data/hard_invalid_v1/hard_invalid_sequences.csv

Output:
  competition/participant-files/predictions/predictions_nextstep_reranked.csv
  competition/participant-files/predictions/rerank_nextstep_report.md
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT / "competition" / "track-details"

# Put repo root on the path so `src` is importable when this script is run
# directly (`python competition/participant-files/rerank_nextstep_with_rules.py`).
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from transformer_xlstm.data.sequence_io import iter_grouped_sequences, norm, read_csv, split_steps
from transformer_xlstm.data.validator import validate_sequence

VALID_TRAIN = TRACK / "data" / "coverage_guided_v1" / "coverage_guided_sequences.csv"
EASY_INVALID = TRACK / "data" / "easy_invalid_v1" / "invalid_sequences.csv"
HARD_INVALID = TRACK / "data" / "hard_invalid_v1" / "hard_invalid_sequences.csv"

EVAL_VALID = ROOT / "competition" / "participant-files" / "eval_input_valid.csv"
PRED_IN = ROOT / "competition" / "participant-files" / "predictions" / "predictions_nextstep.csv"
PRED_OUT = ROOT / "competition" / "participant-files" / "predictions" / "predictions_nextstep_reranked.csv"
REPORT = ROOT / "competition" / "participant-files" / "predictions" / "rerank_nextstep_report.md"


def build_valid_ngram_counts(path: Path):
    print(f"Building valid n-gram counts from {path}")

    counts = {
        "prior": Counter(),
        "fam_prior": defaultdict(Counter),
        "ctx1": defaultdict(Counter),
        "ctx2": defaultdict(Counter),
        "ctx3": defaultdict(Counter),
        "gctx1": defaultdict(Counter),
        "gctx2": defaultdict(Counter),
        "gctx3": defaultdict(Counter),
    }

    nseq = 0
    ntrans = 0

    for _, family, steps, _ in iter_grouped_sequences(path):
        nseq += 1
        for i in range(1, len(steps)):
            nxt = steps[i]
            prev = steps[:i]

            counts["prior"][nxt] += 1
            counts["fam_prior"][family][nxt] += 1

            if len(prev) >= 1:
                key = prev[-1]
                counts["ctx1"][(family, key)][nxt] += 1
                counts["gctx1"][key][nxt] += 1
            if len(prev) >= 2:
                key = tuple(prev[-2:])
                counts["ctx2"][(family, key)][nxt] += 1
                counts["gctx2"][key][nxt] += 1
            if len(prev) >= 3:
                key = tuple(prev[-3:])
                counts["ctx3"][(family, key)][nxt] += 1
                counts["gctx3"][key][nxt] += 1

            ntrans += 1

    print(f"  sequences:   {nseq:,}")
    print(f"  transitions: {ntrans:,}")
    print(f"  vocab next:  {len(counts['prior']):,}")
    return counts


def build_invalid_bad_context_counts(paths: list[Path]):
    """
    Uses invalid mutation metadata:
      MUTATION_INDEX = index where bad step was inserted/moved.
    This directly uses easy/hard invalid data to penalize known bad local continuations.
    """
    bad = {
        "bad1": defaultdict(Counter),
        "bad2": defaultdict(Counter),
        "bad3": defaultdict(Counter),
        "gbad1": defaultdict(Counter),
        "gbad2": defaultdict(Counter),
        "gbad3": defaultdict(Counter),
    }

    nseq = 0
    used = 0

    for path in paths:
        if not path.exists():
            print(f"[WARN] invalid file missing: {path}")
            continue

        print(f"Building invalid-context penalties from {path}")

        for _, family, steps, first_row in iter_grouped_sequences(path):
            nseq += 1

            if not first_row:
                continue

            raw_idx = first_row.get("MUTATION_INDEX", "")
            try:
                idx = int(float(raw_idx))
            except Exception:
                continue

            if idx <= 0 or idx >= len(steps):
                continue

            bad_step = steps[idx]
            prev = steps[:idx]

            if len(prev) >= 1:
                key = prev[-1]
                bad["bad1"][(family, key)][bad_step] += 1
                bad["gbad1"][key][bad_step] += 1
            if len(prev) >= 2:
                key = tuple(prev[-2:])
                bad["bad2"][(family, key)][bad_step] += 1
                bad["gbad2"][key][bad_step] += 1
            if len(prev) >= 3:
                key = tuple(prev[-3:])
                bad["bad3"][(family, key)][bad_step] += 1
                bad["gbad3"][key][bad_step] += 1

            used += 1

    print(f"  invalid sequences scanned: {nseq:,}")
    print(f"  mutation contexts used:    {used:,}")
    return bad


def cget(counter_map, key, cand):
    return counter_map.get(key, Counter()).get(cand, 0)


def score_candidate(
    family: str,
    partial: list[str],
    candidate: str,
    rank_index: int,
    valid_counts,
    bad_counts,
) -> tuple[float, dict[str, float]]:
    """
    Conservative scoring:
    - Original rank remains dominant.
    - Valid n-gram evidence can reorder close candidates.
    - Known invalid contexts and validator violations strongly penalize.
    """
    candidate = norm(candidate)
    family = norm(family)

    # rank_index: 0 for RANK_1, 1 for RANK_2, ...
    base = 10.0 - 1.5 * rank_index
    score = base

    details = {"base": base}

    prior = valid_counts["prior"].get(candidate, 0)
    fam_prior = valid_counts["fam_prior"].get(family, Counter()).get(candidate, 0)

    prior_score = 0.08 * math.log1p(prior)
    fam_prior_score = 0.12 * math.log1p(fam_prior)

    score += prior_score + fam_prior_score
    details["prior"] = prior_score
    details["fam_prior"] = fam_prior_score

    if len(partial) >= 1:
        key1 = partial[-1]
        v = cget(valid_counts["ctx1"], (family, key1), candidate)
        gv = cget(valid_counts["gctx1"], key1, candidate)
        add = 0.35 * math.log1p(v) + 0.15 * math.log1p(gv)
        score += add
        details["ctx1"] = add

        b = cget(bad_counts["bad1"], (family, key1), candidate)
        gb = cget(bad_counts["gbad1"], key1, candidate)
        penalty = 0.45 * math.log1p(b) + 0.20 * math.log1p(gb)
        score -= penalty
        details["bad1"] = -penalty

    if len(partial) >= 2:
        key2 = tuple(partial[-2:])
        v = cget(valid_counts["ctx2"], (family, key2), candidate)
        gv = cget(valid_counts["gctx2"], key2, candidate)
        add = 0.65 * math.log1p(v) + 0.25 * math.log1p(gv)
        score += add
        details["ctx2"] = add

        b = cget(bad_counts["bad2"], (family, key2), candidate)
        gb = cget(bad_counts["gbad2"], key2, candidate)
        penalty = 0.90 * math.log1p(b) + 0.35 * math.log1p(gb)
        score -= penalty
        details["bad2"] = -penalty

    if len(partial) >= 3:
        key3 = tuple(partial[-3:])
        v = cget(valid_counts["ctx3"], (family, key3), candidate)
        gv = cget(valid_counts["gctx3"], key3, candidate)
        add = 0.95 * math.log1p(v) + 0.35 * math.log1p(gv)
        score += add
        details["ctx3"] = add

        b = cget(bad_counts["bad3"], (family, key3), candidate)
        gb = cget(bad_counts["gbad3"], key3, candidate)
        penalty = 1.40 * math.log1p(b) + 0.50 * math.log1p(gb)
        score -= penalty
        details["bad3"] = -penalty

    # Rule validator penalty.
    # This is intentionally strong: if partial + candidate already violates
    # a hard process dependency, do not rank it first.
    violations = validate_sequence(partial + [candidate])
    if violations:
        score -= 50.0
        details["validator_penalty"] = -50.0
    else:
        details["validator_penalty"] = 0.0

    return score, details


def main():
    for p in [VALID_TRAIN, EVAL_VALID, PRED_IN]:
        if not p.exists():
            raise FileNotFoundError(p)

    valid_counts = build_valid_ngram_counts(VALID_TRAIN)
    bad_counts = build_invalid_bad_context_counts([EASY_INVALID, HARD_INVALID])

    eval_rows = read_csv(EVAL_VALID)
    pred_rows = read_csv(PRED_IN)

    eval_by_id = {r["EXAMPLE_ID"].strip(): r for r in eval_rows}

    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)

    changed_top1 = 0
    changed_order = 0
    total = 0
    validator_penalized_candidates = 0
    examples_changed = []

    with PRED_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])

        for row in pred_rows:
            eid = row["EXAMPLE_ID"].strip()
            info = eval_by_id[eid]
            family = info["FAMILY"].strip()
            partial = split_steps(info["PARTIAL_SEQUENCE"])

            old_ranks = [
                norm(row.get("RANK_1", "")),
                norm(row.get("RANK_2", "")),
                norm(row.get("RANK_3", "")),
                norm(row.get("RANK_4", "")),
                norm(row.get("RANK_5", "")),
            ]
            old_ranks = [r for r in old_ranks if r]

            scored = []
            for i, cand in enumerate(old_ranks):
                s, details = score_candidate(
                    family=family,
                    partial=partial,
                    candidate=cand,
                    rank_index=i,
                    valid_counts=valid_counts,
                    bad_counts=bad_counts,
                )
                if details.get("validator_penalty", 0.0) < 0:
                    validator_penalized_candidates += 1
                scored.append((s, cand, details))

            scored.sort(key=lambda x: x[0], reverse=True)
            new_ranks = [cand for _, cand, _ in scored]

            while len(new_ranks) < 5:
                new_ranks.append("")

            writer.writerow([eid] + new_ranks[:5])

            total += 1
            if old_ranks and new_ranks and old_ranks[0] != new_ranks[0]:
                changed_top1 += 1
                if len(examples_changed) < 20:
                    examples_changed.append((eid, family, old_ranks[:5], new_ranks[:5], scored[:5]))

            if old_ranks[:5] != new_ranks[:5]:
                changed_order += 1

    report = []
    report.append("# Next-step Reranking Report\n")
    report.append(
        "Reranked the model's Top-5 predictions using valid-sequence n-gram evidence, invalid-mutation context penalties, and process-rule validator penalties.\n"
    )
    report.append("| Metric | Value |")
    report.append("|---|---:|")
    report.append(f"| Examples | {total} |")
    report.append(f"| Top-1 changed | {changed_top1} |")
    report.append(f"| Top-1 changed % | {changed_top1 / max(1, total):.2%} |")
    report.append(f"| Any order changed | {changed_order} |")
    report.append(f"| Any order changed % | {changed_order / max(1, total):.2%} |")
    report.append(f"| Validator-penalized candidate count | {validator_penalized_candidates} |")
    report.append(f"| Output file | `{PRED_OUT}` |")

    report.append("\n## First changed examples\n")
    for eid, fam, old, new, scored in examples_changed:
        report.append(f"### {eid} / {fam}")
        report.append(f"- old: `{old}`")
        report.append(f"- new: `{new}`")
        brief = [(cand, round(score, 3)) for score, cand, _ in scored]
        report.append(f"- scored: `{brief}`")

    REPORT.write_text("\n\n".join(report), encoding="utf-8")

    print(f"Wrote: {PRED_OUT}")
    print(f"Wrote: {REPORT}")
    print(f"Examples: {total}")
    print(f"Top-1 changed: {changed_top1} ({changed_top1 / max(1, total):.2%})")
    print(f"Any order changed: {changed_order} ({changed_order / max(1, total):.2%})")
    print(f"Validator-penalized candidates: {validator_penalized_candidates}")


if __name__ == "__main__":
    main()
