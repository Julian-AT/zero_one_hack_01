#!/usr/bin/env python3
"""
Build valid + invalid semiconductor process-sequence datasets.

Run from repo track root:
    python scripts/build_augmented_data.py --valid-per-family 100 --invalid-per-rule-per-family 20 --multi-invalid-per-family 50 --out-dir data/generated/dev

Outputs:
    sequences.csv          one row per sequence, pipe-separated
    valid_long.csv         long format SEQUENCE_ID, FAMILY, STEP
    summary.csv            counts by family / validity / rule
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_data"))

from generate_sequences import generate_dataset, validate_sequence  # noqa: E402


FAMILIES = ["mosfet", "igbt", "ic"]

RULES = [
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
]

CLEAN_STEPS = {
    "PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS", "WAFER SURFACE CLEAN",
    "RCA CLEAN 1", "RCA CLEAN 2", "WET CLEAN RCA1", "WET CLEAN RCA2",
    "HF DIP", "OXIDE STRIP", "SURFACE PREP FOR DEPOSITION",
    "FRONTSIDE CLEAN", "BACKSIDE CLEAN", "FRONTSIDE CLEAN FINAL",
    "BACKSIDE CLEAN FINAL", "WAFER CLEAN PRE-GRIND",
    "DRY WAFER", "DRY WAFER BACKSIDE",
    "CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH", "CLEAN AFTER POLY ETCH",
    "CLEAN AFTER VIA ETCH", "CLEAN AFTER METAL ETCH",
    "CLEAN AFTER WINDOW ETCH", "CLEAN AFTER FIELD ETCH",
    "CLEAN PAD OPENING", "BACKSIDE ETCH CLEAN", "BACKSIDE RINSE",
    "THERMAL OXIDATION", "GATE OXIDE PREP", "RAPID THERMAL ANNEAL",
    "EPITAXY ANNEAL", "ANNEAL OXIDE",
}

DEPOSITION_STEPS = {
    "THERMAL OXIDATION", "GATE OXIDE GROWTH", "DEPOSIT PAD OXIDE",
    "EPITAXIAL DEPOSITION", "DEPOSIT POLYSILICON",
    "DEPOSIT SPACER DIELECTRIC", "DEPOSIT FIELD OXIDE",
    "DEPOSIT GATE OXIDE OR DIELECTRIC",
    "DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC",
    "DEPOSIT BARRIER METAL", "DEPOSIT METAL SEED", "DEPOSIT METAL 1",
    "DEPOSIT TOP METAL", "DEPOSIT BACKSIDE METAL",
    "DEPOSIT TUNGSTEN SEED", "DEPOSIT PASSIVATION",
    "DEPOSIT PASSIVATION LAYER", "DEPOSIT BACKSIDE PROTECTION",
}

ETCH_STEPS = {
    "OXIDE ETCH", "OXIDE ETCH DRY",
    "POLYSILICON ETCH", "POLYSILICON ETCH DRY",
    "ETCH SILICON OR OXIDE WINDOW", "FIELD OXIDE ETCH",
    "VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA",
    "METAL ETCH", "METAL ETCH DRY",
    "PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH",
}

METAL_ETCH_STEPS = {"METAL ETCH", "METAL ETCH DRY"}

IMPLANT_STEPS = {
    "IMPLANT WELL", "IMPLANT SOURCE DRAIN", "IMPLANT SOURCE REGION",
    "IMPLANT LDD", "IMPLANT P BODY", "IMPLANT N BUFFER",
    "IMPLANT CHANNEL STOP", "IMPLANT DRAIN / CATHODE REGION", "IMPLANT N-TYPE",
}

IMPLANT_OPENER_STEPS = {
    "OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW",
    "DEVELOP PHOTORESIST",
}

CMP_STEPS = {
    "CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC", "CMP METAL", "CMP VIA FILL",
}

FILL_OR_DEP_STEPS = {"FILL VIA METAL", "FILL VIA TUNGSTEN"} | DEPOSITION_STEPS

PAD_WINDOW_STEPS = {
    "OPEN PAD WINDOW", "OPEN BOND PAD WINDOW",
    "PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO",
}

ELECTRICAL_TEST_STEPS = {
    "PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST",
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST",
    "LEAKAGE TEST", "SWITCHING TEST",
}


def rules_of(seq):
    return sorted({v.rule for v in validate_sequence(seq)})


def first_idx(seq, predicate):
    for i, s in enumerate(seq):
        if predicate(s):
            return i
    return None


def all_idx(seq, predicate):
    return [i for i, s in enumerate(seq) if predicate(s)]


def remove_prior_targets(seq, idx, targets, window):
    out = []
    lo = max(0, idx - window)
    for i, s in enumerate(seq):
        if lo <= i < idx and s in targets:
            continue
        out.append(s)
    return out


def move_step_before(seq, step_idx, before_idx):
    if step_idx is None or before_idx is None or step_idx == before_idx:
        return None
    out = list(seq)
    step = out.pop(step_idx)
    if step_idx < before_idx:
        before_idx -= 1
    out.insert(max(0, before_idx), step)
    return out


def corrupt_once(seq, target_rule, rng):
    seq = list(seq)

    if target_rule == "RULE_DEP_NO_CLEAN":
        candidates = all_idx(seq, lambda s: s in DEPOSITION_STEPS)
        rng.shuffle(candidates)
        for idx in candidates:
            out = remove_prior_targets(seq, idx, CLEAN_STEPS, 12)
            if out != seq:
                return out, "remove_clean_before_deposition"
        return None, "failed"

    if target_rule == "RULE_METAL_ETCH_NO_LITHO":
        idxs = all_idx(seq, lambda s: s in METAL_ETCH_STEPS)
        if not idxs:
            return None, "failed"
        idx = rng.choice(idxs)
        out = remove_prior_targets(seq, idx, {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}, 15)
        out = remove_prior_targets(out, min(idx, len(out)-1), set([s for s in out if s.startswith("EXPOSE LITHO LEVEL")]), 15)
        return out, "remove_litho_before_metal_etch"

    if target_rule == "RULE_ETCH_NO_MASK":
        idxs = all_idx(seq, lambda s: s in ETCH_STEPS)
        if not idxs:
            return None, "failed"
        idx = rng.choice(idxs)
        out = remove_prior_targets(seq, idx, {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}, 12)
        return out, "remove_develop_before_etch"

    if target_rule == "RULE_LITHO_LEVEL_SKIP":
        idxs = all_idx(seq, lambda s: s.startswith("ALIGN MASK LEVEL "))
        if not idxs:
            return None, "failed"
        idx = rng.choice(idxs)
        parts = seq[idx].split("ALIGN MASK LEVEL ")
        if len(parts) != 2 or not parts[1].isdigit():
            return None, "failed"
        old = int(parts[1])
        out = list(seq)
        out[idx] = f"ALIGN MASK LEVEL {old + 2}"
        return out, "rename_align_mask_skip_level"

    if target_rule == "RULE_IMPLANT_NO_MASK":
        idxs = all_idx(seq, lambda s: s in IMPLANT_STEPS)
        if not idxs:
            return None, "failed"
        idx = rng.choice(idxs)
        out = remove_prior_targets(seq, idx, IMPLANT_OPENER_STEPS, 15)
        return out, "remove_opener_before_implant"

    if target_rule == "RULE_CMP_NO_DEP":
        cmp_idx = first_idx(seq, lambda s: s in CMP_STEPS)
        if cmp_idx is None:
            return None, "failed"
        out = remove_prior_targets(seq, cmp_idx, FILL_OR_DEP_STEPS, 6)
        return out, "remove_fill_or_dep_before_cmp"

    if target_rule == "RULE_PAD_OPEN_BEFORE_DEP":
        pad_idx = first_idx(seq, lambda s: s in PAD_WINDOW_STEPS)
        dep_idx = first_idx(seq, lambda s: s in {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"})
        out = move_step_before(seq, pad_idx, dep_idx)
        return out, "move_pad_window_before_passivation"

    if target_rule == "RULE_TEST_BEFORE_PASSIVATION":
        test_idx = first_idx(seq, lambda s: s in ELECTRICAL_TEST_STEPS)
        cure_idx = first_idx(seq, lambda s: s == "CURE PASSIVATION")
        out = move_step_before(seq, test_idx, cure_idx)
        return out, "move_test_before_passivation"

    if target_rule == "RULE_SHIP_BEFORE_TEST":
        ship_idx = first_idx(seq, lambda s: s == "SHIP LOT")
        sort_idx = first_idx(seq, lambda s: s == "WAFER SORT TEST")
        out = move_step_before(seq, ship_idx, sort_idx)
        return out, "move_ship_before_wafer_sort"

    if target_rule == "RULE_BACKSIDE_BEFORE_PASSIVATION":
        back_idx = first_idx(seq, lambda s: s == "DEPOSIT BACKSIDE METAL")
        cure_idx = first_idx(seq, lambda s: s == "CURE PASSIVATION")
        out = move_step_before(seq, back_idx, cure_idx)
        return out, "move_backside_metal_before_passivation"

    return None, "failed"


def make_single_rule_invalid(base_sequences, family, rule, n, rng, max_attempts_factor=200):
    rows = []
    attempts = 0
    max_attempts = max(1000, n * max_attempts_factor)

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        base_id, seq = rng.choice(base_sequences)
        bad, corruption_type = corrupt_once(seq, rule, rng)
        if not bad:
            continue

        detected = rules_of(bad)
        if rule not in detected:
            continue

        rows.append({
            "FAMILY": family.upper(),
            "SEQUENCE": "|".join(bad),
            "IS_VALID": 0,
            "RULE_LABELS": ";".join(detected),
            "PRIMARY_RULE": rule,
            "NUM_VIOLATIONS": len(detected),
            "CORRUPTION_TYPE": corruption_type,
            "BASE_SEQUENCE_ID": base_id,
        })

    if len(rows) < n:
        print(f"[WARN] {family} {rule}: only produced {len(rows)}/{n} examples after {attempts} attempts", file=sys.stderr)

    return rows


def make_multi_rule_invalid(base_sequences, family, n, rng):
    rows = []
    attempts = 0
    max_attempts = max(1000, n * 200)

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        base_id, seq = rng.choice(base_sequences)
        rules = rng.sample(RULES, k=rng.choice([2, 3]))
        bad = list(seq)
        corruptions = []

        for rule in rules:
            candidate, ctype = corrupt_once(bad, rule, rng)
            if candidate:
                bad = candidate
                corruptions.append(ctype)

        detected = rules_of(bad)
        if len(detected) < 2:
            continue

        rows.append({
            "FAMILY": family.upper(),
            "SEQUENCE": "|".join(bad),
            "IS_VALID": 0,
            "RULE_LABELS": ";".join(detected),
            "PRIMARY_RULE": detected[0],
            "NUM_VIOLATIONS": len(detected),
            "CORRUPTION_TYPE": "+".join(corruptions),
            "BASE_SEQUENCE_ID": base_id,
        })

    if len(rows) < n:
        print(f"[WARN] {family} multi-rule: only produced {len(rows)}/{n} examples after {attempts} attempts", file=sys.stderr)

    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "EXAMPLE_ID", "FAMILY", "SEQUENCE", "IS_VALID",
        "RULE_LABELS", "PRIMARY_RULE", "NUM_VIOLATIONS",
        "CORRUPTION_TYPE", "BASE_SEQUENCE_ID",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            row = dict(row)
            row["EXAMPLE_ID"] = f"ex_{i:07d}"
            w.writerow(row)


def write_valid_long(path, valid_by_family):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])
        for family, seqs in valid_by_family.items():
            for sid, seq in seqs:
                for step in seq:
                    w.writerow([sid, family.upper(), step])


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for r in rows:
        key = (r["FAMILY"], r["IS_VALID"], r["PRIMARY_RULE"] or "VALID")
        counts[key] += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["FAMILY", "IS_VALID", "PRIMARY_RULE", "COUNT"])
        for (fam, valid, rule), count in sorted(counts.items()):
            w.writerow([fam, valid, rule, count])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-per-family", type=int, default=100)
    ap.add_argument("--invalid-per-rule-per-family", type=int, default=20)
    ap.add_argument("--multi-invalid-per-family", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    all_rows = []
    valid_by_family = {}

    print("[1/3] Generating valid sequences...")
    for fam_idx, family in enumerate(FAMILIES):
        seqs = generate_dataset(
            family,
            args.valid_per_family,
            seed=args.seed + 1000 * (fam_idx + 1),
            validate=True,
        )
        named = [(f"{family}_valid_{i:06d}", seq) for i, seq in enumerate(seqs, start=1)]
        valid_by_family[family] = named

        for sid, seq in named:
            detected = rules_of(seq)
            if detected:
                raise RuntimeError(f"Generated invalid valid-sequence {sid}: {detected}")
            all_rows.append({
                "FAMILY": family.upper(),
                "SEQUENCE": "|".join(seq),
                "IS_VALID": 1,
                "RULE_LABELS": "",
                "PRIMARY_RULE": "",
                "NUM_VIOLATIONS": 0,
                "CORRUPTION_TYPE": "none",
                "BASE_SEQUENCE_ID": sid,
            })

    print("[2/3] Generating single-rule invalid sequences...")
    for family in FAMILIES:
        base = valid_by_family[family]
        for rule in RULES:
            rows = make_single_rule_invalid(
                base,
                family,
                rule,
                args.invalid_per_rule_per_family,
                rng,
            )
            all_rows.extend(rows)
            print(f"  {family.upper()} {rule}: {len(rows)}")

    print("[3/3] Generating multi-rule invalid sequences...")
    for family in FAMILIES:
        rows = make_multi_rule_invalid(
            valid_by_family[family],
            family,
            args.multi_invalid_per_family,
            rng,
        )
        all_rows.extend(rows)
        print(f"  {family.upper()} MULTI: {len(rows)}")

    write_rows(out_dir / "sequences.csv", all_rows)
    write_valid_long(out_dir / "valid_long.csv", valid_by_family)
    write_summary(out_dir / "summary.csv", all_rows)

    print("\nDone.")
    print(f"  Wrote: {out_dir / 'sequences.csv'}")
    print(f"  Wrote: {out_dir / 'valid_long.csv'}")
    print(f"  Wrote: {out_dir / 'summary.csv'}")
    print(f"  Total examples: {len(all_rows):,}")


if __name__ == "__main__":
    main()