#!/usr/bin/env python3
"""
Generate synthetic OOD semiconductor process families.

Families:
    DIODE
    SCHOTTKY
    SIC_MOSFET

Purpose:
    Broaden self-supervised pretraining beyond MOSFET/IGBT/IC while preserving
    the same process-rule validator from training_data/generate_sequences.py.

Outputs:
    ood_valid_long.csv
    valid_long_augmented.csv

Run from track root:
    python scripts/generate_ood_families.py \
      --count-per-family 1000 \
      --base-valid data/generated/valid_long.csv \
      --out-dir data/generated \
      --seed 777
"""

import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "training_data" / "generate_sequences.py"

if not GENERATOR_PATH.exists():
    raise FileNotFoundError(f"Could not find official generator at: {GENERATOR_PATH}")

spec = importlib.util.spec_from_file_location("generate_sequences", GENERATOR_PATH)
generate_sequences = importlib.util.module_from_spec(spec)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from: {GENERATOR_PATH}")

spec.loader.exec_module(generate_sequences)
validate_sequence = generate_sequences.validate_sequence


OOD_FAMILIES = ["diode", "schottky", "sic_mosfet"]


def _opt(rng, step, prob=0.75):
    return [step] if rng.random() < prob else []


def _meas(rng, step, prob=0.75):
    return [step] if rng.random() < prob else []


def _pre_anneal(rng):
    return ["PRE ANNEAL CHECK"] if rng.random() > 0.4 else []


def _litho(rng, level, inspection):
    steps = [
        "SPIN COAT PHOTORESIST",
        "SOFT BAKE",
        f"ALIGN MASK LEVEL {level}",
        f"EXPOSE LITHO LEVEL {level}",
    ]
    steps += _opt(rng, "POST EXPOSE BAKE", 0.3)
    steps += [
        "DEVELOP PHOTORESIST",
        inspection,
    ]
    steps += _opt(rng, "HARD BAKE", 0.3)
    return steps


def _prefix(rng):
    return [
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        rng.choice(["INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"]),
        rng.choice(["MEASURE THICKNESS", "MEASURE INITIAL THICKNESS"]),
        rng.choice(["MEASURE SURFACE PARTICLES", "MEASURE SURFACE DEFECTS"]),
    ]


def _pre_clean(rng):
    s = ["PRE CLEAN WAFER"]
    s += _opt(rng, "BACKSIDE CLEAN", 0.55)
    s += _opt(rng, "FRONTSIDE CLEAN", 0.55)
    s += [
        rng.choice(["RCA CLEAN 1", "WET CLEAN RCA1"]),
        rng.choice(["RCA CLEAN 2", "WET CLEAN RCA2"]),
        "HF DIP",
    ]
    s += _opt(rng, "DRY WAFER", 0.6)
    return s


def _epi_prep(rng, schottky=False):
    s = [
        "SUBSTRATE CHECK",
        "EPITAXY PREP",
        "EPITAXIAL DEPOSITION",
        "MEASURE EPITAXY THICKNESS",
        "MEASURE RESISTIVITY",
    ]
    if schottky:
        s += _opt(rng, "EPITAXIAL REWORK CHECK", 0.35)
    s += [
        "EPITAXY ANNEAL",
        "WAFER SURFACE CLEAN",
    ]
    return s


def _simple_metal_block(rng, level, family):
    """
    Generic frontside metal patterning.
    Uses only validator-known steps.
    """
    s = [
        "DEPOSIT BARRIER METAL",
        rng.choice(["DEPOSIT METAL 1", "DEPOSIT TOP METAL"]),
        rng.choice(["ANNEAL METAL", "ANNEAL METAL 1"]),
    ]
    s += _meas(rng, "MEASURE METAL THICKNESS", 0.5)
    s += _litho(rng, level, "METAL PATTERN INSPECTION")
    s += [
        "METAL ETCH DRY" if family in ("diode", "schottky") else "METAL ETCH",
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER METAL ETCH",
    ]
    s += _meas(rng, "MEASURE LINE WIDTH", 0.75)
    return s


def _passivation_block(rng):
    return [
        rng.choice(["DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"]),
        "CURE PASSIVATION",
        rng.choice(["MEASURE PASSIVATION THICKNESS", "MEASURE PASSIVATION QUALITY"]),
        rng.choice(["OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"]),
        rng.choice(["PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"]),
        rng.choice(["DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"]),
        rng.choice(["PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN PAD OPENING",
        "MEASURE PAD OPENING",
    ]


def _backside_metal_block(rng):
    return [
        "BACKSIDE CLEAN",
        "BACKSIDE GRIND",
        rng.choice(["MEASURE THICKNESS", "MEASURE WAFER THICKNESS"]),
        "BACKSIDE ETCH CLEAN",
        "BACKSIDE RINSE",
        "BACKSIDE DRY",
        "BACKSIDE METALLIZATION PREP",
        "DEPOSIT BACKSIDE METAL",
        "BACKSIDE ANNEAL",
        "MEASURE BACKSIDE CONTACT",
    ]


def _final_inspection(rng):
    s = ["FINAL CLEAN"]
    s += _meas(rng, "FINAL THICKNESS MEASURE", 0.8)
    s += _meas(rng, "FINAL GEOMETRY CHECK", 0.8)
    s += _meas(rng, "FINAL CD INSPECTION", 0.55)
    s += _meas(rng, "FINAL PARTICLE INSPECTION", 0.8)
    return s


def _test_suite(rng, family):
    s = [rng.choice(["PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST"])]
    s.append("LEAKAGE TEST")

    if family in ("diode", "schottky", "sic_mosfet"):
        s.append("BREAKDOWN VOLTAGE TEST")

    if family == "sic_mosfet":
        s.append("THRESHOLD VOLTAGE TEST")

    s.append("SWITCHING TEST")

    if rng.random() > 0.5:
        s += ["WAFER SORT TEST", "YIELD ANALYSIS"]
    else:
        s += ["YIELD ANALYSIS", "WAFER SORT TEST"]

    return s


def _suffix(rng):
    return [
        rng.choice(["LOT RELEASE", "FINAL LOT RELEASE"]),
        "SHIP LOT",
    ]


def generate_diode(rng):
    """
    Simplified PN diode / rectifier-like process:
    substrate/epi -> oxide window -> junction implant/diffusion -> metal -> passivation -> backside.
    """
    s = []
    s += _prefix(rng)
    s += _pre_clean(rng)
    s += _epi_prep(rng, schottky=False)

    # Junction/window formation
    s += ["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]
    s += _litho(rng, 1, "PATTERN INSPECTION LEVEL 1")
    s += [
        rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER OXIDE ETCH",
    ]
    s += _meas(rng, "MEASURE OPENING CD", 0.8)

    # Diode junction implant
    s += [rng.choice(["IMPLANT P BODY", "IMPLANT N-TYPE"])]
    s += _pre_anneal(rng)
    s += [
        "DRIVE IN DIFFUSION",
        "RAPID THERMAL ANNEAL",
    ]
    s += _meas(rng, "MEASURE JUNCTION DEPTH", 0.8)
    s += _meas(rng, "MEASURE SHEET RESISTANCE", 0.65)

    # Frontside metal
    s += ["WAFER SURFACE CLEAN"]
    s += _simple_metal_block(rng, 2, "diode")

    s += _passivation_block(rng)
    s += _backside_metal_block(rng)
    s += _final_inspection(rng)
    s += _test_suite(rng, "diode")
    s += _suffix(rng)
    return s


def generate_schottky(rng):
    """
    Simplified Schottky diode-like process:
    epi/substrate -> contact opening -> Schottky/contact metal stack -> passivation -> backside.
    Uses existing metal vocabulary rather than introducing non-validator-known tokens.
    """
    s = []
    s += _prefix(rng)
    s += _pre_clean(rng)
    s += _epi_prep(rng, schottky=True)

    # Isolation/contact opening
    s += ["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]
    s += _litho(rng, 1, "VIA OPENING INSPECTION")
    s += [
        rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER WINDOW ETCH",
    ]
    s += _meas(rng, "MEASURE OPENING CD", 0.7)

    # Optional guard/channel-stop type implant to increase diversity
    if rng.random() < 0.45:
        s += ["IMPLANT CHANNEL STOP"]
        s += _pre_anneal(rng)
        s += ["RAPID THERMAL ANNEAL"]
        s += _meas(rng, "MEASURE SHEET RESISTANCE", 0.7)

    # Schottky/contact metal proxy using existing metal vocabulary
    s += ["WAFER SURFACE CLEAN"]
    s += _simple_metal_block(rng, 2, "schottky")

    s += _passivation_block(rng)
    s += _backside_metal_block(rng)
    s += _final_inspection(rng)
    s += _test_suite(rng, "schottky")
    s += _suffix(rng)
    return s


def generate_sic_mosfet(rng):
    """
    SiC MOSFET-like proxy:
    MOSFET-style flow with stronger epi/gate-oxide/metrology emphasis.
    Uses known validator vocabulary to stay rule-compatible.
    """
    s = []
    s += _prefix(rng)
    s += _pre_clean(rng)

    # SiC-like substrate/epi proxy
    s += [
        "SUBSTRATE CHECK",
        "EPITAXY PREP",
        "EPITAXIAL DEPOSITION",
        "MEASURE EPITAXY THICKNESS",
        "MEASURE RESISTIVITY",
    ]
    s += _opt(rng, "EPITAXIAL REWORK CHECK", 0.45)
    s += [
        "EPITAXY ANNEAL",
        "WAFER SURFACE CLEAN",
    ]

    # Body/well formation
    s += ["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]
    s += _litho(rng, 1, "PATTERN INSPECTION LEVEL 1")
    s += [
        rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER ETCH",
    ]
    s += _meas(rng, "MEASURE OPENING CD", 0.8)
    s += ["IMPLANT WELL"]
    s += _pre_anneal(rng)
    s += [
        "DRIVE IN DIFFUSION",
        "RAPID THERMAL ANNEAL",
    ]
    s += _meas(rng, "MEASURE JUNCTION DEPTH", 0.8)

    # Gate oxide / dielectric and poly gate
    s += [
        "THERMAL OXIDATION",
        "GATE OXIDE PREP",
        "GATE OXIDE GROWTH",
        "MEASURE GATE OXIDE THICKNESS",
        "DEPOSIT POLYSILICON",
        "POLYSILICON ANNEAL",
    ]
    s += _meas(rng, "MEASURE POLY THICKNESS", 0.8)

    s += _litho(rng, 2, "POLY PATTERN INSPECTION")
    s += [
        rng.choice(["POLYSILICON ETCH", "POLYSILICON ETCH DRY"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER POLY ETCH",
    ]
    s += _meas(rng, "MEASURE GATE CD", 0.8)

    # Source/drain and LDD/spacer proxy
    s += ["IMPLANT SOURCE DRAIN"]
    s += _pre_anneal(rng)
    s += ["LIGHT ANNEAL"]
    s += _meas(rng, "MEASURE SHEET RESISTANCE", 0.75)

    s += [
        "DEPOSIT SPACER DIELECTRIC",
        "ANISOTROPIC ETCH SPACER",
    ]
    s += _meas(rng, "MEASURE SPACER WIDTH", 0.75)
    s += ["IMPLANT LDD"]
    s += _pre_anneal(rng)
    s += ["RAPID THERMAL ANNEAL"]
    s += _meas(rng, "MEASURE JUNCTION PROFILE", 0.75)

    # ILD / via / metal stack
    s += [
        rng.choice(["DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"]),
        rng.choice(["DENSIFY DIELECTRIC", "DENSIFY OXIDE"]),
        rng.choice(["MEASURE FILM THICKNESS", "MEASURE DIELECTRIC THICKNESS"]),
        rng.choice(["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"]),
        rng.choice(["MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"]),
    ]

    s += _litho(rng, 3, rng.choice(["VIA INSPECTION", "VIA OPENING INSPECTION"]))
    s += [
        rng.choice(["VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER VIA ETCH",
    ]
    s += _meas(rng, "MEASURE VIA CD", 0.75)
    s += [
        "DEPOSIT BARRIER METAL",
        "DEPOSIT METAL SEED",
        "FILL VIA METAL",
        rng.choice(["CMP METAL", "CMP VIA FILL"]),
    ]
    s += _meas(rng, rng.choice(["MEASURE CONTACT RESISTANCE", "MEASURE VIA RESISTANCE"]), 0.75)

    s += _simple_metal_block(rng, 4, "sic_mosfet")
    s += _passivation_block(rng)
    s += _backside_metal_block(rng)
    s += _final_inspection(rng)
    s += _test_suite(rng, "sic_mosfet")
    s += _suffix(rng)

    return s


GENS = {
    "diode": generate_diode,
    "schottky": generate_schottky,
    "sic_mosfet": generate_sic_mosfet,
}


def generate_unique(family, count, seed):
    rng = random.Random(seed)
    gen = GENS[family]

    seen = set()
    seqs = []
    attempts = 0
    max_attempts = max(1000, count * 100)

    while len(seqs) < count and attempts < max_attempts:
        attempts += 1
        seq = gen(rng)
        key = tuple(seq)
        if key in seen:
            continue

        violations = validate_sequence(seq)
        if violations:
            # Skip invalid grammar variants.
            continue

        seen.add(key)
        seqs.append(seq)

    if len(seqs) < count:
        print(
            f"[WARN] {family}: generated only {len(seqs)}/{count} "
            f"unique valid sequences after {attempts} attempts",
            file=sys.stderr,
        )

    return seqs


def write_long_csv(path, family_to_sequences):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])

        for family, seqs in family_to_sequences.items():
            fam_upper = family.upper()
            for i, seq in enumerate(seqs, start=1):
                sid = f"{family}_ood_{i:06d}"
                for step in seq:
                    w.writerow([sid, fam_upper, step])


def combine_valid_files(base_path, ood_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])

        for path in [base_path, ood_path]:
            if not path or not path.exists():
                continue

            with path.open(newline="", encoding="utf-8-sig") as fin:
                r = csv.DictReader(fin)
                for row in r:
                    w.writerow([row["SEQUENCE_ID"], row["FAMILY"], row["STEP"]])


def inspect(path):
    counts = {}
    seq_lengths = {}

    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            fam = row["FAMILY"]
            sid = row["SEQUENCE_ID"]
            counts[fam] = counts.get(fam, 0) + 1
            seq_lengths.setdefault((fam, sid), 0)
            seq_lengths[(fam, sid)] += 1

    fam_seq_counts = {}
    fam_lengths = {}
    for (fam, sid), length in seq_lengths.items():
        fam_seq_counts[fam] = fam_seq_counts.get(fam, 0) + 1
        fam_lengths.setdefault(fam, []).append(length)

    print("\nSummary:")
    for fam in sorted(fam_seq_counts):
        lengths = fam_lengths[fam]
        print(
            f"  {fam}: "
            f"{fam_seq_counts[fam]} sequences, "
            f"rows={counts[fam]}, "
            f"length min/mean/max="
            f"{min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-per-family", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out-dir", default="data/generated")
    ap.add_argument("--base-valid", default="data/generated/valid_long.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    family_to_sequences = {}

    print("Generating OOD valid families...")
    for i, family in enumerate(OOD_FAMILIES):
        seqs = generate_unique(
            family,
            args.count_per_family,
            seed=args.seed + 1000 * i,
        )
        family_to_sequences[family] = seqs
        print(f"  {family.upper()}: {len(seqs)} valid sequences")

    ood_path = out_dir / "ood_valid_long.csv"
    aug_path = out_dir / "valid_long_augmented.csv"
    base_path = Path(args.base_valid)

    write_long_csv(ood_path, family_to_sequences)
    print(f"\nWrote OOD file: {ood_path}")
    inspect(ood_path)

    if base_path.exists():
        combine_valid_files(base_path, ood_path, aug_path)
        print(f"\nWrote augmented file: {aug_path}")
        inspect(aug_path)
    else:
        print(f"\n[WARN] Base valid file not found: {base_path}")
        print("Only wrote OOD file.")


if __name__ == "__main__":
    main()