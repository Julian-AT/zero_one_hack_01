"""On-the-fly generators for synthetic OOD product families.

Adapted from `competition/track-details/scripts/generate_ood_families.py`
(teammate prior work) to be importable from our trainer.

Three synthetic families — DIODE, SCHOTTKY, SIC_MOSFET — are constructed from
the **existing official step vocabulary** so the organizers' validator (the
same one we use for corruption labels) accepts every generated sequence as
valid. They are *not* new families introducing new step strings; they are
re-combinations of known steps into plausible alternative processes.

Used as training-time augmentation: the trainer's online stream draws from
these generators with some probability, expanding the model's exposure
beyond the 3 known families to encourage backbone-level (not family-level)
learning. This is the synthetic-OOD lever targeted at Task 4.

All sequences are validator-clean by construction; we drop the few that
fail `validate_sequence` to be safe.
"""
from __future__ import annotations

import random
from typing import Callable

from transformer_xlstm.data.validator import validate_sequence

OOD_FAMILIES = ("diode", "schottky", "sic_mosfet")


# --------------------------------------------------------------------------- #
# Shared block helpers (mirror the official grammar)                          #
# --------------------------------------------------------------------------- #

def _opt(rng: random.Random, step: str, prob: float = 0.75) -> list[str]:
    return [step] if rng.random() < prob else []


def _pre_anneal(rng: random.Random) -> list[str]:
    return ["PRE ANNEAL CHECK"] if rng.random() > 0.4 else []


def _litho(rng: random.Random, level: int, inspection: str) -> list[str]:
    s = [
        "SPIN COAT PHOTORESIST",
        "SOFT BAKE",
        f"ALIGN MASK LEVEL {level}",
        f"EXPOSE LITHO LEVEL {level}",
    ]
    s += _opt(rng, "POST EXPOSE BAKE", 0.3)
    s += ["DEVELOP PHOTORESIST", inspection]
    s += _opt(rng, "HARD BAKE", 0.3)
    return s


def _prefix(rng: random.Random) -> list[str]:
    return [
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        rng.choice(["INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"]),
        rng.choice(["MEASURE THICKNESS", "MEASURE INITIAL THICKNESS"]),
        rng.choice(["MEASURE SURFACE PARTICLES", "MEASURE SURFACE DEFECTS"]),
    ]


def _pre_clean(rng: random.Random) -> list[str]:
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


def _epi_prep(rng: random.Random, schottky: bool = False) -> list[str]:
    s = [
        "SUBSTRATE CHECK",
        "EPITAXY PREP",
        "EPITAXIAL DEPOSITION",
        "MEASURE EPITAXY THICKNESS",
        "MEASURE RESISTIVITY",
    ]
    if schottky:
        s += _opt(rng, "EPITAXIAL REWORK CHECK", 0.35)
    s += ["EPITAXY ANNEAL", "WAFER SURFACE CLEAN"]
    return s


def _simple_metal_block(rng: random.Random, level: int, family: str) -> list[str]:
    s = [
        "DEPOSIT BARRIER METAL",
        rng.choice(["DEPOSIT METAL 1", "DEPOSIT TOP METAL"]),
        rng.choice(["ANNEAL METAL", "ANNEAL METAL 1"]),
    ]
    if rng.random() < 0.5:
        s += ["MEASURE METAL THICKNESS"]
    s += _litho(rng, level, "METAL PATTERN INSPECTION")
    s += [
        "METAL ETCH DRY" if family in ("diode", "schottky") else "METAL ETCH",
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER METAL ETCH",
    ]
    if rng.random() < 0.75:
        s += ["MEASURE LINE WIDTH"]
    return s


def _passivation_block(rng: random.Random) -> list[str]:
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


def _backside_metal_block(rng: random.Random) -> list[str]:
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


def _final_inspection(rng: random.Random) -> list[str]:
    s = ["FINAL CLEAN"]
    if rng.random() < 0.8:
        s += ["FINAL THICKNESS MEASURE"]
    if rng.random() < 0.8:
        s += ["FINAL GEOMETRY CHECK"]
    if rng.random() < 0.55:
        s += ["FINAL CD INSPECTION"]
    if rng.random() < 0.8:
        s += ["FINAL PARTICLE INSPECTION"]
    return s


def _test_suite(rng: random.Random, family: str) -> list[str]:
    s = [rng.choice(["PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST"]), "LEAKAGE TEST"]
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


def _suffix(rng: random.Random) -> list[str]:
    return [rng.choice(["LOT RELEASE", "FINAL LOT RELEASE"]), "SHIP LOT"]


# --------------------------------------------------------------------------- #
# Family-specific generators                                                  #
# --------------------------------------------------------------------------- #

def generate_diode(rng: random.Random) -> list[str]:
    s = []
    s += _prefix(rng)
    s += _pre_clean(rng)
    s += _epi_prep(rng, schottky=False)
    s += ["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]
    s += _litho(rng, 1, "PATTERN INSPECTION LEVEL 1")
    s += [
        rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER OXIDE ETCH",
    ]
    if rng.random() < 0.8:
        s += ["MEASURE OPENING CD"]
    s += [rng.choice(["IMPLANT P BODY", "IMPLANT N-TYPE"])]
    s += _pre_anneal(rng)
    s += ["DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL"]
    if rng.random() < 0.8: s += ["MEASURE JUNCTION DEPTH"]
    if rng.random() < 0.65: s += ["MEASURE SHEET RESISTANCE"]
    s += ["WAFER SURFACE CLEAN"]
    s += _simple_metal_block(rng, 2, "diode")
    s += _passivation_block(rng)
    s += _backside_metal_block(rng)
    s += _final_inspection(rng)
    s += _test_suite(rng, "diode")
    s += _suffix(rng)
    return s


def generate_schottky(rng: random.Random) -> list[str]:
    s = []
    s += _prefix(rng)
    s += _pre_clean(rng)
    s += _epi_prep(rng, schottky=True)
    s += ["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]
    s += _litho(rng, 1, "VIA OPENING INSPECTION")
    s += [
        rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY", "ETCH SILICON OR OXIDE WINDOW"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER WINDOW ETCH",
    ]
    if rng.random() < 0.7: s += ["MEASURE OPENING CD"]
    if rng.random() < 0.45:
        s += ["IMPLANT CHANNEL STOP"]
        s += _pre_anneal(rng)
        s += ["RAPID THERMAL ANNEAL"]
        if rng.random() < 0.7: s += ["MEASURE SHEET RESISTANCE"]
    s += ["WAFER SURFACE CLEAN"]
    s += _simple_metal_block(rng, 2, "schottky")
    s += _passivation_block(rng)
    s += _backside_metal_block(rng)
    s += _final_inspection(rng)
    s += _test_suite(rng, "schottky")
    s += _suffix(rng)
    return s


def generate_sic_mosfet(rng: random.Random) -> list[str]:
    s = []
    s += _prefix(rng)
    s += _pre_clean(rng)
    s += [
        "SUBSTRATE CHECK", "EPITAXY PREP", "EPITAXIAL DEPOSITION",
        "MEASURE EPITAXY THICKNESS", "MEASURE RESISTIVITY",
    ]
    s += _opt(rng, "EPITAXIAL REWORK CHECK", 0.45)
    s += ["EPITAXY ANNEAL", "WAFER SURFACE CLEAN"]
    # Body/well
    s += ["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]
    s += _litho(rng, 1, "PATTERN INSPECTION LEVEL 1")
    s += [
        rng.choice(["OXIDE ETCH", "OXIDE ETCH DRY"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER ETCH",
    ]
    if rng.random() < 0.8: s += ["MEASURE OPENING CD"]
    s += ["IMPLANT WELL"]
    s += _pre_anneal(rng)
    s += ["DRIVE IN DIFFUSION", "RAPID THERMAL ANNEAL"]
    if rng.random() < 0.8: s += ["MEASURE JUNCTION DEPTH"]
    # Gate oxide + poly
    s += [
        "THERMAL OXIDATION", "GATE OXIDE PREP", "GATE OXIDE GROWTH",
        "MEASURE GATE OXIDE THICKNESS", "DEPOSIT POLYSILICON", "POLYSILICON ANNEAL",
    ]
    if rng.random() < 0.8: s += ["MEASURE POLY THICKNESS"]
    s += _litho(rng, 2, "POLY PATTERN INSPECTION")
    s += [
        rng.choice(["POLYSILICON ETCH", "POLYSILICON ETCH DRY"]),
        rng.choice(["STRIP RESIST", "STRIP PHOTORESIST"]),
        "CLEAN AFTER POLY ETCH",
    ]
    if rng.random() < 0.8: s += ["MEASURE GATE CD"]
    # S/D + LDD spacer
    s += ["IMPLANT SOURCE DRAIN"]
    s += _pre_anneal(rng)
    s += ["LIGHT ANNEAL"]
    if rng.random() < 0.75: s += ["MEASURE SHEET RESISTANCE"]
    s += ["DEPOSIT SPACER DIELECTRIC", "ANISOTROPIC ETCH SPACER"]
    if rng.random() < 0.75: s += ["MEASURE SPACER WIDTH"]
    s += ["IMPLANT LDD"]
    s += _pre_anneal(rng)
    s += ["RAPID THERMAL ANNEAL"]
    if rng.random() < 0.75: s += ["MEASURE JUNCTION PROFILE"]
    # ILD/via/metal
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
    if rng.random() < 0.75: s += ["MEASURE VIA CD"]
    s += [
        "DEPOSIT BARRIER METAL", "DEPOSIT METAL SEED", "FILL VIA METAL",
        rng.choice(["CMP METAL", "CMP VIA FILL"]),
    ]
    if rng.random() < 0.75:
        s += [rng.choice(["MEASURE CONTACT RESISTANCE", "MEASURE VIA RESISTANCE"])]
    s += _simple_metal_block(rng, 4, "sic_mosfet")
    s += _passivation_block(rng)
    s += _backside_metal_block(rng)
    s += _final_inspection(rng)
    s += _test_suite(rng, "sic_mosfet")
    s += _suffix(rng)
    return s


_GENS: dict[str, Callable[[random.Random], list[str]]] = {
    "diode":      generate_diode,
    "schottky":   generate_schottky,
    "sic_mosfet": generate_sic_mosfet,
}


def generate_ood_sequence(family: str, rng: random.Random,
                            max_attempts: int = 5) -> list[str] | None:
    """Generate one OOD-family sequence, retrying if validator finds violations.

    Returns None if every attempt within budget fails validation (shouldn't
    happen by construction; defensive)."""
    gen = _GENS[family]
    for _ in range(max_attempts):
        seq = gen(rng)
        if not validate_sequence(seq):
            return seq
    return None


def random_ood_family(rng: random.Random) -> str:
    return rng.choice(OOD_FAMILIES)


if __name__ == "__main__":
    # Smoke: generate a handful of each family and validate.
    rng = random.Random(0)
    for fam in OOD_FAMILIES:
        ok = 0
        for _ in range(20):
            s = generate_ood_sequence(fam, rng)
            if s is not None and not validate_sequence(s):
                ok += 1
        print(f"{fam:<12} {ok}/20 validator-clean")
