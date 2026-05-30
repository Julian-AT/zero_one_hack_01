"""Role ontology + step->role induction — the OOD generalization lever.

Two distinct uses of "role" in this codebase, kept deliberately separate:

  1. `induce_role(step)` / `role_idx(step)` — a COARSE functional category used as
     a feature for the ranker (PPM / neural). Approximate is fine here.

  2. Validator augmentation (in `nspe.rules.validate_with_roles`) — a PRECISE,
     conservative classification that ONLY applies to genuinely novel (unknown-
     vocab) steps. That logic lives in rules.py with a known-vocab guard so it is
     provably identical to the official validator on any known-vocab sequence.

Known steps are anchored to the validator's own frozensets; unknown (family-4)
strings fall back to surface-pattern induction.
"""
from __future__ import annotations

from nspe.official import gs

ROLES = [
    "LOGISTICS", "INSPECT_MEASURE", "CLEAN", "SUBSTRATE_PREP", "THERMAL_DEP",
    "LITHO", "ETCH", "STRIP", "IMPLANT", "ANNEAL_DIFFUSION", "CMP", "FILL",
    "PASSIVATION", "BACKSIDE", "TEST", "OTHER",
]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
NUM_ROLES = len(ROLES)

# Anchor known steps from the official frozensets (authoritative for rule triggers).
_KNOWN: dict[str, str] = {}
for _s in gs.DEPOSITION_STEPS:
    _KNOWN[_s] = "THERMAL_DEP"
for _s in gs.ETCH_STEPS:
    _KNOWN[_s] = "ETCH"
for _s in gs.IMPLANT_STEPS:
    _KNOWN[_s] = "IMPLANT"
for _s in gs.CMP_STEPS:
    _KNOWN[_s] = "CMP"
for _s in gs.FILL_STEPS:
    _KNOWN.setdefault(_s, "FILL")
for _s in gs.ELECTRICAL_TEST_STEPS:
    _KNOWN[_s] = "TEST"
for _s in gs.CLEAN_STEPS:
    _KNOWN.setdefault(_s, "CLEAN")

_LITHO_KW = ("SPIN COAT", "SOFT BAKE", "ALIGN MASK", "EXPOSE LITHO",
             "POST EXPOSE BAKE", "HARD BAKE", "MASK LEVEL", "PAD WINDOW LITHO")


def induce_role(step: str) -> str:
    """Coarse functional role for a step string (known or novel)."""
    s = step.upper().strip()
    if s in _KNOWN:
        return _KNOWN[s]

    # Lithography block ops (developing/patterning) — check before generic ETCH/INSPECT.
    if any(k in s for k in _LITHO_KW) or s == "DEVELOP PHOTORESIST" or s == "DEVELOP PAD WINDOW":
        return "LITHO"
    if "PATTERN INSPECTION" in s or s.endswith("PATTERN INSPECTION"):
        return "LITHO"

    if s.startswith("DEPOSIT ") or "OXIDATION" in s or s.endswith(" GROWTH") \
            or s == "EPITAXIAL DEPOSITION":
        return "THERMAL_DEP"
    if "DENSIFY" in s:
        return "ANNEAL_DIFFUSION"
    if s.startswith("CMP ") or "PLANAR" in s:
        return "CMP"
    if s.startswith("FILL VIA"):
        return "FILL"
    if " ETCH" in s or s.startswith("ETCH "):
        return "ETCH"
    if s.startswith("STRIP "):
        return "STRIP"
    if s.startswith("IMPLANT "):
        return "IMPLANT"
    if "PASSIVATION" in s or "PAD WINDOW" in s or s == "CURE PASSIVATION":
        return "PASSIVATION"
    if "ANNEAL" in s or "DIFFUSION" in s or "RAPID THERMAL" in s:
        return "ANNEAL_DIFFUSION"
    if "BACKSIDE" in s or "GRIND" in s:
        return "BACKSIDE"
    if s.endswith(" TEST") or "ANALYSIS" in s or "WAFER SORT" in s:
        return "TEST"
    # Inspection / measurement (after the more specific categories above).
    if s.startswith(("MEASURE", "INSPECT")) or s.endswith(("CHECK", "INSPECTION")) \
            or s.startswith("FINAL"):
        return "INSPECT_MEASURE"
    if ("CLEAN" in s) or s.endswith(" RINSE") or s.startswith("RINSE ") \
            or s.startswith("DRY ") or s == "HF DIP":
        return "CLEAN"
    if "EPITAX" in s or "SUBSTRATE" in s or "SURFACE PREP" in s:
        return "SUBSTRATE_PREP"
    if any(k in s for k in ("RECEIVE WAFER", "LOT ", "SHIP LOT", "RELEASE", "PACKAGE")):
        return "LOGISTICS"
    return "OTHER"


def role_idx(step: str) -> int:
    return ROLE_TO_IDX[induce_role(step)]


__all__ = ["ROLES", "ROLE_TO_IDX", "NUM_ROLES", "induce_role", "role_idx"]
