"""Synonym canonicalizer for process steps.

Several pairs of step strings are interchangeable in the grammar (see
generation_rules.md §4 and EDA). Canonicalization is a *post-processing* pass
for Task 2 exact-match: we map every synonym to a single canonical form, both
in the model's outputs and in the ground-truth labels.

Used in both directions:
- At training time: canonicalize inputs so the model sees a smaller vocabulary
  (optional; opt-in via config).
- At eval time: canonicalize both predictions and references before scoring
  exact-match / token-accuracy. This is the synonym-aware exact match.
"""

from __future__ import annotations

# Canonical forms match the organizers' reference sequences to avoid scoring drift.
CANONICAL: dict[str, str] = {
    "STRIP RESIST": "STRIP PHOTORESIST",
    "STRIP RESIST LEVEL 2": "STRIP PHOTORESIST",
    "STRIP RESIST LEVEL 3": "STRIP PHOTORESIST",
    "STRIP RESIST LEVEL 4": "STRIP PHOTORESIST",
    "WET CLEAN RCA1": "RCA CLEAN 1",
    "WET CLEAN RCA2": "RCA CLEAN 2",
    "DEPOSIT INTERLEVEL DIELECTRIC": "DEPOSIT INTERLAYER DIELECTRIC",
    "DENSIFY OXIDE": "DENSIFY DIELECTRIC",
    "MEASURE DIELECTRIC THICKNESS": "MEASURE FILM THICKNESS",
    "CMP INTERLAYER DIELECTRIC": "CMP DIELECTRIC",
    "MEASURE SURFACE PLANARITY": "MEASURE PLANARITY",
    "VIA ETCH THROUGH DIELECTRIC": "VIA ETCH",
    "DIELECTRIC ETCH VIA": "VIA ETCH",
    "CMP VIA FILL": "CMP METAL",
    "METAL ETCH DRY": "METAL ETCH",
    "POLYSILICON ETCH DRY": "POLYSILICON ETCH",
    "OXIDE ETCH DRY": "OXIDE ETCH",
    "FINAL LOT RELEASE": "LOT RELEASE",
    "DEVELOP PAD WINDOW": "DEVELOP PHOTORESIST",
    "OPEN PAD WINDOW LITHO": "PAD WINDOW LITHO",
    "OPEN BOND PAD WINDOW": "OPEN PAD WINDOW",
    "DEPOSIT PASSIVATION LAYER": "DEPOSIT PASSIVATION",
    "MEASURE PASSIVATION QUALITY": "MEASURE PASSIVATION THICKNESS",
    "PASSIVATION ETCH PAD OPENING": "PASSIVATION ETCH",
    "DEPOSIT TOP METAL": "DEPOSIT METAL 1",
    "ANNEAL METAL": "ANNEAL METAL 1",
    "MEASURE INITIAL THICKNESS": "MEASURE THICKNESS",
    "MEASURE INITIAL GEOMETRY": "MEASURE GEOMETRY",
    "WAFER CLEAN PRE PROCESS": "PRE CLEAN WAFER",
    "PRE CLEAN INSPECTION": "INITIAL WAFER INSPECTION",
    "DRY WAFER BACKSIDE": "DRY WAFER",
    "BACKSIDE CLEAN FINAL": "BACKSIDE CLEAN",
    "FRONTSIDE CLEAN FINAL": "FRONTSIDE CLEAN",
    "ANNEAL POLYSILICON": "POLYSILICON ANNEAL",
    "ELECTRICAL PARAMETRIC TEST": "PARAMETRIC TEST",
    "FINAL THICKNESS MEASURE": "MEASURE THICKNESS",
    "POLY PATTERN INSPECTION": "INSPECT PATTERN LEVEL 2",
    "P BODY WINDOW INSPECTION": "INSPECT PATTERN LEVEL 2",
    "FIELD PATTERN INSPECTION": "INSPECT PATTERN LEVEL 3",
    "METAL PATTERN INSPECTION": "INSPECT PATTERN LEVEL 4",
    "VIA INSPECTION": "INSPECT PATTERN LEVEL 3",
    "VIA OPENING INSPECTION": "INSPECT PATTERN LEVEL 3",
    "PATTERN INSPECTION LEVEL 1": "INSPECT PATTERN LEVEL 1",
    "PATTERN INSPECTION LEVEL 2": "INSPECT PATTERN LEVEL 2",
}


def canonicalize_step(step: str) -> str:
    """Return the canonical synonym for a single step."""
    return CANONICAL.get(step, step)


def canonicalize_sequence(steps: list[str]) -> list[str]:
    """Canonicalize every step in a sequence."""
    return [CANONICAL.get(s, s) for s in steps]
