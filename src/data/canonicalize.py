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

# Each key is a synonym; value is its canonical form.
# Keep canonical forms identical to those used most often in the provided
# reference sequences so we don't drift from the organizers' format.
CANONICAL: dict[str, str] = {
    # Resist strip variants
    "STRIP RESIST":              "STRIP PHOTORESIST",
    "STRIP RESIST LEVEL 2":      "STRIP PHOTORESIST",
    "STRIP RESIST LEVEL 3":      "STRIP PHOTORESIST",
    "STRIP RESIST LEVEL 4":      "STRIP PHOTORESIST",

    # RCA cleans
    "WET CLEAN RCA1":            "RCA CLEAN 1",
    "WET CLEAN RCA2":            "RCA CLEAN 2",

    # Inter-layer dielectric
    "DEPOSIT INTERLEVEL DIELECTRIC":  "DEPOSIT INTERLAYER DIELECTRIC",
    "DENSIFY OXIDE":                  "DENSIFY DIELECTRIC",
    "MEASURE DIELECTRIC THICKNESS":   "MEASURE FILM THICKNESS",
    "CMP INTERLAYER DIELECTRIC":      "CMP DIELECTRIC",
    "MEASURE SURFACE PLANARITY":      "MEASURE PLANARITY",

    # Via etch variants
    "VIA ETCH THROUGH DIELECTRIC":    "VIA ETCH",
    "DIELECTRIC ETCH VIA":            "VIA ETCH",

    # CMP via variants
    "CMP VIA FILL":                   "CMP METAL",

    # Metal etch
    "METAL ETCH DRY":                 "METAL ETCH",

    # Etch poly
    "POLYSILICON ETCH DRY":           "POLYSILICON ETCH",

    # Oxide etch
    "OXIDE ETCH DRY":                 "OXIDE ETCH",

    # Lot release
    "FINAL LOT RELEASE":              "LOT RELEASE",

    # Develop pad window
    "DEVELOP PAD WINDOW":             "DEVELOP PHOTORESIST",

    # Pad window litho synonyms
    "OPEN PAD WINDOW LITHO":          "PAD WINDOW LITHO",
    "OPEN BOND PAD WINDOW":           "OPEN PAD WINDOW",

    # Passivation dep
    "DEPOSIT PASSIVATION LAYER":      "DEPOSIT PASSIVATION",
    "MEASURE PASSIVATION QUALITY":    "MEASURE PASSIVATION THICKNESS",
    "PASSIVATION ETCH PAD OPENING":   "PASSIVATION ETCH",

    # Metal dep + anneal
    "DEPOSIT TOP METAL":              "DEPOSIT METAL 1",
    "ANNEAL METAL":                   "ANNEAL METAL 1",

    # Initial measurements
    "MEASURE INITIAL THICKNESS":      "MEASURE THICKNESS",
    "MEASURE INITIAL GEOMETRY":       "MEASURE GEOMETRY",

    # Pre-clean
    "WAFER CLEAN PRE PROCESS":        "PRE CLEAN WAFER",

    # Initial inspection
    "PRE CLEAN INSPECTION":           "INITIAL WAFER INSPECTION",

    # Drying
    "DRY WAFER BACKSIDE":             "DRY WAFER",

    # Backside cleans
    "BACKSIDE CLEAN FINAL":           "BACKSIDE CLEAN",
    "FRONTSIDE CLEAN FINAL":          "FRONTSIDE CLEAN",

    # Anneals on poly
    "ANNEAL POLYSILICON":             "POLYSILICON ANNEAL",

    # Parametric test
    "ELECTRICAL PARAMETRIC TEST":     "PARAMETRIC TEST",

    # Final clean measurements
    "FINAL THICKNESS MEASURE":        "MEASURE THICKNESS",

    # Inspection variants
    "POLY PATTERN INSPECTION":        "INSPECT PATTERN LEVEL 2",
    "P BODY WINDOW INSPECTION":       "INSPECT PATTERN LEVEL 2",
    "FIELD PATTERN INSPECTION":       "INSPECT PATTERN LEVEL 3",
    "METAL PATTERN INSPECTION":       "INSPECT PATTERN LEVEL 4",
    "VIA INSPECTION":                 "INSPECT PATTERN LEVEL 3",
    "VIA OPENING INSPECTION":         "INSPECT PATTERN LEVEL 3",
    "PATTERN INSPECTION LEVEL 1":     "INSPECT PATTERN LEVEL 1",
    "PATTERN INSPECTION LEVEL 2":     "INSPECT PATTERN LEVEL 2",
}


def canonicalize_step(step: str) -> str:
    """Return the canonical synonym for a single step."""
    return CANONICAL.get(step, step)


def canonicalize_sequence(steps: list[str]) -> list[str]:
    """Canonicalize every step in a sequence."""
    return [CANONICAL.get(s, s) for s in steps]


# Inverse map: canonical_form -> [all_synonyms_including_self]
# Built once. Used by training-time synonym randomization to teach the
# model that synonyms are equivalent classes.
import random as _random
from collections import defaultdict as _defaultdict

_SYNONYM_GROUPS: dict[str, list[str]] | None = None


def _build_synonym_groups() -> dict[str, list[str]]:
    """Group every step by its canonical form. The canonical form itself is
    a synonym of itself in each group."""
    groups: dict[str, list[str]] = _defaultdict(list)
    # Every alias maps to its canonical form
    for alias, canon in CANONICAL.items():
        groups[canon].append(alias)
    # The canonical form itself is a valid synonym
    for canon in list(groups.keys()):
        if canon not in groups[canon]:
            groups[canon].append(canon)
    return dict(groups)


def randomize_synonyms(steps: list[str], rng: _random.Random,
                        prob: float = 0.5) -> list[str]:
    """Per step, with probability `prob`, swap to a random synonym from its
    equivalence class. Used at training time to teach the model that
    e.g. STRIP RESIST and STRIP PHOTORESIST are interchangeable.

    This is the *inverse* of `canonicalize_sequence`: rather than
    collapsing synonyms onto a single form, we expand the model's
    exposure to all surface forms during training.
    """
    global _SYNONYM_GROUPS
    if _SYNONYM_GROUPS is None:
        _SYNONYM_GROUPS = _build_synonym_groups()
    out: list[str] = []
    for s in steps:
        # Find the equivalence class containing this step
        canon = CANONICAL.get(s, s)
        group = _SYNONYM_GROUPS.get(canon, [canon])
        if rng.random() < prob and len(group) > 1:
            out.append(rng.choice(group))
        else:
            out.append(s)
    return out
