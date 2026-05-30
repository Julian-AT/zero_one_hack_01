"""Rule checking: exact official validator, a role-augmented validator for OOD,
an incremental prefix check for constrained decoding, and rule attribution.

Key correctness property of `validate_with_roles`: it augments the validator's
trigger frozensets ONLY with genuinely *novel* (unknown-vocabulary) steps. On any
sequence built from known vocabulary, the novel set is empty, so
`validate_with_roles(steps) == validate(steps)` exactly. This guarantees zero
false positives / zero suppression on the known families and makes the role
augmentation strictly additive for the unseen 4th family.
"""
from __future__ import annotations

import functools

from nspe.official import validate_sequence, read_csv_sequences, FAMILY_FILES

RULE_IDS = [
    "RULE_DEP_NO_CLEAN", "RULE_METAL_ETCH_NO_LITHO", "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP", "RULE_IMPLANT_NO_MASK", "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP", "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST", "RULE_BACKSIDE_BEFORE_PASSIVATION",
]
RULE_TO_IDX = {r: i for i, r in enumerate(RULE_IDS)}


@functools.lru_cache(maxsize=1)
def known_vocab() -> frozenset:
    """All step strings observed across the three provided families (~198)."""
    vocab: set[str] = set()
    for path in FAMILY_FILES.values():
        for seq in read_csv_sequences(path).values():
            vocab.update(seq)
    return frozenset(vocab)


# ---------------------------------------------------------------------------
# Role-induction anchors: canonicalize NOVEL (unseen-family) steps to the
# canonical landmark name the official validator recognizes, so all 10 rules —
# including the ones that key on hardcoded string anchors (CURE PASSIVATION,
# SHIP LOT, WAFER SORT TEST, DEVELOP PHOTORESIST, ...) — survive a 4th family
# that *renames* those steps. We only ever touch steps that are NOT in the known
# vocabulary, so on any known-vocab sequence canonicalization is a no-op and
# `validate_with_roles(s) == validate(s)` holds exactly.
# ---------------------------------------------------------------------------

# Canonical representatives (all are known-vocab steps the validator recognizes).
_CANON = {
    "cure_passivation":  "CURE PASSIVATION",      # anchor for rules 7, 8, 10
    "dep_passivation":   "DEPOSIT PASSIVATION",   # anchor for rule 7
    "dep_backside_metal": "DEPOSIT BACKSIDE METAL",  # trigger for rule 10
    "dep_generic":       "DEPOSIT POLYSILICON",   # deposition trigger (rule 1) / fill (rule 6)
    "pad_window":        "OPEN PAD WINDOW",        # trigger for rule 7
    "develop":           "DEVELOP PHOTORESIST",    # prereq for rules 2, 3, 5
    "metal_etch":        "METAL ETCH",             # trigger for rules 2, 3
    "etch_generic":      "OXIDE ETCH",             # etch trigger (rule 3) / implant opener (rule 5)
    "implant":           "IMPLANT WELL",           # trigger for rule 5
    "cmp":               "CMP METAL",              # trigger for rule 6
    "fill":              "FILL VIA METAL",         # CMP prereq (rule 6)
    "ship":              "SHIP LOT",               # rule 9
    "sort_test":         "WAFER SORT TEST",        # rule 9
    "electrical_test":   "LEAKAGE TEST",           # trigger for rule 8
    "clean":             "PRE CLEAN WAFER",        # deposition prereq (rule 1)
}

# Electrical-test keywords: only canonicalize a renamed "* TEST" to an electrical
# test when it clearly is one, to avoid turning an unrelated novel "* TEST" into a
# rule-8 trigger (false-positive guard).
_ELECTRICAL_KW = ("LEAKAGE", "SWITCHING", "PARAMETRIC", "THRESHOLD",
                  "BREAKDOWN", "VOLTAGE", "ELECTRICAL")


def _canonical_landmark(u: str):
    """Map an UPPER-cased novel step string to a canonical landmark literal, or
    None if it does not confidently match one. Most specific patterns first."""
    if "CURE" in u and "PASSIVAT" in u:
        return _CANON["cure_passivation"]
    if u.startswith("DEPOSIT") and "PASSIVAT" in u:
        return _CANON["dep_passivation"]
    if u.startswith("DEPOSIT") and "BACKSIDE" in u and "METAL" in u:
        return _CANON["dep_backside_metal"]
    if (u.startswith("DEPOSIT ") or "OXIDATION" in u
            or u.endswith(" GROWTH") or u == "EPITAXIAL DEPOSITION"):
        return _CANON["dep_generic"]
    # DEVELOP must be checked before pad-window: "DEVELOP PAD WINDOW" is a
    # *develop* step in the official grammar, NOT a pad-window opener.
    if "DEVELOP" in u:
        return _CANON["develop"]
    if "PAD" in u and "WINDOW" in u:
        return _CANON["pad_window"]
    if "METAL" in u and "ETCH" in u:
        return _CANON["metal_etch"]
    if " ETCH" in u or u.startswith("ETCH "):
        return _CANON["etch_generic"]
    if u.startswith("IMPLANT "):
        return _CANON["implant"]
    if u.startswith("CMP ") or "PLANAR" in u:
        return _CANON["cmp"]
    if u.startswith("FILL VIA"):
        return _CANON["fill"]
    if "SHIP" in u and "LOT" in u:
        return _CANON["ship"]
    if "WAFER SORT" in u or ("SORT" in u and "TEST" in u):
        return _CANON["sort_test"]
    # Clean (exclude inspection/measurement which can contain "CLEAN").
    is_inspect = (u.endswith("INSPECTION") or u.startswith(("MEASURE", "INSPECT"))
                  or u.endswith("CHECK"))
    if (not is_inspect) and (("CLEAN" in u) or u.endswith(" RINSE")
                             or u.startswith("RINSE ") or u.startswith("DRY ")
                             or u == "HF DIP"):
        return _CANON["clean"]
    # Electrical test (after sort_test; gated on an electrical keyword).
    if u.endswith(" TEST") and any(k in u for k in _ELECTRICAL_KW):
        return _CANON["electrical_test"]
    return None


def _canonicalize_novel(steps):
    """Return (canonicalized_steps, changed). Only NOVEL (unknown-vocab) steps are
    rewritten to their canonical landmark; positions are preserved 1:1."""
    known = known_vocab()
    out = []
    changed = False
    for s in steps:
        if s in known:
            out.append(s)
            continue
        canon = _canonical_landmark(s.upper().strip())
        if canon is not None:
            out.append(canon)
            changed = True
        else:
            out.append(s)
    return out, changed


def validate(steps):
    """Exact official semantics."""
    return validate_sequence(list(steps))


def validate_with_roles(steps):
    """Official logic, but NOVEL steps are first canonicalized to the landmark
    names the validator recognizes (role-induction anchors). This makes all 10
    rules robust to an unseen 4th family that renames trigger/anchor steps.

    Identical to `validate` whenever `steps` uses only known vocabulary (no novel
    step => no canonicalization)."""
    steps = list(steps)
    canon, changed = _canonicalize_novel(steps)
    if not changed:
        return validate_sequence(steps)
    return validate_sequence(canon)


def is_valid(steps, use_roles: bool = True) -> bool:
    v = validate_with_roles(steps) if use_roles else validate(steps)
    return len(v) == 0


def first_rule(steps, use_roles: bool = True):
    """Rule id of the first (earliest) violation, or None if valid."""
    v = validate_with_roles(steps) if use_roles else validate(steps)
    if not v:
        return None
    return min(v, key=lambda x: x.step_index).rule


def would_violate(prefix, candidate, use_roles: bool = False) -> bool:
    """True iff appending `candidate` to an already-valid `prefix` introduces a
    violation AT the candidate's position (index == len(prefix)).

    Sound because every candidate-induced violation lands at the candidate:
      - windowed rules (dep/etch/implant/cmp/metal-etch) look back from their
        trigger; if the candidate is the trigger, the violation index is len(prefix).
      - global-order rules (ship/test/pad-open/backside/litho-skip) fire at the
        offending step, which is the candidate.
    Appending later history can never retroactively break an earlier valid step.
    """
    seq = list(prefix)
    j = len(seq)
    seq.append(candidate)
    v = validate_with_roles(seq) if use_roles else validate_sequence(seq)
    return any(viol.step_index == j for viol in v)


__all__ = [
    "RULE_IDS", "RULE_TO_IDX", "known_vocab", "validate", "validate_with_roles",
    "is_valid", "first_rule", "would_violate",
]
