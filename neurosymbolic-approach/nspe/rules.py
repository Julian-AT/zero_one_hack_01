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

from nspe.official import gs, validate_sequence, read_csv_sequences, FAMILY_FILES

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


def _novel_trigger_sets(steps):
    """Classify the NOVEL (unknown-vocab) steps in `steps` into validator trigger
    roles, using conservative patterns. Returns (dep, clean, etch, implant) sets."""
    known = known_vocab()
    dep, clean, etch, implant = set(), set(), set(), set()
    for s in {x for x in steps if x not in known}:
        u = s.upper().strip()
        is_inspect = u.endswith("INSPECTION") or u.startswith("MEASURE") or u.endswith("CHECK")
        if u.startswith("DEPOSIT ") or u.endswith(" OXIDATION") or u.endswith(" GROWTH") \
                or u == "EPITAXIAL DEPOSITION" or "OXIDATION" in u:
            dep.add(s)
        elif " ETCH" in u or u.startswith("ETCH "):
            etch.add(s)
        elif u.startswith("IMPLANT "):
            implant.add(s)
        elif (not is_inspect) and (("CLEAN" in u) or u.endswith(" RINSE")
                                   or u.startswith("DRY ") or u == "HF DIP"):
            clean.add(s)
    return dep, clean, etch, implant


def validate(steps):
    """Exact official semantics."""
    return validate_sequence(list(steps))


def validate_with_roles(steps):
    """Official logic with trigger sets augmented by novel-step role induction.

    Identical to `validate` whenever `steps` uses only known vocabulary."""
    steps = list(steps)
    dep, clean, etch, implant = _novel_trigger_sets(steps)
    if not (dep or clean or etch or implant):
        return validate_sequence(steps)
    orig = (gs.DEPOSITION_STEPS, gs.CLEAN_STEPS, gs.ETCH_STEPS, gs.IMPLANT_STEPS,
            gs.FILL_STEPS)
    try:
        gs.DEPOSITION_STEPS = orig[0] | dep
        gs.CLEAN_STEPS = orig[1] | clean
        gs.ETCH_STEPS = orig[2] | etch
        gs.IMPLANT_STEPS = orig[3] | implant
        # FILL_STEPS subsumes deposition (CMP rule); keep it consistent.
        gs.FILL_STEPS = orig[4] | dep
        return validate_sequence(steps)
    finally:
        (gs.DEPOSITION_STEPS, gs.CLEAN_STEPS, gs.ETCH_STEPS,
         gs.IMPLANT_STEPS, gs.FILL_STEPS) = orig


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
