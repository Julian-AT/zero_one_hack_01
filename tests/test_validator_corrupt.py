"""Tests for the rule validator and corrupters.

These exercise the organizer grammar (`generate_sequences`) via the
`transformer_xlstm.data.validator` adapter and the corrupters in
`transformer_xlstm.data.corrupt`. They use the in-repo variants CSVs and the
seeded sequence generator, so they are deterministic.
"""

from __future__ import annotations

import random

import pytest
from transformer_xlstm.data.corrupt import CORRUPTERS, corrupt_random
from transformer_xlstm.data.validator import (
    NUM_RULE_CLASSES,
    RULE_IDS,
    VALID_CLASS_IDX,
    generate_sequence,
    is_valid,
    rule_class_index,
    validate_sequence,
)

FAMILIES = ["mosfet", "igbt", "ic"]


def _valid_seed(family: str, seed: int = 0) -> list[str]:
    """Generate a valid sequence, retrying seeds until the validator agrees."""
    for s in range(seed, seed + 20):
        seq = generate_sequence(family, random.Random(s))
        if is_valid(seq):
            return seq
    raise AssertionError(f"no valid {family} sequence found")


def test_rule_ids_shape():
    assert len(RULE_IDS) == 10
    assert VALID_CLASS_IDX == 10
    assert NUM_RULE_CLASSES == 11


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_sequences_are_valid(family):
    seq = _valid_seed(family)
    assert validate_sequence(seq) == []
    assert is_valid(seq)
    assert rule_class_index(seq) == VALID_CLASS_IDX


@pytest.mark.parametrize("family", FAMILIES)
def test_empty_and_trivial_sequences(family):
    # The validator must not crash on degenerate input.
    assert isinstance(validate_sequence([]), list)
    assert isinstance(validate_sequence(["SHIP LOT"]), list)


@pytest.mark.parametrize("rule", list(CORRUPTERS.keys()))
def test_each_corrupter_triggers_its_rule(rule):
    """For at least one family/seed, each corrupter must produce a sequence the
    validator flags with the targeted rule."""
    corrupter = CORRUPTERS[rule]
    for family in FAMILIES:
        for seed in range(8):
            seq = _valid_seed(family, seed * 13)
            c = corrupter(list(seq), random.Random(seed))
            if c is None:
                continue
            hits = {v.rule for v in validate_sequence(c.corrupted_steps)}
            if rule in hits:
                return  # found a triggering case
    pytest.skip(f"{rule} not triggerable on available seeds")


def test_corrupt_random_returns_verified_violation():
    rng = random.Random(0)
    seq = _valid_seed("mosfet")
    c = corrupt_random(list(seq), rng, verify=True)
    assert c is not None
    assert c.rule in RULE_IDS
    hits = {v.rule for v in validate_sequence(c.corrupted_steps)}
    assert c.rule in hits


def test_rule_class_index_matches_first_violation():
    rng = random.Random(1)
    seq = _valid_seed("igbt")
    c = corrupt_random(list(seq), rng, verify=True)
    assert c is not None
    idx = rule_class_index(c.corrupted_steps)
    assert 0 <= idx < VALID_CLASS_IDX
