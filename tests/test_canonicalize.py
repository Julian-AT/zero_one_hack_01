"""Tests for synonym canonicalization."""

from __future__ import annotations

from transformer_xlstm.data.canonicalize import CANONICAL, canonicalize_sequence, canonicalize_step


def test_known_synonym_maps_to_canonical():
    assert canonicalize_step("STRIP RESIST") == "STRIP PHOTORESIST"
    assert canonicalize_step("WET CLEAN RCA1") == "RCA CLEAN 1"


def test_unknown_step_is_unchanged():
    assert canonicalize_step("DEPOSIT POLYSILICON") == "DEPOSIT POLYSILICON"


def test_canonical_form_is_idempotent():
    # Mapping a canonical target again must not change it.
    for canonical in set(CANONICAL.values()):
        assert canonicalize_step(canonical) == canonical


def test_sequence_canonicalizes_each_step():
    seq = ["STRIP RESIST", "DEPOSIT POLYSILICON", "METAL ETCH DRY"]
    assert canonicalize_sequence(seq) == [
        "STRIP PHOTORESIST",
        "DEPOSIT POLYSILICON",
        "METAL ETCH",
    ]


def test_empty_sequence():
    assert canonicalize_sequence([]) == []
