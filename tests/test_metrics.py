"""Tests for evaluation metrics."""

from __future__ import annotations

from transformer_xlstm.eval.metrics import normalized_edit_distance, reciprocal_rank


def test_edit_distance_identical_is_zero():
    assert normalized_edit_distance(["A", "B", "C"], ["A", "B", "C"]) == 0.0


def test_edit_distance_both_empty_is_zero():
    assert normalized_edit_distance([], []) == 0.0


def test_edit_distance_one_empty_is_one():
    assert normalized_edit_distance(["A"], []) == 1.0
    assert normalized_edit_distance([], ["A", "B"]) == 1.0


def test_edit_distance_single_substitution():
    # One of three positions differs -> 1/3.
    assert normalized_edit_distance(["A", "B", "C"], ["A", "X", "C"]) == 1.0 / 3.0


def test_edit_distance_is_symmetric():
    a = ["A", "B", "C", "D"]
    b = ["A", "X", "C"]
    assert normalized_edit_distance(a, b) == normalized_edit_distance(b, a)


def test_edit_distance_normalized_by_longer_length():
    # Append one extra step -> distance 1, normalized by len 3.
    assert normalized_edit_distance(["A", "B", "C"], ["A", "B"]) == 1.0 / 3.0


def test_reciprocal_rank_hits():
    assert reciprocal_rank("B", ["A", "B", "C"]) == 0.5
    assert reciprocal_rank("A", ["A", "B"]) == 1.0


def test_reciprocal_rank_miss_is_zero():
    assert reciprocal_rank("Z", ["A", "B", "C"]) == 0.0


def test_reciprocal_rank_normalizes_case():
    assert reciprocal_rank("  b  ", ["a", "B", "c"]) == 0.5
