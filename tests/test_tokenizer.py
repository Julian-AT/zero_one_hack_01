"""Tokenizer tests: round-trip encode/decode for both modes.

Build tokenizers from the in-repo variants CSVs (the same source the trainer
uses), so these are deterministic and need no network or checkpoints.
"""

from __future__ import annotations

import pytest

from src.data.tokenizer import (
    SPECIAL_TOKENS,
    CompositionalTokenizer,
    StepTokenizer,
    build_tokenizer,
    split_step_to_words,
)


@pytest.fixture(scope="module")
def step_tok() -> StepTokenizer:
    return StepTokenizer.from_variants_csvs()


@pytest.fixture(scope="module")
def comp_tok() -> CompositionalTokenizer:
    return CompositionalTokenizer.from_variants_csvs()


def test_special_tokens_have_stable_ids(step_tok):
    assert step_tok.pad_id == 0
    assert step_tok.bos_id == 1
    assert step_tok.eos_id == 2
    for name in SPECIAL_TOKENS:
        assert name in step_tok.token_to_id


def test_build_tokenizer_factory():
    assert build_tokenizer("step").mode == "step"
    assert build_tokenizer("compositional").mode == "compositional"
    with pytest.raises(ValueError):
        build_tokenizer("nonsense")


def test_family_ids_distinct(step_tok):
    ids = {step_tok.family_id(f) for f in ["mosfet", "igbt", "ic", "unk"]}
    assert len(ids) == 4


def test_step_tokenizer_roundtrip_known_steps(step_tok):
    # Pick real steps from the learned vocab (skip specials).
    known = [t for t in step_tok.id_to_token if t not in SPECIAL_TOKENS][:6]
    ids = step_tok.encode_steps(known)
    assert step_tok.decode_to_steps(ids) == known


def test_step_tokenizer_unknown_maps_to_unk(step_tok):
    ids = step_tok.encode_steps(["DEFINITELY NOT A REAL STEP"])
    assert ids == [step_tok.unk_id]


def test_compositional_roundtrip_known_steps(comp_tok):
    steps = ["DEPOSIT POLYSILICON", "ALIGN MASK LEVEL 2"]
    # Only assert round-trip for steps whose words are all in-vocab.
    in_vocab = [s for s in steps if all(w in comp_tok.token_to_id for w in split_step_to_words(s))]
    ids = comp_tok.encode_steps(in_vocab)
    assert comp_tok.decode_to_steps(ids) == in_vocab


def test_compositional_step_delimiter_present(comp_tok):
    ids = comp_tok.encode_steps(["CURE PASSIVATION"])
    assert ids[-1] == comp_tok.step_id


def test_split_step_to_words():
    assert split_step_to_words("ALIGN MASK LEVEL 2") == ["ALIGN", "MASK", "LEVEL", "2"]
    assert split_step_to_words("  ") == []
