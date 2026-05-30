"""Data loading, vocabulary, LoFO splits, and role encoding.

Imports only the stdlib + the official reader; no torch. The neural model reads
encoded ids from here but encoding itself is dependency-light.
"""
from __future__ import annotations

import functools
from typing import Iterable, Optional

from nspe.official import FAMILIES, FAMILY_FILES, read_csv_sequences
from nspe.roles import role_idx, NUM_ROLES

# Leave-one-family-out splits: (held_out, [train_families]).
LOFO_SPLITS = [
    ("mosfet", ["igbt", "ic"]),
    ("igbt", ["mosfet", "ic"]),
    ("ic", ["mosfet", "igbt"]),
]


@functools.lru_cache(maxsize=None)
def load_family(family: str) -> tuple:
    """Return a tuple of sequences (each a tuple[str,...]) for one family."""
    family = family.lower()
    seqs = read_csv_sequences(FAMILY_FILES[family])
    return tuple(tuple(s) for s in seqs.values())


def load_families(families: Iterable[str]) -> dict:
    return {f.lower(): load_family(f) for f in families}


def all_sequences(families: Optional[Iterable[str]] = None) -> list:
    """List of (family, sequence) pairs."""
    families = list(families) if families is not None else list(FAMILIES)
    out = []
    for f in families:
        f = f.lower()
        for seq in load_family(f):
            out.append((f, list(seq)))
    return out


@functools.lru_cache(maxsize=None)
def step_vocab(families: tuple = FAMILIES) -> tuple:
    """Sorted tuple of distinct step strings across the given families."""
    vocab: set = set()
    for f in families:
        for seq in load_family(f):
            vocab.update(seq)
    return tuple(sorted(vocab))


def candidate_vocab(families: Optional[Iterable[str]] = None) -> frozenset:
    """Set of steps to consider as next-step candidates."""
    fams = tuple(f.lower() for f in (families if families is not None else FAMILIES))
    return frozenset(step_vocab(fams))


def build_step_index(families: tuple = FAMILIES) -> tuple:
    """(id_to_step, step_to_id) with specials. Index 0 = <PAD>, 1 = <BOS>, 2 = <EOS>."""
    specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    id_to_step = specials + list(step_vocab(tuple(families)))
    step_to_id = {s: i for i, s in enumerate(id_to_step)}
    return id_to_step, step_to_id


def role_encode(seq: Iterable[str]) -> list:
    """Parallel role-id stream for a step sequence (used as a model feature)."""
    return [role_idx(s) for s in seq]


__all__ = [
    "FAMILIES", "LOFO_SPLITS", "NUM_ROLES",
    "load_family", "load_families", "all_sequences", "step_vocab",
    "candidate_vocab", "build_step_index", "role_encode",
]
