"""Two tokenization modes for process sequences:

1. StepTokenizer (step-as-token): each unique step string → one token.
   Vocab ≈ 198. Simple baseline.

2. CompositionalTokenizer (word-as-token): each step is split into word tokens
   plus a <STEP> boundary marker. Vocab ≈ 70 words + delimiters. Designed to
   generalize across unseen step strings in family 4 (the OOD lever).

Both implement the same interface so the trainer is tokenization-agnostic.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from src.data.validator import read_csv_sequences
from src.utils.paths import FAMILY_FILES

# Special tokens — shared by both tokenizers (identical IDs across modes).
SPECIAL_TOKENS = [
    "<PAD>",        # 0
    "<BOS>",        # 1
    "<EOS>",        # 2
    "<MASK>",       # 3
    "<UNK>",        # 4
    "<STEP>",       # 5   end-of-step delimiter (compositional only, but reserved in both)
    "<FAMILY_MOSFET>",  # 6
    "<FAMILY_IGBT>",    # 7
    "<FAMILY_IC>",      # 8
    "<FAMILY_UNK>",     # 9   used during family-dropout and for OOD
]

FAMILY_TOKEN = {
    "mosfet": "<FAMILY_MOSFET>",
    "igbt":   "<FAMILY_IGBT>",
    "ic":     "<FAMILY_IC>",
    "unk":    "<FAMILY_UNK>",
}

# Words that come from the step strings (split on whitespace + a few separators).
_WORD_SPLIT_RE = re.compile(r"[\s/\-_]+")


def split_step_to_words(step: str) -> list[str]:
    """Tokenize a step string into words for compositional encoding.

    Rules:
      - Split on whitespace, slashes, hyphens, underscores.
      - Lowercase numeric suffixes get standalone tokens (e.g. "LEVEL 2" → [LEVEL, 2]).
      - Keep alphanumeric tokens.
    """
    parts = [p for p in _WORD_SPLIT_RE.split(step.strip()) if p]
    return parts


# --------------------------------------------------------------------------- #
# Base interface                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class BaseTokenizer(ABC):
    """Common interface for step + compositional tokenizers."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)
    mode: str = "base"  # overridden by subclass

    # ---- accessors ----
    @property
    def vocab_size(self) -> int: return len(self.id_to_token)
    @property
    def pad_id(self) -> int:  return self.token_to_id["<PAD>"]
    @property
    def bos_id(self) -> int:  return self.token_to_id["<BOS>"]
    @property
    def eos_id(self) -> int:  return self.token_to_id["<EOS>"]
    @property
    def mask_id(self) -> int: return self.token_to_id["<MASK>"]
    @property
    def unk_id(self) -> int:  return self.token_to_id["<UNK>"]
    @property
    def step_id(self) -> int: return self.token_to_id["<STEP>"]

    def family_id(self, family: str) -> int:
        return self.token_to_id[FAMILY_TOKEN[family.lower()]]

    # ---- encoding / decoding ----
    @abstractmethod
    def encode_steps(self, steps: Sequence[str]) -> list[int]: ...
    @abstractmethod
    def decode_to_steps(self, ids: Sequence[int]) -> list[str]: ...

    def wrap_with_family(self, family: str, ids: Sequence[int]) -> list[int]:
        """Prepend [BOS, FAMILY] and append [EOS]."""
        return [self.bos_id, self.family_id(family), *ids, self.eos_id]

    # ---- persistence ----
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"mode": self.mode, "id_to_token": self.id_to_token}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BaseTokenizer":
        with path.open() as f:
            data = json.load(f)
        if data["mode"] == "step":
            tok = StepTokenizer()
        elif data["mode"] == "compositional":
            tok = CompositionalTokenizer()
        else:
            raise ValueError(f"unknown tokenizer mode: {data['mode']}")
        tok.id_to_token = data["id_to_token"]
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok


# --------------------------------------------------------------------------- #
# Step-as-token tokenizer                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class StepTokenizer(BaseTokenizer):
    """Each unique step string maps to exactly one token."""

    mode: str = "step"

    @classmethod
    def from_variants_csvs(cls, paths: Iterable[Path] | None = None) -> "StepTokenizer":
        paths = list(paths) if paths is not None else list(FAMILY_FILES.values())
        steps: set[str] = set()
        for p in paths:
            seqs = read_csv_sequences(p)
            for seq in seqs.values():
                steps.update(seq)
        ordered = list(SPECIAL_TOKENS) + sorted(steps)
        tok = cls(mode="step")
        tok.id_to_token = ordered
        tok.token_to_id = {t: i for i, t in enumerate(ordered)}
        return tok

    def encode_steps(self, steps: Sequence[str]) -> list[int]:
        unk = self.unk_id
        return [self.token_to_id.get(s, unk) for s in steps]

    def decode_to_steps(self, ids: Sequence[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self.id_to_token):
                t = self.id_to_token[i]
                # Skip specials at decode time.
                if t in SPECIAL_TOKENS:
                    continue
                out.append(t)
        return out


# --------------------------------------------------------------------------- #
# Compositional (word-as-token) tokenizer — the OOD lever                     #
# --------------------------------------------------------------------------- #

@dataclass
class CompositionalTokenizer(BaseTokenizer):
    """Each step is split into word tokens followed by a <STEP> delimiter.

    Example:
        "DEPOSIT POLYSILICON" → [DEPOSIT, POLYSILICON, <STEP>]
        "ALIGN MASK LEVEL 2"  → [ALIGN, MASK, LEVEL, 2, <STEP>]

    On decode, we walk forward consuming words until we hit <STEP>, then emit
    one step string per segment. Unknown words become <UNK> at encode time and
    are dropped at decode time.
    """

    mode: str = "compositional"

    @classmethod
    def from_variants_csvs(cls, paths: Iterable[Path] | None = None) -> "CompositionalTokenizer":
        paths = list(paths) if paths is not None else list(FAMILY_FILES.values())
        words: set[str] = set()
        for p in paths:
            seqs = read_csv_sequences(p)
            for seq in seqs.values():
                for step in seq:
                    words.update(split_step_to_words(step))
        ordered = list(SPECIAL_TOKENS) + sorted(words)
        tok = cls(mode="compositional")
        tok.id_to_token = ordered
        tok.token_to_id = {t: i for i, t in enumerate(ordered)}
        return tok

    def encode_steps(self, steps: Sequence[str]) -> list[int]:
        unk = self.unk_id
        step_id = self.step_id
        out: list[int] = []
        for s in steps:
            for w in split_step_to_words(s):
                out.append(self.token_to_id.get(w, unk))
            out.append(step_id)
        return out

    def decode_to_steps(self, ids: Sequence[int]) -> list[str]:
        out: list[str] = []
        current: list[str] = []
        for i in ids:
            if not (0 <= i < len(self.id_to_token)):
                continue
            t = self.id_to_token[i]
            if t == "<STEP>":
                if current:
                    out.append(" ".join(current))
                    current = []
                continue
            if t in SPECIAL_TOKENS:
                continue
            current.append(t)
        if current:
            out.append(" ".join(current))
        return out


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_tokenizer(mode: str) -> BaseTokenizer:
    if mode == "step":
        return StepTokenizer.from_variants_csvs()
    if mode == "compositional":
        return CompositionalTokenizer.from_variants_csvs()
    raise ValueError(f"unknown tokenization mode: {mode!r}; choose 'step' or 'compositional'")


# --------------------------------------------------------------------------- #
# CLI smoke test                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["step", "compositional"], default="step")
    args = p.parse_args()
    tok = build_tokenizer(args.mode)
    print(f"mode={tok.mode}  vocab_size={tok.vocab_size}")
    sample = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "DEPOSIT POLYSILICON",
              "ALIGN MASK LEVEL 2", "STRIP PHOTORESIST"]
    ids = tok.encode_steps(sample)
    print(f"encoded: {ids[:30]} ...")
    print(f"roundtrip: {tok.decode_to_steps(ids)}")
    wrapped = tok.wrap_with_family("mosfet", ids)
    print(f"wrapped len: {len(wrapped)}  starts: {wrapped[:5]}  ends: {wrapped[-3:]}")
