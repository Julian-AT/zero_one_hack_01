"""Datasets and DataLoaders for process sequences.

Two modes:

1. `ProcessSequenceDataset` — backed by a pre-loaded list of (family, steps)
   examples. Standard PyTorch Dataset. Used for held-out validation and eval.

2. `OnlineGeneratorIterableDataset` — calls `generate_sequence(family, rng)`
   on each iteration and (optionally) injects a corruption with a labeled
   rule. Provides infinite, always-fresh training data with on-the-fly hard
   negatives. The main training stream per `plan.md` Stage 1.

Both yield batches with the same schema:
    input_ids   [B, L]   long
    labels      [B, L]   long; -100 marks ignored positions
    attn_mask   [B, L]   long; 1 = real, 0 = pad
    family_id   [B]      long; family token id (subject to family_dropout)
    validity    [B]      long; 1 = valid, 0 = invalid
    rule_class  [B]      long; in [0, NUM_RULE_CLASSES) — see validator.py
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.data.canonicalize import canonicalize_sequence, randomize_synonyms
from src.data.corrupt import corrupt_random
from src.data.ood_generator import generate_ood_sequence, random_ood_family
from src.data.tokenizer import BaseTokenizer
from src.data.validator import (
    VALID_CLASS_IDX,
    generate_sequence,
    read_csv_sequences,
    rule_class_index,
)
from src.utils.paths import FAMILY_FILES

IGNORE_INDEX = -100


@dataclass
class Example:
    family: str
    steps: list[str]
    validity: int = 1  # 1 = valid, 0 = corrupted
    rule_class: int = VALID_CLASS_IDX  # which rule was injected (for multi-task)


# --------------------------------------------------------------------------- #
# Data loading from CSV                                                       #
# --------------------------------------------------------------------------- #


def load_family(family: str, path: Path | None = None, canonicalize: bool = False) -> list[Example]:
    p = Path(path) if path is not None else FAMILY_FILES[family.lower()]
    raw = read_csv_sequences(p)
    out: list[Example] = []
    for steps in raw.values():
        s = canonicalize_sequence(steps) if canonicalize else steps
        out.append(Example(family=family.lower(), steps=list(s)))
    return out


def load_all_families(
    families: Iterable[str] | None = None, canonicalize: bool = False
) -> list[Example]:
    fams = list(families) if families is not None else list(FAMILY_FILES.keys())
    out: list[Example] = []
    for fam in fams:
        out.extend(load_family(fam, canonicalize=canonicalize))
    return out


# --------------------------------------------------------------------------- #
# Encoding to tensors                                                         #
# --------------------------------------------------------------------------- #


def encode_example(
    ex: Example,
    tokenizer: BaseTokenizer,
    max_len: int,
    family_dropout: float = 0.0,
    rng: random.Random | None = None,
) -> dict[str, torch.Tensor]:
    """Encode one example into model-input tensors with optional family-token dropout."""
    rng = rng or random
    step_ids = tokenizer.encode_steps(ex.steps)
    fam = ex.family
    if rng.random() < family_dropout:
        fam = "unk"
    family_tok_id = tokenizer.family_id(fam)

    wrapped = [tokenizer.bos_id, family_tok_id, *step_ids, tokenizer.eos_id]
    # Truncate keeping BOS + FAMILY at start, EOS at the end.
    if len(wrapped) > max_len:
        head = wrapped[:2]
        tail = wrapped[-(max_len - 2) :]
        wrapped = head + tail

    pad = max_len - len(wrapped)
    input_ids = wrapped + [tokenizer.pad_id] * pad
    attn_mask = [1] * len(wrapped) + [0] * pad
    labels = list(input_ids)
    labels[0] = IGNORE_INDEX
    labels[1] = IGNORE_INDEX
    for i in range(len(wrapped), max_len):
        labels[i] = IGNORE_INDEX

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
        "family_id": torch.tensor(family_tok_id, dtype=torch.long),
        "validity": torch.tensor(ex.validity, dtype=torch.long),
        "rule_class": torch.tensor(ex.rule_class, dtype=torch.long),
    }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# --------------------------------------------------------------------------- #
# Static-pool dataset (held-out eval, fixed splits)                           #
# --------------------------------------------------------------------------- #


class ProcessSequenceDataset(Dataset):
    """A fixed list of Examples. Used for validation, held-out eval, and LoFO."""

    def __init__(
        self,
        examples: Iterable[Example],
        tokenizer: BaseTokenizer,
        max_len: int = 768,
        family_dropout: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.examples = list(examples)
        self.tok = tokenizer
        self.max_len = max_len
        self.family_dropout = family_dropout
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return encode_example(
            self.examples[idx],
            self.tok,
            self.max_len,
            family_dropout=self.family_dropout,
            rng=self.rng,
        )


# --------------------------------------------------------------------------- #
# Online-generator IterableDataset (Stage-1 training stream)                  #
# --------------------------------------------------------------------------- #


class OnlineGeneratorIterableDataset(IterableDataset):
    """Stream of generated + optionally corrupted sequences.

    Each iteration:
      1. Pick a family uniformly at random (or per ratio).
      2. Call `generate_sequence(family, rng)` → fresh valid sequence.
      3. With prob `corrupt_fraction`, apply `corrupt_random` and label the
         injected rule. Otherwise label as valid.
      4. Optionally canonicalize.
      5. Encode + return.

    No `__len__` (infinite stream). The trainer iterates for a fixed step count.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        families: list[str],
        max_len: int = 768,
        corrupt_fraction: float = 0.25,
        canonicalize: bool = False,
        family_dropout: float = 0.0,
        seed: int = 42,
        ood_family_prob: float = 0.0,
        synonym_randomize_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.families = families
        self.max_len = max_len
        self.corrupt_fraction = corrupt_fraction
        self.canonicalize = canonicalize
        self.family_dropout = family_dropout
        self.seed = seed
        self.ood_family_prob = ood_family_prob
        self.synonym_randomize_prob = synonym_randomize_prob

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id * 9973)

        while True:
            # With ood_family_prob, draw from synthetic OOD families
            # (DIODE / SCHOTTKY / SIC_MOSFET). Labeled as <FAMILY_UNK> so the
            # model treats them as "unknown family" examples — encourages
            # backbone-level learning rather than family-token shortcuts.
            if self.ood_family_prob > 0 and rng.random() < self.ood_family_prob:
                ood_fam = random_ood_family(rng)
                steps = generate_ood_sequence(ood_fam, rng)
                if steps is None:
                    # Defensive: extremely rare. Fall back to a real family.
                    family = rng.choice(self.families)
                    steps = generate_sequence(family, rng)
                else:
                    family = "unk"  # train as <FAMILY_UNK>
            else:
                family = rng.choice(self.families)
                steps = generate_sequence(family, rng)

            ex = Example(family=family, steps=steps, validity=1, rule_class=VALID_CLASS_IDX)

            if rng.random() < self.corrupt_fraction:
                c = corrupt_random(list(steps), rng, verify=True)
                if c is not None:
                    ex = Example(
                        family=family,
                        steps=c.corrupted_steps,
                        validity=0,
                        rule_class=rule_class_index(c.corrupted_steps),
                    )

            if self.canonicalize:
                ex.steps = canonicalize_sequence(ex.steps)
            elif self.synonym_randomize_prob > 0:
                # Mutually exclusive with canonicalize: synonym randomization
                # *expands* exposure to all surface forms, canonicalize
                # *collapses* them. We pick one.
                ex.steps = randomize_synonyms(ex.steps, rng, prob=self.synonym_randomize_prob)

            yield encode_example(
                ex, self.tokenizer, self.max_len, family_dropout=self.family_dropout, rng=rng
            )


# --------------------------------------------------------------------------- #
# Convenience DataLoader builders                                             #
# --------------------------------------------------------------------------- #


def make_static_loader(
    examples: list[Example],
    tokenizer: BaseTokenizer,
    batch_size: int,
    max_len: int = 768,
    shuffle: bool = True,
    num_workers: int = 0,
    family_dropout: float = 0.0,
) -> DataLoader:
    ds = ProcessSequenceDataset(examples, tokenizer, max_len=max_len, family_dropout=family_dropout)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        drop_last=shuffle,
    )


def make_online_loader(
    tokenizer: BaseTokenizer,
    families: list[str],
    batch_size: int,
    max_len: int = 768,
    corrupt_fraction: float = 0.25,
    canonicalize: bool = False,
    family_dropout: float = 0.0,
    num_workers: int = 0,
    seed: int = 42,
    ood_family_prob: float = 0.0,
    synonym_randomize_prob: float = 0.0,
) -> DataLoader:
    ds = OnlineGeneratorIterableDataset(
        tokenizer,
        families,
        max_len=max_len,
        corrupt_fraction=corrupt_fraction,
        canonicalize=canonicalize,
        family_dropout=family_dropout,
        seed=seed,
        ood_family_prob=ood_family_prob,
        synonym_randomize_prob=synonym_randomize_prob,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )


if __name__ == "__main__":
    # Smoke test: build tokenizer, draw a few online batches, print shapes.
    from src.data.tokenizer import build_tokenizer

    tok = build_tokenizer("step")
    loader = make_online_loader(
        tok,
        families=["mosfet", "igbt", "ic"],
        batch_size=4,
        max_len=768,
        corrupt_fraction=0.5,
        num_workers=0,
        seed=0,
    )
    it = iter(loader)
    for i in range(2):
        batch = next(it)
        print(f"batch {i}: " + ", ".join(f"{k}={tuple(v.shape)}" for k, v in batch.items()))
        print(
            f"   validity={batch['validity'].tolist()}  rule_class={batch['rule_class'].tolist()}"
        )
