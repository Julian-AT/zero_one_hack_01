#!/usr/bin/env python3
"""
generate_hard_invalid_sequences.py — Hard invalid near-miss generation.

Place this file in:
    tracks/industrial-infineon/data/generate_hard_invalid_sequences.py

Purpose
-------
Generate hard invalid semiconductor process sequences by perturbing realistic
valid process regions.

Compared to generate_invalid_sequences.py, this script avoids mostly-easy
"insert invalid step at index 3" examples. Instead, it creates near-misses such as:

- valid clean -> deposition becomes clean -> many neutral steps -> deposition
- valid develop -> etch becomes develop -> many neutral steps -> etch
- valid fill -> CMP becomes fill -> many neutral steps -> CMP
- pad window is moved just before CURE PASSIVATION
- electrical test is moved just before CURE PASSIVATION
- SHIP LOT is moved before WAFER SORT TEST
- lithography level is changed to skip/decrease a level

Recommended smoke test:
    python generate_hard_invalid_sequences.py \
      --valid-input coverage_guided_v1/coverage_guided_sequences.csv \
      --target-per-rule-family 20 \
      --max-tries-per-rule-family 10000 \
      --output-dir hard_invalid_test \
      --write-mixed \
      --seed 42

Recommended production run:
    python generate_hard_invalid_sequences.py \
      --valid-input coverage_guided_v1/coverage_guided_sequences.csv \
      --target-per-rule-family 1000 \
      --max-tries-per-rule-family 300000 \
      --max-validator-violations 4 \
      --output-dir hard_invalid_v1 \
      --write-mixed \
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Robust local imports
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
TRAINING_DATA_DIR = PROJECT_DIR / "training_data"

if str(TRAINING_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DATA_DIR))

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from generate_sequences import (  # type: ignore
        validate_sequence,
        DEPOSITION_STEPS,
        CLEAN_STEPS,
        ETCH_STEPS,
        METAL_ETCH_STEPS,
        IMPLANT_STEPS,
        IMPLANT_OPENER_STEPS,
        CMP_STEPS,
        FILL_STEPS,
        PAD_WINDOW_STEPS,
        ELECTRICAL_TEST_STEPS,
        BACKSIDE_METAL_STEPS,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import validator and rule constants from "
        f"{TRAINING_DATA_DIR / 'generate_sequences.py'}.\n"
        "Make sure generate_sequences.py is in tracks/industrial-infineon/training_data/."
    ) from exc

try:
    from coverage_tracker import (  # type: ignore
        SequenceRecord,
        read_sequences_from_csv,
        compute_coverage,
        build_undercovered_targets,
        write_outputs,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import coverage utilities from coverage_tracker.py.\n"
        "Make sure coverage_tracker.py is in tracks/industrial-infineon/data/."
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAMILIES = ("mosfet", "igbt", "ic")

RULES = [
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
]

# Existing, mostly neutral process-measurement steps.
# These are used to push a required dependency outside a validator window.
# They should NOT be clean/develop/deposition/fill/implant/opening/CMP/test steps.
NEUTRAL_FILLER_STEPS = [
    "MEASURE FILM THICKNESS",
    "MEASURE SURFACE PLANARITY",
    "MEASURE LINE WIDTH",
    "MEASURE OXIDE THICKNESS",
    "MEASURE SHEET RESISTANCE",
    "MEASURE JUNCTION DEPTH",
    "MEASURE OPENING CD",
    "MEASURE GATE CD",
    "MEASURE VIA CD",
    "MEASURE SURFACE UNIFORMITY",
    "MEASURE DEVICE PARAMETER",
    "MEASURE POLY THICKNESS",
    "MEASURE METAL THICKNESS",
    "MEASURE CONTACT RESISTANCE",
    "MEASURE VIA RESISTANCE",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MutationResult:
    steps: list[str]
    mutation_type: str
    mutation_index: int
    target_step: str
    original_index: int


@dataclass
class InvalidSample:
    sequence_id: str
    family: str
    steps: list[str]
    violated_rule: str
    validator_rules: list[str]
    validator_violation_count: int
    mutation_type: str
    mutation_index: int
    target_step: str
    original_index: int
    original_sequence_id: str
    seed: int


@dataclass
class RuleFamilyStats:
    family: str
    rule: str
    target: int
    accepted: int
    attempts: int
    skipped_no_mutation: int
    skipped_not_intended_rule: int
    skipped_too_many_violations: int
    skipped_duplicate: int
    skipped_too_early: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def clean_id(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def sequence_hash(steps: list[str]) -> str:
    return hashlib.sha1("\n".join(steps).encode("utf-8")).hexdigest()


def stable_split(sequence_id: str, seed: int) -> str:
    h = int(hashlib.md5(f"{sequence_id}|{seed}".encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def infer_family(record: SequenceRecord) -> str:
    fam = (record.family or "").lower()
    if fam in FAMILIES:
        return fam

    text = f"{record.sequence_id} {record.source_file}".lower()
    for candidate in FAMILIES:
        if candidate in text:
            return candidate

    return "unknown"


def first_index(steps: list[str], target: str) -> Optional[int]:
    for i, step in enumerate(steps):
        if step == target:
            return i
    return None


def find_indices(steps: list[str], predicate: Callable[[str], bool]) -> list[int]:
    return [i for i, step in enumerate(steps) if predicate(step)]


def previous_indices_within(
    steps: list[str],
    index: int,
    targets: set[str] | frozenset[str],
    window: int,
) -> list[int]:
    lo = max(0, index - window)
    return [j for j in range(lo, index) if steps[j] in targets]


def previous_indices_within_predicate(
    steps: list[str],
    index: int,
    predicate: Callable[[str], bool],
    window: int,
) -> list[int]:
    lo = max(0, index - window)
    return [j for j in range(lo, index) if predicate(steps[j])]


def move_step(steps: list[str], from_index: int, to_index: int) -> list[str]:
    new_steps = list(steps)
    step = new_steps.pop(from_index)

    if to_index > from_index:
        to_index -= 1

    to_index = max(0, min(to_index, len(new_steps)))
    new_steps.insert(to_index, step)
    return new_steps


def insert_steps_before(steps: list[str], index: int, inserted: list[str]) -> list[str]:
    return list(steps[:index]) + list(inserted) + list(steps[index:])


def neutral_fillers(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(NEUTRAL_FILLER_STEPS) for _ in range(n)]


def choose_index(
    steps: list[str],
    rng: random.Random,
    predicate: Callable[[int, str], bool],
) -> Optional[int]:
    candidates = [i for i, step in enumerate(steps) if predicate(i, step)]
    if not candidates:
        return None
    return rng.choice(candidates)


def replace_nearby_expose_level(
    steps: list[str],
    align_index: int,
    old_level: int,
    new_level: int,
) -> list[str]:
    new_steps = list(steps)

    next_align = len(new_steps)
    for j in range(align_index + 1, len(new_steps)):
        if new_steps[j].startswith("ALIGN MASK LEVEL "):
            next_align = j
            break

    old_expose = f"EXPOSE LITHO LEVEL {old_level}"
    new_expose = f"EXPOSE LITHO LEVEL {new_level}"

    for j in range(align_index + 1, next_align):
        if new_steps[j] == old_expose:
            new_steps[j] = new_expose
            break

    return new_steps


def validator_rule_set(steps: list[str]) -> tuple[list[str], int]:
    violations = validate_sequence(steps)
    rules = sorted({v.rule for v in violations})
    return rules, len(violations)


# ---------------------------------------------------------------------------
# Hard mutators
# ---------------------------------------------------------------------------

def hard_RULE_DEP_NO_CLEAN(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Pick a real deposition step that currently has a clean in the prior 12 steps.
    Insert 13 neutral steps immediately before the deposition, making the clean
    too stale while preserving the original local process structure.
    """
    index = choose_index(
        steps,
        rng,
        lambda i, s: (
            i >= min_mutation_index
            and s in DEPOSITION_STEPS
            and bool(previous_indices_within(steps, i, CLEAN_STEPS, 12))
        ),
    )
    if index is None:
        return None

    inserted = neutral_fillers(rng, 13)
    mutated = insert_steps_before(steps, index, inserted)

    return MutationResult(
        steps=mutated,
        mutation_type="stale_clean_before_real_deposition_insert_13_neutral_steps",
        mutation_index=index,
        target_step=steps[index],
        original_index=index,
    )


def hard_RULE_METAL_ETCH_NO_LITHO(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Pick a real metal etch that currently has lithography/develop nearby.
    Insert 16 neutral steps before it so expose/develop are outside the 15-step
    metal-etch window.
    """
    index = choose_index(
        steps,
        rng,
        lambda i, s: (
            i >= min_mutation_index
            and s in METAL_ETCH_STEPS
            and bool(previous_indices_within_predicate(
                steps,
                i,
                lambda x: x.startswith("EXPOSE LITHO LEVEL"),
                15,
            ))
            and bool(previous_indices_within(
                steps,
                i,
                frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}),
                15,
            ))
        ),
    )
    if index is None:
        return None

    inserted = neutral_fillers(rng, 16)
    mutated = insert_steps_before(steps, index, inserted)

    return MutationResult(
        steps=mutated,
        mutation_type="stale_lithography_before_real_metal_etch_insert_16_neutral_steps",
        mutation_index=index,
        target_step=steps[index],
        original_index=index,
    )


def hard_RULE_ETCH_NO_MASK(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Pick a real non-metal patterned etch that currently has DEVELOP PHOTORESIST
    within 12 steps. Insert 13 neutral steps before the etch so the mask is stale.
    """
    index = choose_index(
        steps,
        rng,
        lambda i, s: (
            i >= min_mutation_index
            and s in ETCH_STEPS
            and s not in METAL_ETCH_STEPS
            and bool(previous_indices_within(
                steps,
                i,
                frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}),
                12,
            ))
        ),
    )
    if index is None:
        return None

    inserted = neutral_fillers(rng, 13)
    mutated = insert_steps_before(steps, index, inserted)

    return MutationResult(
        steps=mutated,
        mutation_type="stale_mask_before_real_etch_insert_13_neutral_steps",
        mutation_index=index,
        target_step=steps[index],
        original_index=index,
    )


def hard_RULE_LITHO_LEVEL_SKIP(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Change a real lithography level in the middle of the process to skip or
    decrease a level. This preserves most of the lithography block.
    """
    aligns: list[tuple[int, int]] = []
    prefix = "ALIGN MASK LEVEL "

    for i, step in enumerate(steps):
        if i < min_mutation_index:
            continue
        if step.startswith(prefix):
            suffix = step.split(prefix, 1)[1]
            if suffix.isdigit():
                aligns.append((i, int(suffix)))

    if len(aligns) < 2:
        return None

    pos = rng.randrange(1, len(aligns))
    prev_level = aligns[pos - 1][1]
    align_index, old_level = aligns[pos]

    if rng.random() < 0.75:
        new_level = prev_level + 2
        mutation_kind = "skip_up"
    else:
        new_level = max(1, prev_level - 1)
        mutation_kind = "decrease"

    if new_level == old_level:
        new_level = old_level + 2
        mutation_kind = "skip_up_forced"

    mutated = list(steps)
    mutated[align_index] = f"ALIGN MASK LEVEL {new_level}"
    mutated = replace_nearby_expose_level(
        mutated,
        align_index=align_index,
        old_level=old_level,
        new_level=new_level,
    )

    return MutationResult(
        steps=mutated,
        mutation_type=f"hard_litho_level_{mutation_kind}_from_{old_level}_to_{new_level}",
        mutation_index=align_index,
        target_step=steps[align_index],
        original_index=align_index,
    )


def hard_RULE_IMPLANT_NO_MASK(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Pick a real implant that currently has an opener in the prior 15 steps.
    Insert 16 neutral steps before the implant, making the opener too stale.
    """
    index = choose_index(
        steps,
        rng,
        lambda i, s: (
            i >= min_mutation_index
            and s in IMPLANT_STEPS
            and bool(previous_indices_within(steps, i, IMPLANT_OPENER_STEPS, 15))
        ),
    )
    if index is None:
        return None

    inserted = neutral_fillers(rng, 16)
    mutated = insert_steps_before(steps, index, inserted)

    return MutationResult(
        steps=mutated,
        mutation_type="stale_implant_window_before_real_implant_insert_16_neutral_steps",
        mutation_index=index,
        target_step=steps[index],
        original_index=index,
    )


def hard_RULE_CMP_NO_DEP(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Pick a real CMP step that currently has deposition/fill in the prior 6 steps.
    Insert 7 neutral steps before CMP, making the fill too stale.
    """
    index = choose_index(
        steps,
        rng,
        lambda i, s: (
            i >= min_mutation_index
            and s in CMP_STEPS
            and bool(previous_indices_within(steps, i, FILL_STEPS, 6))
        ),
    )
    if index is None:
        return None

    inserted = neutral_fillers(rng, 7)
    mutated = insert_steps_before(steps, index, inserted)

    return MutationResult(
        steps=mutated,
        mutation_type="stale_fill_before_real_cmp_insert_7_neutral_steps",
        mutation_index=index,
        target_step=steps[index],
        original_index=index,
    )


def hard_RULE_PAD_OPEN_BEFORE_DEP(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Move an actual pad-window step to just before CURE PASSIVATION.
    This keeps it after passivation deposition if possible, but before cure.
    """
    cure_idx = first_index(steps, "CURE PASSIVATION")
    if cure_idx is None or cure_idx < min_mutation_index:
        return None

    passivation_dep_indices = [
        i for i, s in enumerate(steps)
        if s in {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"} and i < cure_idx
    ]
    if not passivation_dep_indices:
        return None

    pad_indices = [
        i for i, s in enumerate(steps)
        if s in PAD_WINDOW_STEPS and i > cure_idx
    ]
    if not pad_indices:
        return None

    from_idx = rng.choice(pad_indices)
    to_idx = cure_idx

    mutated = move_step(steps, from_index=from_idx, to_index=to_idx)

    return MutationResult(
        steps=mutated,
        mutation_type="move_real_pad_window_step_to_before_cure_passivation",
        mutation_index=to_idx,
        target_step=steps[from_idx],
        original_index=from_idx,
    )


def hard_RULE_TEST_BEFORE_PASSIVATION(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Move an actual electrical test to just before CURE PASSIVATION.
    """
    cure_idx = first_index(steps, "CURE PASSIVATION")
    if cure_idx is None or cure_idx < min_mutation_index:
        return None

    test_indices = [
        i for i, s in enumerate(steps)
        if s in ELECTRICAL_TEST_STEPS and i > cure_idx
    ]
    if not test_indices:
        return None

    from_idx = rng.choice(test_indices)
    to_idx = cure_idx

    mutated = move_step(steps, from_index=from_idx, to_index=to_idx)

    return MutationResult(
        steps=mutated,
        mutation_type="move_real_electrical_test_to_before_cure_passivation",
        mutation_index=to_idx,
        target_step=steps[from_idx],
        original_index=from_idx,
    )


def hard_RULE_SHIP_BEFORE_TEST(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Move the actual SHIP LOT step directly before WAFER SORT TEST.
    """
    ship_idx = first_index(steps, "SHIP LOT")
    sort_idx = first_index(steps, "WAFER SORT TEST")

    if ship_idx is None or sort_idx is None:
        return None
    if ship_idx < sort_idx:
        return None
    if sort_idx < min_mutation_index:
        return None

    mutated = move_step(steps, from_index=ship_idx, to_index=sort_idx)

    return MutationResult(
        steps=mutated,
        mutation_type="move_real_ship_lot_to_before_wafer_sort_test",
        mutation_index=sort_idx,
        target_step="SHIP LOT",
        original_index=ship_idx,
    )


def hard_RULE_BACKSIDE_BEFORE_PASSIVATION(
    steps: list[str],
    family: str,
    rng: random.Random,
    min_mutation_index: int,
) -> Optional[MutationResult]:
    """
    Hard mutation:
    Move or insert DEPOSIT BACKSIDE METAL into the passivation region before
    CURE PASSIVATION. Insert BACKSIDE CLEAN immediately before it to avoid
    making RULE_DEP_NO_CLEAN the main issue.
    """
    cure_idx = first_index(steps, "CURE PASSIVATION")
    if cure_idx is None or cure_idx < min_mutation_index:
        return None

    mutated = list(steps)

    existing_idx = first_index(mutated, "DEPOSIT BACKSIDE METAL")
    if existing_idx is not None:
        backside_step = mutated.pop(existing_idx)
        if existing_idx < cure_idx:
            cure_idx -= 1
        original_index = existing_idx
    else:
        backside_step = "DEPOSIT BACKSIDE METAL"
        original_index = -1

    insert_idx = cure_idx
    insertion = ["BACKSIDE CLEAN", backside_step]

    mutated = insert_steps_before(mutated, insert_idx, insertion)

    return MutationResult(
        steps=mutated,
        mutation_type="move_or_insert_backside_metal_with_clean_before_cure_passivation",
        mutation_index=insert_idx + 1,
        target_step=backside_step,
        original_index=original_index,
    )


HARD_MUTATORS: dict[str, Callable[[list[str], str, random.Random, int], Optional[MutationResult]]] = {
    "RULE_DEP_NO_CLEAN": hard_RULE_DEP_NO_CLEAN,
    "RULE_METAL_ETCH_NO_LITHO": hard_RULE_METAL_ETCH_NO_LITHO,
    "RULE_ETCH_NO_MASK": hard_RULE_ETCH_NO_MASK,
    "RULE_LITHO_LEVEL_SKIP": hard_RULE_LITHO_LEVEL_SKIP,
    "RULE_IMPLANT_NO_MASK": hard_RULE_IMPLANT_NO_MASK,
    "RULE_CMP_NO_DEP": hard_RULE_CMP_NO_DEP,
    "RULE_PAD_OPEN_BEFORE_DEP": hard_RULE_PAD_OPEN_BEFORE_DEP,
    "RULE_TEST_BEFORE_PASSIVATION": hard_RULE_TEST_BEFORE_PASSIVATION,
    "RULE_SHIP_BEFORE_TEST": hard_RULE_SHIP_BEFORE_TEST,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": hard_RULE_BACKSIDE_BEFORE_PASSIVATION,
}


# ---------------------------------------------------------------------------
# Loading valid seed data
# ---------------------------------------------------------------------------

def load_valid_records(paths: list[Path]) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []

    for path in paths:
        loaded = read_sequences_from_csv(path)
        records.extend(loaded)

    valid_records: list[SequenceRecord] = []
    invalid_count = 0
    unknown_family_count = 0

    for record in records:
        violations = validate_sequence(record.steps)
        if violations:
            invalid_count += 1
            continue

        family = infer_family(record)
        if family not in FAMILIES:
            unknown_family_count += 1
            continue

        valid_records.append(
            SequenceRecord(
                sequence_id=record.sequence_id,
                family=family,
                steps=record.steps,
                source_file=record.source_file,
            )
        )

    print(f"Loaded {len(records):,} total sequences from valid input.")
    print(f"Kept {len(valid_records):,} validator-confirmed valid sequences.")
    if invalid_count:
        print(f"Skipped {invalid_count:,} invalid input sequences.")
    if unknown_family_count:
        print(f"Skipped {unknown_family_count:,} sequences with unknown family.")

    return valid_records


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_hard_invalid_for_rule_family(
    family: str,
    rule: str,
    seed_records: list[SequenceRecord],
    target: int,
    max_tries: int,
    max_validator_violations: int,
    min_mutation_index: int,
    rng: random.Random,
    global_seen_hashes: set[str],
    seed: int,
) -> tuple[list[InvalidSample], RuleFamilyStats]:
    start = time.time()

    mutator = HARD_MUTATORS[rule]
    accepted: list[InvalidSample] = []

    attempts = 0
    skipped_no_mutation = 0
    skipped_not_intended_rule = 0
    skipped_too_many_violations = 0
    skipped_duplicate = 0
    skipped_too_early = 0

    if not seed_records:
        return accepted, RuleFamilyStats(
            family=family,
            rule=rule,
            target=target,
            accepted=0,
            attempts=0,
            skipped_no_mutation=0,
            skipped_not_intended_rule=0,
            skipped_too_many_violations=0,
            skipped_duplicate=0,
            skipped_too_early=0,
            elapsed_seconds=0.0,
        )

    while len(accepted) < target and attempts < max_tries:
        attempts += 1

        original = rng.choice(seed_records)
        result = mutator(original.steps, family, rng, min_mutation_index=min_mutation_index)

        if result is None:
            skipped_no_mutation += 1
            continue

        if result.mutation_index < min_mutation_index:
            skipped_too_early += 1
            continue

        h = sequence_hash(result.steps)
        if h in global_seen_hashes:
            skipped_duplicate += 1
            continue

        validator_rules, violation_count = validator_rule_set(result.steps)

        if rule not in validator_rules:
            skipped_not_intended_rule += 1
            continue

        if max_validator_violations > 0 and violation_count > max_validator_violations:
            skipped_too_many_violations += 1
            continue

        sequence_id = f"hard_invalid_{family}_{rule}_{len(accepted) + 1:06d}"

        accepted.append(
            InvalidSample(
                sequence_id=sequence_id,
                family=family,
                steps=result.steps,
                violated_rule=rule,
                validator_rules=validator_rules,
                validator_violation_count=violation_count,
                mutation_type=result.mutation_type,
                mutation_index=result.mutation_index,
                target_step=result.target_step,
                original_index=result.original_index,
                original_sequence_id=original.sequence_id,
                seed=seed,
            )
        )

        global_seen_hashes.add(h)

    elapsed = time.time() - start

    stats = RuleFamilyStats(
        family=family,
        rule=rule,
        target=target,
        accepted=len(accepted),
        attempts=attempts,
        skipped_no_mutation=skipped_no_mutation,
        skipped_not_intended_rule=skipped_not_intended_rule,
        skipped_too_many_violations=skipped_too_many_violations,
        skipped_duplicate=skipped_duplicate,
        skipped_too_early=skipped_too_early,
        elapsed_seconds=elapsed,
    )

    return accepted, stats


def generate_hard_invalid_dataset(
    valid_records: list[SequenceRecord],
    families: list[str],
    rules: list[str],
    target_per_rule_family: int,
    max_tries_per_rule_family: int,
    max_validator_violations: int,
    min_mutation_index: int,
    seed: int,
) -> tuple[list[InvalidSample], list[RuleFamilyStats]]:
    rng = random.Random(seed)

    records_by_family: dict[str, list[SequenceRecord]] = defaultdict(list)
    for record in valid_records:
        family = infer_family(record)
        if family in FAMILIES:
            records_by_family[family].append(record)

    print("\nValid seed records by family:")
    for fam in FAMILIES:
        print(f"  {fam}: {len(records_by_family[fam]):,}")

    global_seen_hashes: set[str] = set()
    all_invalid: list[InvalidSample] = []
    all_stats: list[RuleFamilyStats] = []

    for family in families:
        for rule in rules:
            print("\n" + "=" * 100)
            print(f"Generating HARD invalid samples for family={family}, rule={rule}")
            print("=" * 100)

            samples, stats = generate_hard_invalid_for_rule_family(
                family=family,
                rule=rule,
                seed_records=records_by_family[family],
                target=target_per_rule_family,
                max_tries=max_tries_per_rule_family,
                max_validator_violations=max_validator_violations,
                min_mutation_index=min_mutation_index,
                rng=rng,
                global_seen_hashes=global_seen_hashes,
                seed=seed,
            )

            all_invalid.extend(samples)
            all_stats.append(stats)

            print(f"  accepted:                      {stats.accepted:,}/{stats.target:,}")
            print(f"  attempts:                      {stats.attempts:,}")
            print(f"  skipped_no_mutation:           {stats.skipped_no_mutation:,}")
            print(f"  skipped_not_intended_rule:     {stats.skipped_not_intended_rule:,}")
            print(f"  skipped_too_many_violations:   {stats.skipped_too_many_violations:,}")
            print(f"  skipped_duplicate:             {stats.skipped_duplicate:,}")
            print(f"  skipped_too_early:             {stats.skipped_too_early:,}")
            print(f"  elapsed_seconds:               {stats.elapsed_seconds:.2f}")

    return all_invalid, all_stats


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_invalid_long_csv(
    path: Path,
    samples: list[InvalidSample],
    split_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SEQUENCE_ID",
            "FAMILY",
            "STEP_INDEX",
            "STEP",
            "IS_VALID",
            "VIOLATED_RULE",
            "VALIDATOR_RULES",
            "VALIDATOR_VIOLATION_COUNT",
            "MUTATION_TYPE",
            "MUTATION_INDEX",
            "TARGET_STEP",
            "ORIGINAL_INDEX",
            "SPLIT",
            "SOURCE",
            "ORIGINAL_SEQUENCE_ID",
        ])

        for sample in samples:
            split = stable_split(sample.sequence_id, split_seed)
            validator_rules = "|".join(sample.validator_rules)

            for step_index, step in enumerate(sample.steps):
                writer.writerow([
                    sample.sequence_id,
                    sample.family,
                    step_index,
                    step,
                    0,
                    sample.violated_rule,
                    validator_rules,
                    sample.validator_violation_count,
                    sample.mutation_type,
                    sample.mutation_index,
                    sample.target_step,
                    sample.original_index,
                    split,
                    "hard_controlled_invalid_mutation",
                    sample.original_sequence_id,
                ])


def write_invalid_summary_csv(
    path: Path,
    samples: list[InvalidSample],
    split_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SEQUENCE_ID",
            "FAMILY",
            "LENGTH",
            "IS_VALID",
            "VIOLATED_RULE",
            "VALIDATOR_RULES",
            "VALIDATOR_VIOLATION_COUNT",
            "MUTATION_TYPE",
            "MUTATION_INDEX",
            "TARGET_STEP",
            "ORIGINAL_INDEX",
            "SPLIT",
            "ORIGINAL_SEQUENCE_ID",
        ])

        for sample in samples:
            writer.writerow([
                sample.sequence_id,
                sample.family,
                len(sample.steps),
                0,
                sample.violated_rule,
                "|".join(sample.validator_rules),
                sample.validator_violation_count,
                sample.mutation_type,
                sample.mutation_index,
                sample.target_step,
                sample.original_index,
                stable_split(sample.sequence_id, split_seed),
                sample.original_sequence_id,
            ])


def write_mixed_csv(
    path: Path,
    valid_records: list[SequenceRecord],
    invalid_samples: list[InvalidSample],
    split_seed: int,
    max_valid_sequences: Optional[int],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    selected_valid = list(valid_records)

    if max_valid_sequences is not None and len(selected_valid) > max_valid_sequences:
        rng.shuffle(selected_valid)
        selected_valid = selected_valid[:max_valid_sequences]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SEQUENCE_ID",
            "FAMILY",
            "STEP_INDEX",
            "STEP",
            "IS_VALID",
            "VIOLATED_RULE",
            "VALIDATOR_RULES",
            "VALIDATOR_VIOLATION_COUNT",
            "MUTATION_TYPE",
            "MUTATION_INDEX",
            "TARGET_STEP",
            "ORIGINAL_INDEX",
            "SPLIT",
            "SOURCE",
            "ORIGINAL_SEQUENCE_ID",
        ])

        for record in selected_valid:
            family = infer_family(record)
            seq_id = f"valid_{clean_id(record.sequence_id)}"
            split = stable_split(seq_id, split_seed)

            for step_index, step in enumerate(record.steps):
                writer.writerow([
                    seq_id,
                    family,
                    step_index,
                    step,
                    1,
                    "",
                    "",
                    0,
                    "",
                    "",
                    "",
                    "",
                    split,
                    "valid_seed",
                    record.sequence_id,
                ])

        for sample in invalid_samples:
            split = stable_split(sample.sequence_id, split_seed)
            validator_rules = "|".join(sample.validator_rules)

            for step_index, step in enumerate(sample.steps):
                writer.writerow([
                    sample.sequence_id,
                    sample.family,
                    step_index,
                    step,
                    0,
                    sample.violated_rule,
                    validator_rules,
                    sample.validator_violation_count,
                    sample.mutation_type,
                    sample.mutation_index,
                    sample.target_step,
                    sample.original_index,
                    split,
                    "hard_controlled_invalid_mutation",
                    sample.original_sequence_id,
                ])


def write_generation_stats_csv(path: Path, stats: list[RuleFamilyStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "rule",
                "target",
                "accepted",
                "attempts",
                "skipped_no_mutation",
                "skipped_not_intended_rule",
                "skipped_too_many_violations",
                "skipped_duplicate",
                "skipped_too_early",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for item in stats:
            writer.writerow(asdict(item))


def make_invalid_records(samples: list[InvalidSample], source_file: str) -> list[SequenceRecord]:
    return [
        SequenceRecord(
            sequence_id=s.sequence_id,
            family=s.family,
            steps=s.steps,
            source_file=source_file,
        )
        for s in samples
    ]


def write_coverage_reports(
    output_dir: Path,
    valid_records: list[SequenceRecord],
    invalid_samples: list[InvalidSample],
    invalid_csv_path: Path,
    mixed_csv_path: Optional[Path],
    min_count: int,
    top_k: int,
) -> None:
    invalid_records = make_invalid_records(invalid_samples, str(invalid_csv_path))

    invalid_report_dir = output_dir / "coverage_report_hard_invalid_only"
    invalid_coverage = compute_coverage(invalid_records, validate=True)
    invalid_undercovered = build_undercovered_targets(invalid_coverage, min_count=min_count)
    write_outputs(
        output_dir=invalid_report_dir,
        coverage=invalid_coverage,
        undercovered=invalid_undercovered,
        top_k=top_k,
    )

    if mixed_csv_path is not None:
        mixed_records = list(valid_records) + invalid_records
        mixed_report_dir = output_dir / "coverage_report_mixed"
        mixed_coverage = compute_coverage(mixed_records, validate=True)
        mixed_undercovered = build_undercovered_targets(mixed_coverage, min_count=min_count)
        write_outputs(
            output_dir=mixed_report_dir,
            coverage=mixed_coverage,
            undercovered=mixed_undercovered,
            top_k=top_k,
        )


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    valid_records: list[SequenceRecord],
    invalid_samples: list[InvalidSample],
    stats: list[RuleFamilyStats],
    output_files: dict[str, str],
) -> None:
    valid_family_counts = Counter(infer_family(r) for r in valid_records)
    invalid_family_counts = Counter(s.family for s in invalid_samples)
    invalid_rule_counts = Counter(s.violated_rule for s in invalid_samples)
    mutation_counts = Counter(s.mutation_type for s in invalid_samples)

    manifest = {
        "created_at_unix": time.time(),
        "script": "generate_hard_invalid_sequences.py",
        "description": "Hard invalid near-miss generation using realistic local process perturbations.",
        "arguments": vars(args),
        "valid_seed_data": {
            "num_sequences": len(valid_records),
            "num_step_rows": sum(len(r.steps) for r in valid_records),
            "family_counts": dict(sorted(valid_family_counts.items())),
        },
        "hard_invalid_generated_data": {
            "num_sequences": len(invalid_samples),
            "num_step_rows": sum(len(s.steps) for s in invalid_samples),
            "family_counts": dict(sorted(invalid_family_counts.items())),
            "rule_counts": dict(sorted(invalid_rule_counts.items())),
            "mutation_type_counts": dict(sorted(mutation_counts.items())),
        },
        "generation_stats": [asdict(s) for s in stats],
        "output_files": output_files,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_hard_invalid_sequences.py",
        description="Generate hard controlled invalid near-miss semiconductor process sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--valid-input",
        nargs="+",
        required=True,
        help="One or more valid long-format CSV files. Recommended: coverage_guided_v1/coverage_guided_sequences.csv",
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(FAMILIES),
        default=list(FAMILIES),
        help="Families to generate hard invalid samples for.",
    )

    parser.add_argument(
        "--rules",
        nargs="+",
        choices=RULES,
        default=RULES,
        help="Rules to generate hard invalid samples for.",
    )

    parser.add_argument(
        "--target-per-rule-family",
        type=int,
        default=1000,
        help="Number of hard invalid samples per family × rule.",
    )

    parser.add_argument(
        "--max-tries-per-rule-family",
        type=int,
        default=300000,
        help="Maximum mutation attempts for each family × rule pair.",
    )

    parser.add_argument(
        "--max-validator-violations",
        type=int,
        default=4,
        help="Reject samples with more than this many validator violations. Use 0 to disable.",
    )

    parser.add_argument(
        "--min-mutation-index",
        type=int,
        default=20,
        help="Reject mutations before this step index to avoid trivial early invalids.",
    )

    parser.add_argument(
        "--output-dir",
        default="hard_invalid_v1",
        help="Output directory.",
    )

    parser.add_argument(
        "--write-mixed",
        action="store_true",
        help="Also write a combined valid+hard-invalid long-format CSV.",
    )

    parser.add_argument(
        "--max-valid-in-mixed",
        type=int,
        default=None,
        help="Optional cap on valid sequences in mixed CSV.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--coverage-report-min-count",
        type=int,
        default=20,
        help="Minimum count used for undercovered target extraction.",
    )

    parser.add_argument(
        "--coverage-report-top-k",
        type=int,
        default=25,
        help="Number of top/rare items in markdown coverage reports.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_paths = [Path(p) for p in args.valid_input]

    print("Loading and validating seed data...")
    valid_records = load_valid_records(valid_paths)

    if not valid_records:
        raise RuntimeError("No valid records loaded. Check --valid-input paths.")

    expected_total = len(args.families) * len(args.rules) * args.target_per_rule_family

    print("\nHard invalid generation target:")
    print(f"  families:                  {args.families}")
    print(f"  rules:                     {len(args.rules)}")
    print(f"  target per rule/family:    {args.target_per_rule_family:,}")
    print(f"  expected hard invalids:    {expected_total:,}")
    print(f"  min mutation index:        {args.min_mutation_index}")

    invalid_samples, stats = generate_hard_invalid_dataset(
        valid_records=valid_records,
        families=args.families,
        rules=args.rules,
        target_per_rule_family=args.target_per_rule_family,
        max_tries_per_rule_family=args.max_tries_per_rule_family,
        max_validator_violations=args.max_validator_violations,
        min_mutation_index=args.min_mutation_index,
        seed=args.seed,
    )

    invalid_csv_path = output_dir / "hard_invalid_sequences.csv"
    invalid_summary_path = output_dir / "hard_invalid_sequence_summary.csv"
    stats_path = output_dir / "hard_invalid_generation_stats.csv"
    manifest_path = output_dir / "hard_invalid_manifest.json"
    mixed_csv_path: Optional[Path] = None

    print("\nWriting hard invalid dataset files...")

    write_invalid_long_csv(
        path=invalid_csv_path,
        samples=invalid_samples,
        split_seed=args.seed,
    )

    write_invalid_summary_csv(
        path=invalid_summary_path,
        samples=invalid_samples,
        split_seed=args.seed,
    )

    write_generation_stats_csv(
        path=stats_path,
        stats=stats,
    )

    if args.write_mixed:
        mixed_csv_path = output_dir / "mixed_valid_hard_invalid_sequences.csv"
        print("Writing mixed valid+hard-invalid dataset...")
        write_mixed_csv(
            path=mixed_csv_path,
            valid_records=valid_records,
            invalid_samples=invalid_samples,
            split_seed=args.seed,
            max_valid_sequences=args.max_valid_in_mixed,
            seed=args.seed,
        )

    print("Writing coverage reports...")
    write_coverage_reports(
        output_dir=output_dir,
        valid_records=valid_records,
        invalid_samples=invalid_samples,
        invalid_csv_path=invalid_csv_path,
        mixed_csv_path=mixed_csv_path,
        min_count=args.coverage_report_min_count,
        top_k=args.coverage_report_top_k,
    )

    output_files = {
        "hard_invalid_sequences_csv": str(invalid_csv_path),
        "hard_invalid_sequence_summary_csv": str(invalid_summary_path),
        "hard_invalid_generation_stats_csv": str(stats_path),
        "hard_invalid_manifest_json": str(manifest_path),
        "coverage_report_hard_invalid_only_md": str(
            output_dir / "coverage_report_hard_invalid_only" / "coverage_report.md"
        ),
    }

    if mixed_csv_path is not None:
        output_files["mixed_valid_hard_invalid_sequences_csv"] = str(mixed_csv_path)
        output_files["coverage_report_mixed_md"] = str(
            output_dir / "coverage_report_mixed" / "coverage_report.md"
        )

    write_manifest(
        path=manifest_path,
        args=args,
        valid_records=valid_records,
        invalid_samples=invalid_samples,
        stats=stats,
        output_files=output_files,
    )

    rule_counts = Counter(s.violated_rule for s in invalid_samples)
    family_counts = Counter(s.family for s in invalid_samples)
    mutation_counts = Counter(s.mutation_type for s in invalid_samples)

    print("\nDone.")
    print("\nGenerated files:")
    print(f"  {invalid_csv_path}")
    print(f"  {invalid_summary_path}")
    print(f"  {stats_path}")
    print(f"  {manifest_path}")
    print(f"  {output_dir / 'coverage_report_hard_invalid_only' / 'coverage_report.md'}")

    if mixed_csv_path is not None:
        print(f"  {mixed_csv_path}")
        print(f"  {output_dir / 'coverage_report_mixed' / 'coverage_report.md'}")

    print("\nFinal hard invalid dataset summary:")
    print(f"  hard invalid sequences:       {len(invalid_samples):,}")
    print(f"  hard invalid step rows:       {sum(len(s.steps) for s in invalid_samples):,}")
    print(f"  family counts:                {dict(sorted(family_counts.items()))}")
    print("  rule counts:")
    for rule in RULES:
        print(f"    {rule}: {rule_counts.get(rule, 0):,}")

    print("\nMutation type counts:")
    for mutation_type, count in sorted(mutation_counts.items()):
        print(f"  {mutation_type}: {count:,}")

    incomplete = [s for s in stats if s.accepted < s.target]

    if incomplete:
        print("\nWarning: some family × rule targets were not fully reached:")
        for s in incomplete:
            print(
                f"  {s.family} / {s.rule}: "
                f"{s.accepted:,}/{s.target:,} accepted after {s.attempts:,} attempts"
            )
        print("\nPossible fixes:")
        print("  - Increase --max-tries-per-rule-family")
        print("  - Lower --min-mutation-index")
        print("  - Increase --max-validator-violations")
        print("  - Use more valid seed data")
    else:
        print("\nAll family × rule targets reached.")

    print("\nNext step:")
    print("  Inspect hard_invalid_generation_stats.csv and coverage_report_mixed/coverage_report.md.")
    print("  If balanced, we move to build_task_datasets.py.")


if __name__ == "__main__":
    main()