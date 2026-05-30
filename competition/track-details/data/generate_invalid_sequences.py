#!/usr/bin/env python3
"""
generate_invalid_sequences.py — Controlled invalid near-miss generation.

Place this file in:
    competition/track-details/data/generate_invalid_sequences.py

Expected project structure:
    competition/track-details/
    ├── data/
    │   ├── coverage_tracker.py
    │   ├── generate_coverage_guided.py
    │   └── generate_invalid_sequences.py
    └── training_data/
        ├── generate_sequences.py
        ├── MOSFET_variants.csv
        ├── IGBT_variants.csv
        └── IC_variants.csv

Purpose
-------
Generate labeled invalid semiconductor process sequences from valid seeds.

The mutations are controlled near-misses:
- one intended process-rule violation per generated sample,
- validator-confirmed,
- labeled with violated rule,
- suitable for anomaly detection and rule-attribution training.

Recommended use after coverage-guided valid generation:
    python generate_invalid_sequences.py \
      --valid-input coverage_guided_v1/coverage_guided_sequences.csv \
      --target-per-rule-family 1000 \
      --output-dir invalid_v1 \
      --write-mixed \
      --seed 42

This creates:
    10 rules × 3 families × 1000 = 30,000 invalid sequences

For a larger dataset:
    --target-per-rule-family 3000

This creates:
    10 rules × 3 families × 3000 = 90,000 invalid sequences
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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

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
        CLEAN_STEPS,
        validate_sequence,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import validate_sequence and rule constants from "
        f"{TRAINING_DATA_DIR / 'generate_sequences.py'}.\n"
        "Make sure generate_sequences.py is located in competition/track-details/training_data/."
    ) from exc

try:
    from coverage_tracker import (  # type: ignore
        SequenceRecord,
        build_undercovered_targets,
        compute_coverage,
        read_sequences_from_csv,
        write_outputs,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import utilities from coverage_tracker.py.\n"
        "Make sure coverage_tracker.py is located in competition/track-details/data/."
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MutationResult:
    steps: list[str]
    mutation_type: str
    mutation_index: int


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
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# General utilities
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
    """
    Deterministic 80/10/10 split.
    """
    h = int(hashlib.md5(f"{sequence_id}|{seed}".encode()).hexdigest(), 16) % 100
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


def insert_step(steps: list[str], index: int, step: str) -> list[str]:
    new_steps = list(steps)
    index = max(0, min(index, len(new_steps)))
    new_steps.insert(index, step)
    return new_steps


def move_step(steps: list[str], from_index: int, to_index: int) -> list[str]:
    """
    Move one step from from_index to to_index.
    """
    new_steps = list(steps)
    if from_index < 0 or from_index >= len(new_steps):
        return new_steps

    step = new_steps.pop(from_index)

    if to_index > from_index:
        to_index -= 1

    to_index = max(0, min(to_index, len(new_steps)))
    new_steps.insert(to_index, step)
    return new_steps


def replace_nearby_expose_level(
    steps: list[str],
    align_index: int,
    old_level: int,
    new_level: int,
) -> list[str]:
    """
    If we mutate ALIGN MASK LEVEL k to k', also mutate the corresponding nearby
    EXPOSE LITHO LEVEL k to k' before the next ALIGN MASK step.
    """
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


def find_indices(steps: list[str], predicate: Callable[[str], bool]) -> list[int]:
    return [i for i, step in enumerate(steps) if predicate(step)]


def first_index(steps: list[str], target: str) -> int | None:
    for i, step in enumerate(steps):
        if step == target:
            return i
    return None


def any_previous_within(
    steps: list[str],
    index: int,
    targets: set[str] | frozenset[str],
    window: int,
) -> bool:
    lo = max(0, index - window)
    return any(steps[j] in targets for j in range(lo, index))


# ---------------------------------------------------------------------------
# Controlled invalid mutations
# ---------------------------------------------------------------------------


def mutate_RULE_DEP_NO_CLEAN(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult:
    """
    Insert a deposition step early in the sequence, before any clean step exists.
    This should trigger RULE_DEP_NO_CLEAN.
    """
    index = min(3, len(steps))
    dep_step = {
        "mosfet": "DEPOSIT POLYSILICON",
        "igbt": "DEPOSIT FIELD OXIDE",
        "ic": "DEPOSIT PAD OXIDE",
    }.get(family, "DEPOSIT POLYSILICON")

    return MutationResult(
        steps=insert_step(steps, index, dep_step),
        mutation_type=f"insert_early_deposition_without_prior_clean:{dep_step}",
        mutation_index=index,
    )


def mutate_RULE_METAL_ETCH_NO_LITHO(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult:
    """
    Insert a metal etch early, before lithography/expose/develop.
    This usually also triggers RULE_ETCH_NO_MASK, which is acceptable because
    metal etch is a special case of patterned etch.
    """
    index = min(3, len(steps))
    etch_step = "METAL ETCH DRY" if family in {"igbt", "ic"} else "METAL ETCH"

    return MutationResult(
        steps=insert_step(steps, index, etch_step),
        mutation_type=f"insert_early_metal_etch_without_litho:{etch_step}",
        mutation_index=index,
    )


def mutate_RULE_ETCH_NO_MASK(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult:
    """
    Insert a patterned non-metal etch early, before any develop step.
    """
    index = min(3, len(steps))
    etch_step = {
        "mosfet": "OXIDE ETCH",
        "igbt": "OXIDE ETCH DRY",
        "ic": "OXIDE ETCH DRY",
    }.get(family, "OXIDE ETCH")

    return MutationResult(
        steps=insert_step(steps, index, etch_step),
        mutation_type=f"insert_early_etch_without_mask:{etch_step}",
        mutation_index=index,
    )


def mutate_RULE_LITHO_LEVEL_SKIP(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult | None:
    """
    Change a later ALIGN MASK LEVEL so that the sequence skips a level.
    Example:
        LEVEL 1, LEVEL 2, LEVEL 3
    becomes:
        LEVEL 1, LEVEL 3, LEVEL 3
    """
    align_pattern = "ALIGN MASK LEVEL "

    aligns: list[tuple[int, int]] = []
    for i, step in enumerate(steps):
        if step.startswith(align_pattern):
            suffix = step.split(align_pattern, 1)[1]
            if suffix.isdigit():
                aligns.append((i, int(suffix)))

    if len(aligns) < 2:
        return None

    pos = rng.randrange(1, len(aligns))
    prev_level = aligns[pos - 1][1]
    align_index, old_level = aligns[pos]

    new_level = prev_level + 2
    if new_level == old_level:
        new_level += 1

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
        mutation_type=f"change_litho_level_from_{old_level}_to_{new_level}",
        mutation_index=align_index,
    )


def mutate_RULE_IMPLANT_NO_MASK(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult:
    """
    Insert an implant early, before lithography/develop/window opening.
    """
    index = min(3, len(steps))
    implant_step = {
        "mosfet": "IMPLANT WELL",
        "igbt": "IMPLANT P BODY",
        "ic": "IMPLANT N-TYPE",
    }.get(family, "IMPLANT WELL")

    return MutationResult(
        steps=insert_step(steps, index, implant_step),
        mutation_type=f"insert_early_implant_without_mask:{implant_step}",
        mutation_index=index,
    )


def mutate_RULE_CMP_NO_DEP(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult:
    """
    Insert CMP early, before any deposition/fill step exists.
    """
    index = min(3, len(steps))
    cmp_step = rng.choice(
        [
            "CMP DIELECTRIC",
            "CMP INTERLAYER DIELECTRIC",
            "CMP METAL",
            "CMP VIA FILL",
        ]
    )

    return MutationResult(
        steps=insert_step(steps, index, cmp_step),
        mutation_type=f"insert_early_cmp_without_fill:{cmp_step}",
        mutation_index=index,
    )


def mutate_RULE_PAD_OPEN_BEFORE_DEP(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult:
    """
    Insert pad-opening operation before passivation deposition/cure.
    """
    index = min(3, len(steps))
    pad_step = rng.choice(
        [
            "OPEN PAD WINDOW",
            "OPEN BOND PAD WINDOW",
            "PAD WINDOW LITHO",
            "OPEN PAD WINDOW LITHO",
        ]
    )

    return MutationResult(
        steps=insert_step(steps, index, pad_step),
        mutation_type=f"insert_early_pad_window_before_passivation:{pad_step}",
        mutation_index=index,
    )


def mutate_RULE_TEST_BEFORE_PASSIVATION(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult:
    """
    Insert electrical test before CURE PASSIVATION.
    """
    index = min(3, len(steps))
    test_step = {
        "mosfet": "THRESHOLD VOLTAGE TEST",
        "igbt": "BREAKDOWN VOLTAGE TEST",
        "ic": "PARAMETRIC TEST",
    }.get(family, "PARAMETRIC TEST")

    return MutationResult(
        steps=insert_step(steps, index, test_step),
        mutation_type=f"insert_early_electrical_test_before_passivation:{test_step}",
        mutation_index=index,
    )


def mutate_RULE_SHIP_BEFORE_TEST(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult | None:
    """
    Move SHIP LOT before WAFER SORT TEST.
    """
    ship_idx = first_index(steps, "SHIP LOT")
    sort_idx = first_index(steps, "WAFER SORT TEST")

    if ship_idx is None or sort_idx is None:
        return None

    if ship_idx < sort_idx:
        return None

    mutated = move_step(steps, from_index=ship_idx, to_index=sort_idx)

    return MutationResult(
        steps=mutated,
        mutation_type="move_ship_lot_before_wafer_sort_test",
        mutation_index=sort_idx,
    )


def mutate_RULE_BACKSIDE_BEFORE_PASSIVATION(
    steps: list[str],
    family: str,
    rng: random.Random,
) -> MutationResult | None:
    """
    Put DEPOSIT BACKSIDE METAL before CURE PASSIVATION while trying to keep
    a clean step immediately before it, so the intended rule is mainly
    RULE_BACKSIDE_BEFORE_PASSIVATION.
    """
    cure_idx = first_index(steps, "CURE PASSIVATION")
    if cure_idx is None:
        return None

    mutated = list(steps)

    # If a backside metal step already exists, remove it first and reinsert it.
    existing_idx = first_index(mutated, "DEPOSIT BACKSIDE METAL")
    if existing_idx is not None:
        backside_step = mutated.pop(existing_idx)
        if existing_idx < cure_idx:
            cure_idx -= 1
    else:
        backside_step = "DEPOSIT BACKSIDE METAL"

    # Prefer insertion directly after a clean step before cure, to avoid
    # accidentally creating RULE_DEP_NO_CLEAN as the primary violation.
    clean_candidates = [i for i, step in enumerate(mutated[:cure_idx]) if step in CLEAN_STEPS]

    if clean_candidates:
        insert_idx = clean_candidates[-1] + 1
    else:
        insert_idx = min(3, len(mutated))

    mutated.insert(insert_idx, backside_step)

    return MutationResult(
        steps=mutated,
        mutation_type="move_or_insert_backside_metal_before_cure_passivation",
        mutation_index=insert_idx,
    )


MUTATORS: dict[str, Callable[[list[str], str, random.Random], MutationResult | None]] = {
    "RULE_DEP_NO_CLEAN": mutate_RULE_DEP_NO_CLEAN,
    "RULE_METAL_ETCH_NO_LITHO": mutate_RULE_METAL_ETCH_NO_LITHO,
    "RULE_ETCH_NO_MASK": mutate_RULE_ETCH_NO_MASK,
    "RULE_LITHO_LEVEL_SKIP": mutate_RULE_LITHO_LEVEL_SKIP,
    "RULE_IMPLANT_NO_MASK": mutate_RULE_IMPLANT_NO_MASK,
    "RULE_CMP_NO_DEP": mutate_RULE_CMP_NO_DEP,
    "RULE_PAD_OPEN_BEFORE_DEP": mutate_RULE_PAD_OPEN_BEFORE_DEP,
    "RULE_TEST_BEFORE_PASSIVATION": mutate_RULE_TEST_BEFORE_PASSIVATION,
    "RULE_SHIP_BEFORE_TEST": mutate_RULE_SHIP_BEFORE_TEST,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": mutate_RULE_BACKSIDE_BEFORE_PASSIVATION,
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

    for record in records:
        violations = validate_sequence(record.steps)
        if violations:
            invalid_count += 1
            continue

        family = infer_family(record)
        if family not in FAMILIES:
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

    return valid_records


# ---------------------------------------------------------------------------
# Invalid generation
# ---------------------------------------------------------------------------


def generate_invalid_for_rule_family(
    family: str,
    rule: str,
    seed_records: list[SequenceRecord],
    target: int,
    max_tries: int,
    max_validator_violations: int,
    rng: random.Random,
    global_seen_hashes: set[str],
    seed: int,
) -> tuple[list[InvalidSample], RuleFamilyStats]:
    start = time.time()

    mutator = MUTATORS[rule]
    accepted: list[InvalidSample] = []

    attempts = 0
    skipped_no_mutation = 0
    skipped_not_intended_rule = 0
    skipped_too_many_violations = 0
    skipped_duplicate = 0

    if not seed_records:
        stats = RuleFamilyStats(
            family=family,
            rule=rule,
            target=target,
            accepted=0,
            attempts=0,
            skipped_no_mutation=0,
            skipped_not_intended_rule=0,
            skipped_too_many_violations=0,
            skipped_duplicate=0,
            elapsed_seconds=0.0,
        )
        return accepted, stats

    while len(accepted) < target and attempts < max_tries:
        attempts += 1

        original = rng.choice(seed_records)
        result = mutator(original.steps, family, rng)

        if result is None:
            skipped_no_mutation += 1
            continue

        h = sequence_hash(result.steps)
        if h in global_seen_hashes:
            skipped_duplicate += 1
            continue

        violations = validate_sequence(result.steps)
        validator_rules = sorted({v.rule for v in violations})

        if rule not in validator_rules:
            skipped_not_intended_rule += 1
            continue

        if max_validator_violations > 0 and len(violations) > max_validator_violations:
            skipped_too_many_violations += 1
            continue

        sequence_id = f"invalid_{family}_{rule}_{len(accepted) + 1:06d}"

        accepted.append(
            InvalidSample(
                sequence_id=sequence_id,
                family=family,
                steps=result.steps,
                violated_rule=rule,
                validator_rules=validator_rules,
                validator_violation_count=len(violations),
                mutation_type=result.mutation_type,
                mutation_index=result.mutation_index,
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
        elapsed_seconds=elapsed,
    )

    return accepted, stats


def generate_invalid_dataset(
    valid_records: list[SequenceRecord],
    families: list[str],
    rules: list[str],
    target_per_rule_family: int,
    max_tries_per_rule_family: int,
    max_validator_violations: int,
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
            print(f"Generating invalid samples for family={family}, rule={rule}")
            print("=" * 100)

            samples, stats = generate_invalid_for_rule_family(
                family=family,
                rule=rule,
                seed_records=records_by_family[family],
                target=target_per_rule_family,
                max_tries=max_tries_per_rule_family,
                max_validator_violations=max_validator_violations,
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
        writer.writerow(
            [
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
                "SPLIT",
                "SOURCE",
                "ORIGINAL_SEQUENCE_ID",
            ]
        )

        for sample in samples:
            split = stable_split(sample.sequence_id, split_seed)
            validator_rules = "|".join(sample.validator_rules)

            for step_index, step in enumerate(sample.steps):
                writer.writerow(
                    [
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
                        split,
                        "controlled_invalid_mutation",
                        sample.original_sequence_id,
                    ]
                )


def write_invalid_summary_csv(
    path: Path,
    samples: list[InvalidSample],
    split_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "SEQUENCE_ID",
                "FAMILY",
                "LENGTH",
                "IS_VALID",
                "VIOLATED_RULE",
                "VALIDATOR_RULES",
                "VALIDATOR_VIOLATION_COUNT",
                "MUTATION_TYPE",
                "MUTATION_INDEX",
                "SPLIT",
                "ORIGINAL_SEQUENCE_ID",
            ]
        )

        for sample in samples:
            writer.writerow(
                [
                    sample.sequence_id,
                    sample.family,
                    len(sample.steps),
                    0,
                    sample.violated_rule,
                    "|".join(sample.validator_rules),
                    sample.validator_violation_count,
                    sample.mutation_type,
                    sample.mutation_index,
                    stable_split(sample.sequence_id, split_seed),
                    sample.original_sequence_id,
                ]
            )


def write_mixed_csv(
    path: Path,
    valid_records: list[SequenceRecord],
    invalid_samples: list[InvalidSample],
    split_seed: int,
    max_valid_sequences: int | None,
    seed: int,
) -> None:
    """
    Write combined valid+invalid long-format CSV.

    If max_valid_sequences is provided, valid sequences are downsampled.
    This is useful if invalid data is smaller than valid data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    selected_valid = list(valid_records)
    if max_valid_sequences is not None and len(selected_valid) > max_valid_sequences:
        rng.shuffle(selected_valid)
        selected_valid = selected_valid[:max_valid_sequences]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
                "SPLIT",
                "SOURCE",
                "ORIGINAL_SEQUENCE_ID",
            ]
        )

        for record in selected_valid:
            family = infer_family(record)
            seq_id = f"valid_{clean_id(record.sequence_id)}"
            split = stable_split(seq_id, split_seed)

            for step_index, step in enumerate(record.steps):
                writer.writerow(
                    [
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
                        split,
                        "valid_seed",
                        record.sequence_id,
                    ]
                )

        for sample in invalid_samples:
            split = stable_split(sample.sequence_id, split_seed)
            validator_rules = "|".join(sample.validator_rules)

            for step_index, step in enumerate(sample.steps):
                writer.writerow(
                    [
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
                        split,
                        "controlled_invalid_mutation",
                        sample.original_sequence_id,
                    ]
                )


def write_generation_stats_csv(
    path: Path,
    stats: list[RuleFamilyStats],
) -> None:
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
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for item in stats:
            writer.writerow(asdict(item))


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

    manifest = {
        "created_at_unix": time.time(),
        "script": "generate_invalid_sequences.py",
        "arguments": vars(args),
        "valid_seed_data": {
            "num_sequences": len(valid_records),
            "num_step_rows": sum(len(r.steps) for r in valid_records),
            "family_counts": dict(sorted(valid_family_counts.items())),
        },
        "invalid_generated_data": {
            "num_sequences": len(invalid_samples),
            "num_step_rows": sum(len(s.steps) for s in invalid_samples),
            "family_counts": dict(sorted(invalid_family_counts.items())),
            "rule_counts": dict(sorted(invalid_rule_counts.items())),
        },
        "generation_stats": [asdict(s) for s in stats],
        "output_files": output_files,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Coverage reports
# ---------------------------------------------------------------------------


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
    mixed_csv_path: Path | None,
    min_count: int,
    top_k: int,
) -> None:
    invalid_records = make_invalid_records(invalid_samples, str(invalid_csv_path))

    invalid_report_dir = output_dir / "coverage_report_invalid_only"
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_invalid_sequences.py",
        description="Generate controlled invalid near-miss semiconductor process sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--valid-input",
        nargs="+",
        required=True,
        help=(
            "One or more valid long-format CSV files. "
            "Recommended: coverage_guided_v1/coverage_guided_sequences.csv"
        ),
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(FAMILIES),
        default=list(FAMILIES),
        help="Families to generate invalid samples for.",
    )

    parser.add_argument(
        "--rules",
        nargs="+",
        choices=RULES,
        default=RULES,
        help="Rules to generate invalid samples for.",
    )

    parser.add_argument(
        "--target-per-rule-family",
        type=int,
        default=1000,
        help=(
            "Number of invalid samples to generate for each family × rule pair. "
            "Total target = families × rules × this value."
        ),
    )

    parser.add_argument(
        "--max-tries-per-rule-family",
        type=int,
        default=200000,
        help="Maximum mutation attempts for each family × rule pair.",
    )

    parser.add_argument(
        "--max-validator-violations",
        type=int,
        default=4,
        help=(
            "Reject samples with more than this many validator violations. "
            "Use 0 to disable this filter. Recommended: 3 or 4."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="invalid_v1",
        help="Output directory.",
    )

    parser.add_argument(
        "--write-mixed",
        action="store_true",
        help="Also write a combined valid+invalid long-format CSV.",
    )

    parser.add_argument(
        "--max-valid-in-mixed",
        type=int,
        default=None,
        help=(
            "Optional cap on number of valid sequences included in mixed CSV. "
            "Useful for balancing. Default keeps all valid input sequences."
        ),
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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_paths = [Path(p) for p in args.valid_input]

    print("Loading and validating seed data...")
    valid_records = load_valid_records(valid_paths)

    if not valid_records:
        raise RuntimeError("No valid records loaded. Check --valid-input paths.")

    expected_total = len(args.families) * len(args.rules) * args.target_per_rule_family
    print("\nGeneration target:")
    print(f"  families:                  {args.families}")
    print(f"  rules:                     {len(args.rules)}")
    print(f"  target per rule/family:    {args.target_per_rule_family:,}")
    print(f"  expected invalid total:    {expected_total:,}")

    invalid_samples, stats = generate_invalid_dataset(
        valid_records=valid_records,
        families=args.families,
        rules=args.rules,
        target_per_rule_family=args.target_per_rule_family,
        max_tries_per_rule_family=args.max_tries_per_rule_family,
        max_validator_violations=args.max_validator_violations,
        seed=args.seed,
    )

    invalid_csv_path = output_dir / "invalid_sequences.csv"
    invalid_summary_path = output_dir / "invalid_sequence_summary.csv"
    stats_path = output_dir / "invalid_generation_stats.csv"
    manifest_path = output_dir / "invalid_manifest.json"
    mixed_csv_path: Path | None = None

    print("\nWriting invalid dataset files...")

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
        mixed_csv_path = output_dir / "mixed_valid_invalid_sequences.csv"
        print("Writing mixed valid+invalid dataset...")
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
        "invalid_sequences_csv": str(invalid_csv_path),
        "invalid_sequence_summary_csv": str(invalid_summary_path),
        "invalid_generation_stats_csv": str(stats_path),
        "invalid_manifest_json": str(manifest_path),
        "coverage_report_invalid_only_md": str(
            output_dir / "coverage_report_invalid_only" / "coverage_report.md"
        ),
    }

    if mixed_csv_path is not None:
        output_files["mixed_valid_invalid_sequences_csv"] = str(mixed_csv_path)
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

    print("\nDone.")
    print("\nGenerated files:")
    print(f"  {invalid_csv_path}")
    print(f"  {invalid_summary_path}")
    print(f"  {stats_path}")
    print(f"  {manifest_path}")
    print(f"  {output_dir / 'coverage_report_invalid_only' / 'coverage_report.md'}")

    if mixed_csv_path is not None:
        print(f"  {mixed_csv_path}")
        print(f"  {output_dir / 'coverage_report_mixed' / 'coverage_report.md'}")

    print("\nFinal invalid dataset summary:")
    print(f"  invalid sequences:       {len(invalid_samples):,}")
    print(f"  invalid step rows:       {sum(len(s.steps) for s in invalid_samples):,}")
    print(f"  family counts:           {dict(sorted(family_counts.items()))}")
    print("  rule counts:")
    for rule in RULES:
        print(f"    {rule}: {rule_counts.get(rule, 0):,}")

    incomplete = [s for s in stats if s.accepted < s.target]

    if incomplete:
        print("\nWarning: some family × rule targets were not fully reached:")
        for s in incomplete:
            print(
                f"  {s.family} / {s.rule}: "
                f"{s.accepted:,}/{s.target:,} accepted after {s.attempts:,} attempts"
            )
        print("\nYou can increase --max-tries-per-rule-family if needed.")

    print("\nNext step:")
    print("  Inspect invalid_generation_stats.csv and coverage_report_mixed/coverage_report.md.")
    print(
        "  Then build task-specific training files for next-step prediction, completion, and anomaly detection."
    )


if __name__ == "__main__":
    main()
