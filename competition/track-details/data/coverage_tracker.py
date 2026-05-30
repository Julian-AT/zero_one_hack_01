#!/usr/bin/env python3
"""
coverage_tracker.py — Coverage reporting for semiconductor process sequences.

Place this file in:
    competition/track-details/data/coverage_tracker.py

Purpose
-------
Reads generated semiconductor process-sequence CSV files and computes coverage
statistics needed before implementing coverage-guided data generation.

The script supports the current CSV format produced by generate_sequences.py:

    SEQUENCE_ID,STEP
    seq_0001,RECEIVE WAFER LOT
    seq_0001,LOT IDENTIFICATION
    ...

It also supports optional FAMILY and STEP_INDEX columns if they are added later.

Outputs
-------
By default, writes:

    coverage_outputs/coverage_report.json
    coverage_outputs/coverage_report.md
    coverage_outputs/undercovered_targets.csv
    coverage_outputs/step_counts.csv
    coverage_outputs/transition_counts.csv
    coverage_outputs/trigram_counts.csv
    coverage_outputs/block_transition_counts.csv
    coverage_outputs/rule_boundary_counts.csv

Examples
--------
From competition/track-details/data/:

    python coverage_tracker.py \
      --input MOSFET_variants.csv \
      --family mosfet \
      --output-dir coverage_outputs

Multiple files, family inferred from filename if possible:

    python coverage_tracker.py \
      --input MOSFET_variants.csv IGBT_variants.csv IC_variants.csv \
      --output-dir coverage_outputs

Validate while computing coverage, if generate_sequences.py is available:

    python coverage_tracker.py \
      --input MOSFET_variants.csv IGBT_variants.csv IC_variants.csv \
      --output-dir coverage_outputs \
      --validate
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional integration with generate_sequences.py
# ---------------------------------------------------------------------------

try:
    from generate_sequences import (  # type: ignore
        BACKSIDE_METAL_STEPS,
        CLEAN_STEPS,
        CMP_STEPS,
        DEPOSITION_STEPS,
        ELECTRICAL_TEST_STEPS,
        ETCH_STEPS,
        FILL_STEPS,
        IMPLANT_OPENER_STEPS,
        IMPLANT_STEPS,
        METAL_ETCH_STEPS,
        PAD_WINDOW_STEPS,
        validate_sequence,
    )

    GENERATOR_IMPORT_OK = True

except Exception:
    GENERATOR_IMPORT_OK = False
    validate_sequence = None  # type: ignore

    # Fallback sets. These are intentionally small. If generate_sequences.py is
    # available in the same folder, the script will use the authoritative sets.
    DEPOSITION_STEPS = frozenset(
        {
            "THERMAL OXIDATION",
            "GATE OXIDE GROWTH",
            "DEPOSIT PAD OXIDE",
            "EPITAXIAL DEPOSITION",
            "DEPOSIT POLYSILICON",
            "DEPOSIT SPACER DIELECTRIC",
            "DEPOSIT FIELD OXIDE",
            "DEPOSIT GATE OXIDE OR DIELECTRIC",
            "DEPOSIT INTERLAYER DIELECTRIC",
            "DEPOSIT INTERLEVEL DIELECTRIC",
            "DEPOSIT BARRIER METAL",
            "DEPOSIT METAL SEED",
            "DEPOSIT METAL 1",
            "DEPOSIT TOP METAL",
            "DEPOSIT BACKSIDE METAL",
            "DEPOSIT TUNGSTEN SEED",
            "DEPOSIT PASSIVATION",
            "DEPOSIT PASSIVATION LAYER",
            "DEPOSIT BACKSIDE PROTECTION",
        }
    )

    CLEAN_STEPS = frozenset(
        {
            "PRE CLEAN WAFER",
            "WAFER CLEAN PRE PROCESS",
            "WAFER SURFACE CLEAN",
            "RCA CLEAN 1",
            "RCA CLEAN 2",
            "WET CLEAN RCA1",
            "WET CLEAN RCA2",
            "HF DIP",
            "OXIDE STRIP",
            "SURFACE PREP FOR DEPOSITION",
            "FRONTSIDE CLEAN",
            "BACKSIDE CLEAN",
            "FRONTSIDE CLEAN FINAL",
            "BACKSIDE CLEAN FINAL",
            "WAFER CLEAN PRE-GRIND",
            "DRY WAFER",
            "DRY WAFER BACKSIDE",
            "CLEAN AFTER ETCH",
            "CLEAN AFTER OXIDE ETCH",
            "CLEAN AFTER POLY ETCH",
            "CLEAN AFTER VIA ETCH",
            "CLEAN AFTER METAL ETCH",
            "CLEAN AFTER WINDOW ETCH",
            "CLEAN AFTER FIELD ETCH",
            "CLEAN PAD OPENING",
            "BACKSIDE ETCH CLEAN",
            "BACKSIDE RINSE",
            "THERMAL OXIDATION",
            "GATE OXIDE PREP",
            "RAPID THERMAL ANNEAL",
            "EPITAXY ANNEAL",
            "ANNEAL OXIDE",
        }
    )

    ETCH_STEPS = frozenset(
        {
            "OXIDE ETCH",
            "OXIDE ETCH DRY",
            "POLYSILICON ETCH",
            "POLYSILICON ETCH DRY",
            "ETCH SILICON OR OXIDE WINDOW",
            "FIELD OXIDE ETCH",
            "VIA ETCH",
            "VIA ETCH THROUGH DIELECTRIC",
            "DIELECTRIC ETCH VIA",
            "METAL ETCH",
            "METAL ETCH DRY",
            "PASSIVATION ETCH PAD OPENING",
            "PASSIVATION ETCH",
        }
    )

    METAL_ETCH_STEPS = frozenset({"METAL ETCH", "METAL ETCH DRY"})

    IMPLANT_STEPS = frozenset(
        {
            "IMPLANT WELL",
            "IMPLANT SOURCE DRAIN",
            "IMPLANT SOURCE REGION",
            "IMPLANT LDD",
            "IMPLANT P BODY",
            "IMPLANT N BUFFER",
            "IMPLANT CHANNEL STOP",
            "IMPLANT DRAIN / CATHODE REGION",
            "IMPLANT N-TYPE",
        }
    )

    IMPLANT_OPENER_STEPS = frozenset(
        {
            "OXIDE ETCH",
            "OXIDE ETCH DRY",
            "ETCH SILICON OR OXIDE WINDOW",
            "DEVELOP PHOTORESIST",
        }
    )

    CMP_STEPS = frozenset(
        {
            "CMP DIELECTRIC",
            "CMP INTERLAYER DIELECTRIC",
            "CMP METAL",
            "CMP VIA FILL",
        }
    )

    FILL_STEPS = (
        frozenset(
            {
                "FILL VIA METAL",
                "FILL VIA TUNGSTEN",
            }
        )
        | DEPOSITION_STEPS
    )

    PAD_WINDOW_STEPS = frozenset(
        {
            "OPEN PAD WINDOW",
            "OPEN BOND PAD WINDOW",
            "PAD WINDOW LITHO",
            "OPEN PAD WINDOW LITHO",
        }
    )

    ELECTRICAL_TEST_STEPS = frozenset(
        {
            "PARAMETRIC TEST",
            "ELECTRICAL PARAMETRIC TEST",
            "THRESHOLD VOLTAGE TEST",
            "BREAKDOWN VOLTAGE TEST",
            "LEAKAGE TEST",
            "SWITCHING TEST",
        }
    )

    BACKSIDE_METAL_STEPS = frozenset({"DEPOSIT BACKSIDE METAL"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    family: str
    steps: list[str]
    source_file: str


@dataclass
class CoverageSummary:
    num_sequences: int
    num_step_rows: int
    num_families: int
    families: dict[str, int]
    min_length: int
    mean_length: float
    max_length: int
    unique_steps: int
    unique_transitions: int
    unique_trigrams: int
    unique_blocks: int
    unique_block_transitions: int
    validation_enabled: bool
    validator_import_ok: bool
    valid_sequences: int | None
    invalid_sequences: int | None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAMILIES = ("mosfet", "igbt", "ic")

LENGTH_BINS = (
    ("short_0_100", 0, 100),
    ("medium_101_140", 101, 140),
    ("long_141_plus", 141, 10**9),
)

OPTIONAL_STEP_GROUPS: dict[str, set[str]] = {
    "post_expose_bake": {"POST EXPOSE BAKE"},
    "hard_bake": {"HARD BAKE"},
    "pre_anneal_check": {"PRE ANNEAL CHECK"},
    "dry_wafer": {"DRY WAFER", "DRY WAFER BACKSIDE", "BACKSIDE DRY"},
    "optional_measurements": {
        "MEASURE THICKNESS",
        "MEASURE INITIAL THICKNESS",
        "MEASURE INITIAL GEOMETRY",
        "MEASURE GEOMETRY",
        "MEASURE SURFACE PARTICLES",
        "MEASURE SURFACE DEFECTS",
        "MEASURE BACKSIDE ROUGHNESS",
        "MEASURE EPITAXY THICKNESS",
        "MEASURE RESISTIVITY",
        "MEASURE OXIDE THICKNESS",
        "MEASURE FILM THICKNESS",
        "MEASURE DIELECTRIC THICKNESS",
        "MEASURE PLANARITY",
        "MEASURE SURFACE PLANARITY",
        "MEASURE VIA CD",
        "MEASURE CONTACT RESISTANCE",
        "MEASURE VIA RESISTANCE",
        "MEASURE METAL THICKNESS",
        "MEASURE LINE WIDTH",
        "MEASURE PASSIVATION THICKNESS",
        "MEASURE PASSIVATION QUALITY",
        "MEASURE PAD OPENING",
        "MEASURE BACKSIDE CONTACT",
        "FINAL THICKNESS MEASURE",
        "FINAL GEOMETRY CHECK",
        "FINAL OXIDE CHECK",
        "FINAL CD INSPECTION",
        "FINAL PARTICLE INSPECTION",
    },
    "package_preparation": {"PACKAGE PREPARATION"},
    "epitaxial_rework_check": {"EPITAXIAL REWORK CHECK"},
    "backside_thinning_check": {"BACKSIDE THINNING CHECK"},
    "final_electrical_test_prep": {"FINAL ELECTRICAL TEST PREP"},
}

EXPECTED_RULES = [
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
# CSV loading
# ---------------------------------------------------------------------------


def _normalise_header(name: str) -> str:
    return name.lstrip("\ufeff").strip().strip('"').strip().upper()


def infer_family_from_name(path: Path) -> str:
    name = path.name.lower()
    for fam in FAMILIES:
        if fam in name:
            return fam
    if "syntheticigbt" in name:
        return "igbt"
    if "syntheticic" in name:
        return "ic"
    return "unknown"


def read_sequences_from_csv(path: Path, family_hint: str | None = None) -> list[SequenceRecord]:
    """
    Read one sequence CSV.

    Supports:
    - SEQUENCE_ID, STEP
    - SEQUENCE_ID, FAMILY, STEP
    - STEP only, treated as one sequence
    """
    if not path.exists():
        raise FileNotFoundError(path)

    family_hint = family_hint.lower() if family_hint else None

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_fields = reader.fieldnames or []
        norm_to_raw = {_normalise_header(h): h for h in raw_fields}

        if "STEP" not in norm_to_raw:
            raise ValueError(
                f"Cannot parse {path}: expected a STEP column. Found headers: {raw_fields}"
            )

        seq_key = norm_to_raw.get("SEQUENCE_ID")
        step_key = norm_to_raw["STEP"]
        fam_key = norm_to_raw.get("FAMILY")

        grouped_steps: dict[str, list[str]] = defaultdict(list)
        grouped_family: dict[str, str] = {}

        for _row_idx, row in enumerate(reader):
            raw_step = (row.get(step_key) or "").strip().strip('"')
            if not raw_step:
                continue

            seq_id = (row.get(seq_key) or "seq_0001").strip() if seq_key else "seq_0001"
            family = (row.get(fam_key) or "").strip().lower() if fam_key else ""

            if not family:
                family = family_hint or infer_family_from_name(path)

            grouped_steps[seq_id].append(raw_step)
            grouped_family[seq_id] = family

    records = [
        SequenceRecord(
            sequence_id=f"{path.stem}:{seq_id}",
            family=grouped_family.get(seq_id, family_hint or infer_family_from_name(path)),
            steps=steps,
            source_file=str(path),
        )
        for seq_id, steps in grouped_steps.items()
    ]

    return records


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def length_bin(length: int) -> str:
    for label, lo, hi in LENGTH_BINS:
        if lo <= length <= hi:
            return label
    return "unknown"


def pairwise(items: list[str]) -> Iterable[tuple[str, str]]:
    for i in range(len(items) - 1):
        yield items[i], items[i + 1]


def trigrams(items: list[str]) -> Iterable[tuple[str, str, str]]:
    for i in range(len(items) - 2):
        yield items[i], items[i + 1], items[i + 2]


def stringify_tuple(values: tuple[str, ...]) -> str:
    return " -> ".join(values)


def extract_litho_levels(steps: list[str]) -> list[int]:
    levels: list[int] = []
    pattern = re.compile(r"^ALIGN MASK LEVEL\s+(\d+)$")
    for step in steps:
        match = pattern.match(step)
        if match:
            levels.append(int(match.group(1)))
    return levels


def classify_step_to_block(step: str) -> str:
    """
    Rule-based high-level process block classifier.

    This does not need to be perfect. It is for coverage monitoring and
    identifying underrepresented regions.
    """
    if step in {
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        "LOT RELEASE",
        "FINAL LOT RELEASE",
        "SHIP LOT",
    }:
        return "LOGISTICS"

    if "INSPECTION" in step or step.startswith("INSPECT") or "CHECK" in step:
        if step.startswith("FINAL"):
            return "FINAL_INSPECTION"
        return "INSPECTION"

    if step.startswith("MEASURE"):
        return "MEASUREMENT"

    if step in CLEAN_STEPS or "CLEAN" in step or "RINSE" in step or "DRY WAFER" in step:
        return "CLEANING"

    if step in DEPOSITION_STEPS or step.startswith("DEPOSIT") or "EPITAXIAL DEPOSITION" in step:
        return "DEPOSITION"

    if (
        "ANNEAL" in step
        or "THERMAL" in step
        or "DIFFUSION" in step
        or "DENSIFY" in step
        or "CURE" in step
    ):
        return "THERMAL"

    if (
        "LITHO" in step
        or "PHOTORESIST" in step
        or step.startswith("ALIGN MASK")
        or step.startswith("EXPOSE")
        or "BAKE" in step
        or "PATTERN" in step
    ):
        return "LITHOGRAPHY"

    if step in ETCH_STEPS or "ETCH" in step:
        return "ETCH"

    if step in IMPLANT_STEPS or step.startswith("IMPLANT"):
        return "IMPLANT"

    if step in CMP_STEPS or step.startswith("CMP"):
        return "CMP"

    if "VIA" in step or "CONTACT" in step:
        return "VIA_CONTACT"

    if "METAL" in step or "TUNGSTEN" in step:
        return "METALLIZATION"

    if "PASSIVATION" in step or "PAD WINDOW" in step or "BOND PAD" in step:
        return "PASSIVATION_PAD"

    if "BACKSIDE" in step or "GRIND" in step:
        return "BACKSIDE"

    if step in ELECTRICAL_TEST_STEPS or "TEST" in step or "YIELD" in step:
        return "TEST"

    if "PACKAGE" in step:
        return "PACKAGING"

    return "OTHER"


def nearest_previous_distance(
    steps: list[str],
    index: int,
    targets: set[str] | frozenset[str],
    max_window: int,
) -> int | None:
    """
    Return distance to nearest previous target step within max_window.
    Distance 1 means directly previous step.
    """
    lo = max(0, index - max_window)
    for j in range(index - 1, lo - 1, -1):
        if steps[j] in targets:
            return index - j
    return None


def has_previous_within(
    steps: list[str],
    index: int,
    predicate: Any,
    max_window: int,
) -> bool:
    lo = max(0, index - max_window)
    return any(predicate(steps[j]) for j in range(lo, index))


def extract_rule_boundary_features(steps: list[str]) -> Counter[str]:
    """
    Count valid/invalid boundary situations around the 10 rules.

    These are not replacements for validate_sequence().
    They are coverage features that tell us whether the dataset contains
    easy, medium, and boundary cases for the rules.
    """
    c: Counter[str] = Counter()

    # RULE_DEP_NO_CLEAN: deposition requires clean within prior 12.
    for i, step in enumerate(steps):
        if step in DEPOSITION_STEPS:
            dist = nearest_previous_distance(steps, i, CLEAN_STEPS, 12)
            if dist is None:
                c["RULE_DEP_NO_CLEAN:missing_or_gt12"] += 1
            else:
                c[f"RULE_DEP_NO_CLEAN:clean_distance={dist}"] += 1
                if dist == 12:
                    c["RULE_DEP_NO_CLEAN:boundary_exact_12"] += 1

    # RULE_METAL_ETCH_NO_LITHO: metal etch requires expose + develop within prior 15.
    for i, step in enumerate(steps):
        if step in METAL_ETCH_STEPS:
            has_expose = has_previous_within(
                steps,
                i,
                lambda s: s.startswith("EXPOSE LITHO LEVEL"),
                15,
            )
            has_develop = has_previous_within(
                steps,
                i,
                lambda s: s in {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"},
                15,
            )
            key = f"RULE_METAL_ETCH_NO_LITHO:expose={int(has_expose)}:develop={int(has_develop)}"
            c[key] += 1

    # RULE_ETCH_NO_MASK: patterned etch requires develop within prior 12.
    for i, step in enumerate(steps):
        if step in ETCH_STEPS:
            dist = nearest_previous_distance(
                steps,
                i,
                frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}),
                12,
            )
            if dist is None:
                c["RULE_ETCH_NO_MASK:develop_missing_or_gt12"] += 1
            else:
                c[f"RULE_ETCH_NO_MASK:develop_distance={dist}"] += 1
                if dist == 12:
                    c["RULE_ETCH_NO_MASK:boundary_exact_12"] += 1

    # RULE_LITHO_LEVEL_SKIP: sequential/non-decreasing litho mask levels.
    levels = extract_litho_levels(steps)
    if levels:
        c[f"RULE_LITHO_LEVEL_SKIP:num_levels={len(levels)}"] += 1
        for prev, curr in zip(levels, levels[1:], strict=False):
            delta = curr - prev
            c[f"RULE_LITHO_LEVEL_SKIP:delta={delta}"] += 1
            if delta == 1:
                c["RULE_LITHO_LEVEL_SKIP:sequential_increment"] += 1
            elif delta == 0:
                c["RULE_LITHO_LEVEL_SKIP:repeated_level"] += 1
            elif delta > 1:
                c["RULE_LITHO_LEVEL_SKIP:skipped_level"] += 1
            else:
                c["RULE_LITHO_LEVEL_SKIP:decreased_level"] += 1

    # RULE_IMPLANT_NO_MASK: implant requires opener within prior 15.
    for i, step in enumerate(steps):
        if step in IMPLANT_STEPS:
            dist = nearest_previous_distance(steps, i, IMPLANT_OPENER_STEPS, 15)
            if dist is None:
                c["RULE_IMPLANT_NO_MASK:opener_missing_or_gt15"] += 1
            else:
                c[f"RULE_IMPLANT_NO_MASK:opener_distance={dist}"] += 1
                if dist == 15:
                    c["RULE_IMPLANT_NO_MASK:boundary_exact_15"] += 1

    # RULE_CMP_NO_DEP: CMP requires deposition/fill within prior 6.
    for i, step in enumerate(steps):
        if step in CMP_STEPS:
            dist = nearest_previous_distance(steps, i, FILL_STEPS, 6)
            if dist is None:
                c["RULE_CMP_NO_DEP:fill_missing_or_gt6"] += 1
            else:
                c[f"RULE_CMP_NO_DEP:fill_distance={dist}"] += 1
                if dist == 6:
                    c["RULE_CMP_NO_DEP:boundary_exact_6"] += 1

    # RULE_PAD_OPEN_BEFORE_DEP: pad opening after passivation deposition and cure.
    last_passivation_dep: int | None = None
    last_cure: int | None = None
    for i, step in enumerate(steps):
        if step in {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"}:
            last_passivation_dep = i
        if step == "CURE PASSIVATION":
            last_cure = i
        if step in PAD_WINDOW_STEPS:
            has_dep_before = last_passivation_dep is not None and last_passivation_dep < i
            has_cure_before = last_cure is not None and last_cure < i
            c[
                "RULE_PAD_OPEN_BEFORE_DEP:"
                f"dep_before={int(has_dep_before)}:"
                f"cure_before={int(has_cure_before)}"
            ] += 1

    # RULE_TEST_BEFORE_PASSIVATION: electrical tests after cure passivation.
    first_cure_idx = next((i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None)
    for i, step in enumerate(steps):
        if step in ELECTRICAL_TEST_STEPS:
            after_cure = first_cure_idx is not None and i > first_cure_idx
            c[f"RULE_TEST_BEFORE_PASSIVATION:after_cure={int(after_cure)}"] += 1

    # RULE_SHIP_BEFORE_TEST: ship after wafer sort test.
    ship_idx = next((i for i, s in enumerate(steps) if s == "SHIP LOT"), None)
    sort_idx = next((i for i, s in enumerate(steps) if s == "WAFER SORT TEST"), None)
    if ship_idx is not None:
        after_sort = sort_idx is not None and ship_idx > sort_idx
        c[f"RULE_SHIP_BEFORE_TEST:ship_after_sort={int(after_sort)}"] += 1

    # RULE_BACKSIDE_BEFORE_PASSIVATION: backside metal after cure passivation.
    for i, step in enumerate(steps):
        if step in BACKSIDE_METAL_STEPS:
            after_cure = first_cure_idx is not None and i > first_cure_idx
            c[f"RULE_BACKSIDE_BEFORE_PASSIVATION:after_cure={int(after_cure)}"] += 1

    return c


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------


def compute_coverage(
    records: list[SequenceRecord],
    validate: bool = False,
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    length_bins: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    block_transition_counts: Counter[str] = Counter()
    litho_level_counts: Counter[str] = Counter()
    optional_presence_counts: Counter[str] = Counter()
    rule_boundary_counts: Counter[str] = Counter()
    source_file_counts: Counter[str] = Counter()

    validation_rule_counts: Counter[str] = Counter()
    invalid_sequence_ids: list[str] = []
    valid_sequences: int | None = None
    invalid_sequences: int | None = None

    lengths: list[int] = []

    for rec in records:
        steps = rec.steps
        family_counts[rec.family] += 1
        source_file_counts[rec.source_file] += 1
        lengths.append(len(steps))
        length_bins[length_bin(len(steps))] += 1

        step_counts.update(steps)

        for a, b in pairwise(steps):
            transition_counts[f"{a} -> {b}"] += 1

        for tri in trigrams(steps):
            trigram_counts[stringify_tuple(tri)] += 1

        blocks = [classify_step_to_block(s) for s in steps]
        block_counts.update(blocks)

        for a, b in pairwise(blocks):
            block_transition_counts[f"{a} -> {b}"] += 1

        for lvl in extract_litho_levels(steps):
            litho_level_counts[f"LEVEL_{lvl}"] += 1

        step_set = set(steps)
        for group_name, group_steps in OPTIONAL_STEP_GROUPS.items():
            present = bool(step_set & group_steps)
            optional_presence_counts[f"{group_name}:present={int(present)}"] += 1

        rule_boundary_counts.update(extract_rule_boundary_features(steps))

        if validate:
            if not GENERATOR_IMPORT_OK or validate_sequence is None:
                raise RuntimeError(
                    "Validation requested, but generate_sequences.py could not be imported. "
                    "Make sure coverage_tracker.py is in the same folder as generate_sequences.py."
                )

            violations = validate_sequence(steps)
            if violations:
                invalid_sequence_ids.append(rec.sequence_id)
                for v in violations:
                    validation_rule_counts[v.rule] += 1
            else:
                validation_rule_counts["VALID"] += 1

    if validate:
        invalid_sequences = len(invalid_sequence_ids)
        valid_sequences = len(records) - invalid_sequences

    summary = CoverageSummary(
        num_sequences=len(records),
        num_step_rows=sum(lengths),
        num_families=len(family_counts),
        families=dict(sorted(family_counts.items())),
        min_length=min(lengths) if lengths else 0,
        mean_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        max_length=max(lengths) if lengths else 0,
        unique_steps=len(step_counts),
        unique_transitions=len(transition_counts),
        unique_trigrams=len(trigram_counts),
        unique_blocks=len(block_counts),
        unique_block_transitions=len(block_transition_counts),
        validation_enabled=validate,
        validator_import_ok=GENERATOR_IMPORT_OK,
        valid_sequences=valid_sequences,
        invalid_sequences=invalid_sequences,
    )

    return {
        "summary": asdict(summary),
        "source_files": dict(sorted(source_file_counts.items())),
        "length_bins": dict(sorted(length_bins.items())),
        "step_counts": dict(step_counts.most_common()),
        "transition_counts": dict(transition_counts.most_common()),
        "trigram_counts": dict(trigram_counts.most_common()),
        "block_counts": dict(block_counts.most_common()),
        "block_transition_counts": dict(block_transition_counts.most_common()),
        "litho_level_counts": dict(sorted(litho_level_counts.items())),
        "optional_presence_counts": dict(sorted(optional_presence_counts.items())),
        "rule_boundary_counts": dict(rule_boundary_counts.most_common()),
        "validation_rule_counts": dict(validation_rule_counts.most_common()),
        "invalid_sequence_ids": invalid_sequence_ids,
    }


# ---------------------------------------------------------------------------
# Undercovered target extraction
# ---------------------------------------------------------------------------


def build_undercovered_targets(
    coverage: dict[str, Any],
    min_count: int,
) -> list[dict[str, Any]]:
    """
    Extract low-count coverage items.

    This does not know the full theoretical target universe yet. It reports
    observed-but-rare targets and missing high-level family/option/rule targets.
    """
    rows: list[dict[str, Any]] = []

    def add(target_type: str, target: str, count: int, note: str) -> None:
        rows.append(
            {
                "target_type": target_type,
                "target": target,
                "count": count,
                "min_count": min_count,
                "note": note,
            }
        )

    families = coverage["summary"]["families"]
    for fam in FAMILIES:
        count = int(families.get(fam, 0))
        if count < min_count:
            add("family", fam, count, "family has too few sequences")

    for bin_name, _, _ in LENGTH_BINS:
        count = int(coverage["length_bins"].get(bin_name, 0))
        if count < min_count:
            add("length_bin", bin_name, count, "length bin has too few sequences")

    for key, count in coverage["optional_presence_counts"].items():
        if int(count) < min_count:
            add("optional_presence", key, int(count), "optional present/absent state is rare")

    for key, count in coverage["litho_level_counts"].items():
        if int(count) < min_count:
            add("litho_level", key, int(count), "lithography level is rare")

    for key, count in coverage["block_transition_counts"].items():
        if int(count) < min_count:
            add("block_transition", key, int(count), "block transition is rare")

    for key, count in coverage["rule_boundary_counts"].items():
        if int(count) < min_count:
            add("rule_boundary", key, int(count), "rule-boundary case is rare")

    if coverage["summary"]["validation_enabled"]:
        validation_counts = coverage.get("validation_rule_counts", {})
        for rule in EXPECTED_RULES:
            count = int(validation_counts.get(rule, 0))
            if count == 0:
                add(
                    "invalid_rule",
                    rule,
                    0,
                    "no invalid examples for this rule; expected for valid-only datasets",
                )

    rows.sort(key=lambda r: (r["count"], r["target_type"], r["target"]))
    return rows


# ---------------------------------------------------------------------------
# Writing outputs
# ---------------------------------------------------------------------------


def write_counter_csv(path: Path, counter_dict: dict[str, int], key_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, "COUNT"])
        for key, count in counter_dict.items():
            writer.writerow([key, count])


def write_undercovered_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["target_type", "target", "count", "min_count", "note"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None._\n"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def top_items(counter_dict: dict[str, int], n: int) -> list[tuple[str, int]]:
    return list(counter_dict.items())[:n]


def bottom_items(counter_dict: dict[str, int], n: int) -> list[tuple[str, int]]:
    items = list(counter_dict.items())
    items.sort(key=lambda kv: kv[1])
    return items[:n]


def write_markdown_report(
    path: Path,
    coverage: dict[str, Any],
    undercovered: list[dict[str, Any]],
    top_k: int,
) -> None:
    summary = coverage["summary"]

    lines: list[str] = []
    lines.append("# Semiconductor Sequence Coverage Report\n")

    lines.append("## Summary\n")
    lines.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Number of sequences", summary["num_sequences"]],
                ["Number of step rows", summary["num_step_rows"]],
                ["Families covered", summary["num_families"]],
                ["Min sequence length", summary["min_length"]],
                ["Mean sequence length", f"{summary['mean_length']:.2f}"],
                ["Max sequence length", summary["max_length"]],
                ["Unique steps", summary["unique_steps"]],
                ["Unique adjacent transitions", summary["unique_transitions"]],
                ["Unique trigrams", summary["unique_trigrams"]],
                ["Unique blocks", summary["unique_blocks"]],
                ["Unique block transitions", summary["unique_block_transitions"]],
                ["Validation enabled", summary["validation_enabled"]],
                ["Validator import ok", summary["validator_import_ok"]],
                ["Valid sequences", summary["valid_sequences"]],
                ["Invalid sequences", summary["invalid_sequences"]],
            ],
        )
    )

    lines.append("\n## Family Counts\n")
    lines.append(
        markdown_table(
            ["Family", "Count"],
            [[k, v] for k, v in summary["families"].items()],
        )
    )

    lines.append("\n## Length Bins\n")
    lines.append(
        markdown_table(
            ["Length bin", "Count"],
            [[k, v] for k, v in coverage["length_bins"].items()],
        )
    )

    lines.append("\n## Optional Step Presence\n")
    lines.append(
        markdown_table(
            ["Optional feature", "Count"],
            [[k, v] for k, v in coverage["optional_presence_counts"].items()],
        )
    )

    lines.append("\n## Lithography Levels\n")
    lines.append(
        markdown_table(
            ["Litho level", "Count"],
            [[k, v] for k, v in coverage["litho_level_counts"].items()],
        )
    )

    lines.append(f"\n## Top {top_k} Steps\n")
    lines.append(
        markdown_table(
            ["Step", "Count"],
            [[k, v] for k, v in top_items(coverage["step_counts"], top_k)],
        )
    )

    lines.append(f"\n## Rarest {top_k} Observed Steps\n")
    lines.append(
        markdown_table(
            ["Step", "Count"],
            [[k, v] for k, v in bottom_items(coverage["step_counts"], top_k)],
        )
    )

    lines.append(f"\n## Top {top_k} Block Transitions\n")
    lines.append(
        markdown_table(
            ["Block transition", "Count"],
            [[k, v] for k, v in top_items(coverage["block_transition_counts"], top_k)],
        )
    )

    lines.append(f"\n## Rarest {top_k} Observed Block Transitions\n")
    lines.append(
        markdown_table(
            ["Block transition", "Count"],
            [[k, v] for k, v in bottom_items(coverage["block_transition_counts"], top_k)],
        )
    )

    lines.append(f"\n## Top {top_k} Rule-Boundary Cases\n")
    lines.append(
        markdown_table(
            ["Rule-boundary feature", "Count"],
            [[k, v] for k, v in top_items(coverage["rule_boundary_counts"], top_k)],
        )
    )

    lines.append(f"\n## Rarest {top_k} Observed Rule-Boundary Cases\n")
    lines.append(
        markdown_table(
            ["Rule-boundary feature", "Count"],
            [[k, v] for k, v in bottom_items(coverage["rule_boundary_counts"], top_k)],
        )
    )

    if coverage["summary"]["validation_enabled"]:
        lines.append("\n## Validation Rule Counts\n")
        lines.append(
            markdown_table(
                ["Validation result / rule", "Count"],
                [[k, v] for k, v in coverage["validation_rule_counts"].items()],
            )
        )

    lines.append("\n## Undercovered Targets\n")
    lines.append(
        markdown_table(
            ["Target type", "Target", "Count", "Min count", "Note"],
            [
                [
                    r["target_type"],
                    r["target"],
                    r["count"],
                    r["min_count"],
                    r["note"],
                ]
                for r in undercovered[:100]
            ],
        )
    )

    lines.append("\n## Interpretation\n")
    lines.append(
        "- If a family has low count, generate more sequences for that product family.\n"
        "- If an optional feature has low present or absent count, bias the generator toward that branch.\n"
        "- If rule-boundary cases are missing, add targeted generation around that rule.\n"
        "- If invalid-rule counts are zero, this is expected for valid-only generated data. "
        "The next milestone should add controlled invalid mutations.\n"
        "- If transitions or trigrams are highly skewed, coverage-guided generation should prefer "
        "candidates that introduce rare local patterns.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    coverage: dict[str, Any],
    undercovered: list[dict[str, Any]],
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "coverage_report.json"
    json_path.write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_markdown_report(
        output_dir / "coverage_report.md",
        coverage=coverage,
        undercovered=undercovered,
        top_k=top_k,
    )

    write_undercovered_csv(output_dir / "undercovered_targets.csv", undercovered)

    write_counter_csv(output_dir / "step_counts.csv", coverage["step_counts"], "STEP")
    write_counter_csv(
        output_dir / "transition_counts.csv", coverage["transition_counts"], "TRANSITION"
    )
    write_counter_csv(output_dir / "trigram_counts.csv", coverage["trigram_counts"], "TRIGRAM")
    write_counter_csv(
        output_dir / "block_transition_counts.csv",
        coverage["block_transition_counts"],
        "BLOCK_TRANSITION",
    )
    write_counter_csv(
        output_dir / "rule_boundary_counts.csv",
        coverage["rule_boundary_counts"],
        "RULE_BOUNDARY_FEATURE",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coverage_tracker.py",
        description="Compute coverage reports for semiconductor process-sequence CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more generated CSV files.",
    )

    parser.add_argument(
        "--family",
        choices=["mosfet", "igbt", "ic", "unknown"],
        default=None,
        help=(
            "Optional family hint. Use this when passing one file without a FAMILY column. "
            "For multiple files, family is usually inferred from each filename."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="coverage_outputs",
        help="Directory where coverage reports are written.",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run generate_sequences.validate_sequence on each sequence.",
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum desired count for undercovered observed targets.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top/rare items to show in the markdown report.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_paths = [Path(p) for p in args.input]
    output_dir = Path(args.output_dir)

    all_records: list[SequenceRecord] = []

    for path in input_paths:
        family_hint = args.family
        if len(input_paths) > 1 and args.family is not None:
            print(
                "[WARN] --family was provided with multiple input files. "
                "The same family hint will be applied to all files unless the CSV has a FAMILY column.",
                file=sys.stderr,
            )

        records = read_sequences_from_csv(path, family_hint=family_hint)
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No sequences were loaded. Check your input CSV files.")

    if args.validate and not GENERATOR_IMPORT_OK:
        raise RuntimeError(
            "You passed --validate, but generate_sequences.py could not be imported. "
            "Place coverage_tracker.py in the same folder as generate_sequences.py."
        )

    coverage = compute_coverage(all_records, validate=args.validate)
    undercovered = build_undercovered_targets(coverage, min_count=args.min_count)

    write_outputs(
        output_dir=output_dir,
        coverage=coverage,
        undercovered=undercovered,
        top_k=args.top_k,
    )

    summary = coverage["summary"]

    print("\nCoverage report written to:")
    print(f"  {output_dir / 'coverage_report.md'}")
    print(f"  {output_dir / 'coverage_report.json'}")
    print(f"  {output_dir / 'undercovered_targets.csv'}")

    print("\nSummary:")
    print(f"  sequences:              {summary['num_sequences']}")
    print(f"  step rows:              {summary['num_step_rows']}")
    print(f"  families:               {summary['families']}")
    print(f"  unique steps:           {summary['unique_steps']}")
    print(f"  unique transitions:     {summary['unique_transitions']}")
    print(f"  unique trigrams:        {summary['unique_trigrams']}")
    print(f"  unique block trans.:    {summary['unique_block_transitions']}")
    print(f"  validation enabled:     {summary['validation_enabled']}")
    print(f"  validator import ok:    {summary['validator_import_ok']}")

    if args.validate:
        print(f"  valid sequences:        {summary['valid_sequences']}")
        print(f"  invalid sequences:      {summary['invalid_sequences']}")

    print("\nNext step:")
    print("  Inspect coverage_outputs/undercovered_targets.csv")
    print("  Then bias the generator toward the rare/missing targets.")


if __name__ == "__main__":
    main()
