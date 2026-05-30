#!/usr/bin/env python3
"""
generate_coverage_guided.py — Coverage-guided valid sequence generation.

Place this file in:
    tracks/industrial-infineon/data/generate_coverage_guided.py

Expected project structure:
    tracks/industrial-infineon/
    ├── data/
    │   ├── coverage_tracker.py
    │   └── generate_coverage_guided.py
    └── training_data/
        ├── generate_sequences.py
        ├── MOSFET_variants.csv
        ├── IGBT_variants.csv
        └── IC_variants.csv

Purpose
-------
Generate a large, high-quality valid synthetic dataset by selecting new
sequences according to coverage gain, not just random uniqueness.

The script:
- imports the existing grammar generator from ../training_data/generate_sequences.py
- imports coverage utilities from coverage_tracker.py
- optionally seeds coverage with existing CSVs
- generates new valid candidates
- scores candidates by rare/new coverage features
- writes a long-format CSV
- writes a manifest
- writes a fresh coverage report

Example smoke test:
    python generate_coverage_guided.py \
      --existing ../training_data/MOSFET_variants.csv ../training_data/IGBT_variants.csv ../training_data/IC_variants.csv \
      --target-per-family 200 \
      --max-candidates-per-family 5000 \
      --output-dir coverage_guided_test \
      --include-existing-in-output

Example production run:
    python generate_coverage_guided.py \
      --existing ../training_data/MOSFET_variants.csv ../training_data/IGBT_variants.csv ../training_data/IC_variants.csv \
      --target-per-family 10000 \
      --max-candidates-per-family 400000 \
      --min-feature-count 250 \
      --output-dir coverage_guided_v1 \
      --include-existing-in-output \
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
    from generate_sequences import generate_sequence, validate_sequence  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "Could not import generate_sequence / validate_sequence from "
        f"{TRAINING_DATA_DIR / 'generate_sequences.py'}.\n"
        "Make sure generate_sequences.py is located in tracks/industrial-infineon/training_data/."
    ) from exc

try:
    from coverage_tracker import (  # type: ignore
        OPTIONAL_STEP_GROUPS,
        SequenceRecord,
        build_undercovered_targets,
        classify_step_to_block,
        compute_coverage,
        extract_litho_levels,
        extract_rule_boundary_features,
        length_bin,
        pairwise,
        read_sequences_from_csv,
        stringify_tuple,
        trigrams,
        write_outputs,
    )
except Exception as exc:
    raise RuntimeError(
        "Could not import utilities from coverage_tracker.py.\n"
        "Make sure coverage_tracker.py is located in tracks/industrial-infineon/data/."
    ) from exc


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

FAMILIES = ("mosfet", "igbt", "ic")


@dataclass
class GeneratedSample:
    sequence_id: str
    family: str
    steps: list[str]
    score: float
    score_details: dict[str, float]
    source: str
    seed: int
    attempt_index: int


@dataclass
class FamilyGenerationStats:
    family: str
    target_new_sequences: int
    accepted_new_sequences: int
    candidates_seen: int
    duplicate_candidates: int
    invalid_candidates: int
    forced_accepts: int
    mean_accept_score: float
    max_accept_score: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def sequence_hash(steps: list[str]) -> str:
    text = "\n".join(steps)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_split(sequence_id: str, seed: int) -> str:
    """
    Deterministic 80/10/10 split based on sequence ID and seed.
    """
    key = f"{sequence_id}|{seed}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def infer_family_from_record(record: SequenceRecord) -> str:
    fam = (record.family or "").lower()
    if fam in FAMILIES:
        return fam

    name = Path(record.source_file).name.lower()
    for candidate in FAMILIES:
        if candidate in name:
            return candidate
    return "unknown"


# ---------------------------------------------------------------------------
# Coverage feature extraction
# ---------------------------------------------------------------------------

FEATURE_WEIGHTS: dict[str, float] = {
    "family": 10.0,
    "length_bin": 8.0,
    "step": 4.0,
    "transition": 6.0,
    "trigram": 2.0,
    "block": 3.0,
    "block_transition": 8.0,
    "litho_level": 5.0,
    "optional_presence": 10.0,
    "rule_boundary": 12.0,
}


def features_for_sequence(family: str, steps: list[str]) -> dict[str, set[str]]:
    """
    Extract coverage features from one candidate sequence.

    These are sequence-level coverage features, not raw occurrence counts.
    A feature is counted once per sequence when updating the coverage table.
    """
    blocks = [classify_step_to_block(step) for step in steps]
    step_set = set(steps)

    features: dict[str, set[str]] = {
        "family": {family},
        "length_bin": {length_bin(len(steps))},
        "step": set(steps),
        "transition": {f"{a} -> {b}" for a, b in pairwise(steps)},
        "trigram": {stringify_tuple(t) for t in trigrams(steps)},
        "block": set(blocks),
        "block_transition": {f"{a} -> {b}" for a, b in pairwise(blocks)},
        "litho_level": {f"LEVEL_{lvl}" for lvl in extract_litho_levels(steps)},
        "optional_presence": set(),
        "rule_boundary": set(extract_rule_boundary_features(steps).keys()),
    }

    for group_name, group_steps in OPTIONAL_STEP_GROUPS.items():
        present = bool(step_set & group_steps)
        features["optional_presence"].add(f"{group_name}:present={int(present)}")

    return features


def update_coverage_counts(
    coverage_counts: Counter[str],
    features: dict[str, set[str]],
) -> None:
    for feature_type, values in features.items():
        for value in values:
            coverage_counts[f"{feature_type}::{value}"] += 1


def score_candidate(
    coverage_counts: Counter[str],
    features: dict[str, set[str]],
    min_feature_count: int,
) -> tuple[float, dict[str, float]]:
    """
    Score a candidate by how much it helps rare or unseen coverage features.

    High score = good candidate.

    The score rewards:
    - completely new features,
    - observed but underrepresented features,
    - rare rule-boundary cases,
    - rare optional-feature combinations,
    - rare transitions/trigrams.
    """
    total = 0.0
    details: dict[str, float] = defaultdict(float)

    for feature_type, values in features.items():
        weight = FEATURE_WEIGHTS.get(feature_type, 1.0)

        for value in values:
            key = f"{feature_type}::{value}"
            count = coverage_counts.get(key, 0)

            if count == 0:
                gain = 10.0 * weight
            elif count < min_feature_count:
                rarity = (min_feature_count - count) / max(1, min_feature_count)
                gain = weight * rarity
            else:
                gain = 0.0

            if gain > 0:
                total += gain
                details[feature_type] += gain

    return total, dict(details)


def seed_coverage_from_records(
    records: list[SequenceRecord],
    coverage_counts: Counter[str],
    seen_hashes: set[str],
) -> None:
    for record in records:
        family = infer_family_from_record(record)
        features = features_for_sequence(family, record.steps)
        update_coverage_counts(coverage_counts, features)
        seen_hashes.add(sequence_hash(record.steps))


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def write_long_csv(
    path: Path,
    existing_records: list[SequenceRecord],
    new_samples: list[GeneratedSample],
    include_existing: bool,
    split_seed: int,
) -> None:
    """
    Write long-format CSV:

        SEQUENCE_ID,FAMILY,STEP_INDEX,STEP,IS_VALID,SPLIT,SOURCE,GENERATION_SCORE
    """
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
                "SPLIT",
                "SOURCE",
                "GENERATION_SCORE",
            ]
        )

        if include_existing:
            for _idx, record in enumerate(existing_records, start=1):
                family = infer_family_from_record(record)
                clean_id = record.sequence_id.replace(":", "_").replace("\\", "_").replace("/", "_")
                sequence_id = f"existing_{clean_id}"
                split = stable_split(sequence_id, split_seed)

                for step_index, step in enumerate(record.steps):
                    writer.writerow(
                        [
                            sequence_id,
                            family,
                            step_index,
                            step,
                            1,
                            split,
                            "existing",
                            "",
                        ]
                    )

        for sample in new_samples:
            split = stable_split(sample.sequence_id, split_seed)
            for step_index, step in enumerate(sample.steps):
                writer.writerow(
                    [
                        sample.sequence_id,
                        sample.family,
                        step_index,
                        step,
                        1,
                        split,
                        sample.source,
                        f"{sample.score:.6f}",
                    ]
                )


def write_sequence_summary_csv(
    path: Path,
    existing_records: list[SequenceRecord],
    new_samples: list[GeneratedSample],
    include_existing: bool,
    split_seed: int,
) -> None:
    """
    Write one row per sequence. Useful for debugging and filtering.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "SEQUENCE_ID",
                "FAMILY",
                "LENGTH",
                "IS_VALID",
                "SPLIT",
                "SOURCE",
                "GENERATION_SCORE",
                "SCORE_DETAILS_JSON",
            ]
        )

        if include_existing:
            for record in existing_records:
                family = infer_family_from_record(record)
                clean_id = record.sequence_id.replace(":", "_").replace("\\", "_").replace("/", "_")
                sequence_id = f"existing_{clean_id}"
                writer.writerow(
                    [
                        sequence_id,
                        family,
                        len(record.steps),
                        1,
                        stable_split(sequence_id, split_seed),
                        "existing",
                        "",
                        "",
                    ]
                )

        for sample in new_samples:
            writer.writerow(
                [
                    sample.sequence_id,
                    sample.family,
                    len(sample.steps),
                    1,
                    stable_split(sample.sequence_id, split_seed),
                    sample.source,
                    f"{sample.score:.6f}",
                    json.dumps(sample.score_details, sort_keys=True),
                ]
            )


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_for_family(
    family: str,
    target_new_sequences: int,
    max_candidates: int,
    min_feature_count: int,
    min_accept_score: float,
    force_accept_after: int,
    seed: int,
    coverage_counts: Counter[str],
    seen_hashes: set[str],
    progress_every: int,
) -> tuple[list[GeneratedSample], FamilyGenerationStats]:
    """
    Generate coverage-guided valid sequences for one family.
    """
    start_time = time.time()
    rng = random.Random(seed)

    accepted: list[GeneratedSample] = []

    candidates_seen = 0
    duplicate_candidates = 0
    invalid_candidates = 0
    forced_accepts = 0

    attempts_since_accept = 0
    best_buffered: tuple[list[str], float, dict[str, float], int] | None = None

    def accept_candidate(
        steps: list[str],
        score: float,
        score_details: dict[str, float],
        attempt_index: int,
        forced: bool,
    ) -> None:
        nonlocal forced_accepts, attempts_since_accept, best_buffered

        seq_id = f"{family}_cov_{len(accepted) + 1:06d}"

        sample = GeneratedSample(
            sequence_id=seq_id,
            family=family,
            steps=steps,
            score=score,
            score_details=score_details,
            source="coverage_guided",
            seed=seed,
            attempt_index=attempt_index,
        )

        accepted.append(sample)
        seen_hashes.add(sequence_hash(steps))

        features = features_for_sequence(family, steps)
        update_coverage_counts(coverage_counts, features)

        if forced:
            forced_accepts += 1

        attempts_since_accept = 0
        best_buffered = None

    while len(accepted) < target_new_sequences and candidates_seen < max_candidates:
        candidates_seen += 1
        attempts_since_accept += 1

        try:
            steps = generate_sequence(family, rng)
        except Exception as exc:
            print(f"[WARN] Generator failed for family={family}: {exc}", file=sys.stderr)
            continue

        seq_hash = sequence_hash(steps)
        if seq_hash in seen_hashes:
            duplicate_candidates += 1
            continue

        violations = validate_sequence(steps)
        if violations:
            invalid_candidates += 1
            continue

        features = features_for_sequence(family, steps)
        score, score_details = score_candidate(
            coverage_counts=coverage_counts,
            features=features,
            min_feature_count=min_feature_count,
        )

        if best_buffered is None or score > best_buffered[1]:
            best_buffered = (steps, score, score_details, candidates_seen)

        if score >= min_accept_score:
            accept_candidate(
                steps=steps,
                score=score,
                score_details=score_details,
                attempt_index=candidates_seen,
                forced=False,
            )

        elif attempts_since_accept >= force_accept_after and best_buffered is not None:
            buffered_steps, buffered_score, buffered_details, buffered_attempt = best_buffered

            if sequence_hash(buffered_steps) not in seen_hashes:
                accept_candidate(
                    steps=buffered_steps,
                    score=buffered_score,
                    score_details=buffered_details,
                    attempt_index=buffered_attempt,
                    forced=True,
                )
            else:
                best_buffered = None
                attempts_since_accept = 0

        if progress_every > 0 and len(accepted) > 0 and len(accepted) % progress_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[{family}] accepted={len(accepted):,}/{target_new_sequences:,} "
                f"candidates={candidates_seen:,} "
                f"duplicates={duplicate_candidates:,} "
                f"invalid={invalid_candidates:,} "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    scores = [s.score for s in accepted]

    stats = FamilyGenerationStats(
        family=family,
        target_new_sequences=target_new_sequences,
        accepted_new_sequences=len(accepted),
        candidates_seen=candidates_seen,
        duplicate_candidates=duplicate_candidates,
        invalid_candidates=invalid_candidates,
        forced_accepts=forced_accepts,
        mean_accept_score=(sum(scores) / len(scores)) if scores else 0.0,
        max_accept_score=max(scores) if scores else 0.0,
        elapsed_seconds=elapsed,
    )

    return accepted, stats


# ---------------------------------------------------------------------------
# Existing data loading
# ---------------------------------------------------------------------------


def load_existing_records(paths: list[Path]) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for path in paths:
        loaded = read_sequences_from_csv(path)
        records.extend(loaded)
    return records


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    existing_records: list[SequenceRecord],
    new_samples: list[GeneratedSample],
    family_stats: list[FamilyGenerationStats],
    output_files: dict[str, str],
) -> None:
    family_counts_existing: Counter[str] = Counter(
        infer_family_from_record(record) for record in existing_records
    )
    family_counts_new: Counter[str] = Counter(sample.family for sample in new_samples)

    total_step_rows_existing = sum(len(r.steps) for r in existing_records)
    total_step_rows_new = sum(len(s.steps) for s in new_samples)

    manifest = {
        "created_at_unix": time.time(),
        "script": "generate_coverage_guided.py",
        "arguments": vars(args),
        "existing": {
            "num_sequences": len(existing_records),
            "num_step_rows": total_step_rows_existing,
            "family_counts": dict(sorted(family_counts_existing.items())),
        },
        "generated_new": {
            "num_sequences": len(new_samples),
            "num_step_rows": total_step_rows_new,
            "family_counts": dict(sorted(family_counts_new.items())),
        },
        "combined_if_existing_included": {
            "num_sequences": len(new_samples)
            + (len(existing_records) if args.include_existing_in_output else 0),
            "num_step_rows": total_step_rows_new
            + (total_step_rows_existing if args.include_existing_in_output else 0),
        },
        "family_generation_stats": [asdict(s) for s in family_stats],
        "output_files": output_files,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_coverage_guided.py",
        description="Generate large coverage-guided valid semiconductor process datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--existing",
        nargs="*",
        default=[],
        help=(
            "Existing CSVs used to seed coverage and avoid duplicates. "
            "Example: ../training_data/MOSFET_variants.csv ../training_data/IGBT_variants.csv ../training_data/IC_variants.csv"
        ),
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=list(FAMILIES),
        default=list(FAMILIES),
        help="Families to generate.",
    )

    parser.add_argument(
        "--target-per-family",
        type=int,
        default=10000,
        help="Number of NEW coverage-guided sequences to generate per family.",
    )

    parser.add_argument(
        "--max-candidates-per-family",
        type=int,
        default=400000,
        help="Maximum candidate sequences sampled per family before stopping.",
    )

    parser.add_argument(
        "--min-feature-count",
        type=int,
        default=250,
        help=(
            "Target count for coverage features. Higher values produce more balancing pressure "
            "toward rare features."
        ),
    )

    parser.add_argument(
        "--min-accept-score",
        type=float,
        default=30.0,
        help=(
            "Candidate is accepted immediately if its coverage score is at least this value. "
            "Lower values accept more, higher values are stricter."
        ),
    )

    parser.add_argument(
        "--force-accept-after",
        type=int,
        default=500,
        help=(
            "If no candidate passes min-accept-score for this many attempts, accept the best "
            "candidate seen in that window. This guarantees progress while still preferring quality."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="coverage_guided_v1",
        help="Output directory.",
    )

    parser.add_argument(
        "--include-existing-in-output",
        action="store_true",
        help="Include existing seed sequences in the final output CSV.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N accepted sequences per family. Use 0 to disable.",
    )

    parser.add_argument(
        "--coverage-report-min-count",
        type=int,
        default=20,
        help="Minimum count used for undercovered target extraction in the final coverage report.",
    )

    parser.add_argument(
        "--coverage-report-top-k",
        type=int,
        default=25,
        help="Top/rare items shown in the final markdown coverage report.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_paths = [Path(p) for p in args.existing]
    existing_records: list[SequenceRecord] = []

    if existing_paths:
        print("Loading existing seed data...")
        existing_records = load_existing_records(existing_paths)
        print(f"  Loaded {len(existing_records):,} existing sequences.")

    coverage_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()

    if existing_records:
        print("Seeding coverage from existing data...")
        seed_coverage_from_records(existing_records, coverage_counts, seen_hashes)
        print(f"  Seeded {len(coverage_counts):,} coverage features.")
        print(f"  Seeded {len(seen_hashes):,} known sequence hashes.")

    all_new_samples: list[GeneratedSample] = []
    family_stats: list[FamilyGenerationStats] = []

    for fam_idx, family in enumerate(args.families):
        family_seed = args.seed + 1009 * fam_idx

        print("\n" + "=" * 80)
        print(f"Generating family: {family}")
        print("=" * 80)

        samples, stats = generate_for_family(
            family=family,
            target_new_sequences=args.target_per_family,
            max_candidates=args.max_candidates_per_family,
            min_feature_count=args.min_feature_count,
            min_accept_score=args.min_accept_score,
            force_accept_after=args.force_accept_after,
            seed=family_seed,
            coverage_counts=coverage_counts,
            seen_hashes=seen_hashes,
            progress_every=args.progress_every,
        )

        all_new_samples.extend(samples)
        family_stats.append(stats)

        print(f"\nFinished {family}:")
        print(f"  accepted_new_sequences: {stats.accepted_new_sequences:,}")
        print(f"  candidates_seen:        {stats.candidates_seen:,}")
        print(f"  duplicates:             {stats.duplicate_candidates:,}")
        print(f"  invalid_candidates:     {stats.invalid_candidates:,}")
        print(f"  forced_accepts:         {stats.forced_accepts:,}")
        print(f"  mean_accept_score:      {stats.mean_accept_score:.3f}")
        print(f"  max_accept_score:       {stats.max_accept_score:.3f}")
        print(f"  elapsed_seconds:        {stats.elapsed_seconds:.1f}")

    long_csv_path = output_dir / "coverage_guided_sequences.csv"
    summary_csv_path = output_dir / "coverage_guided_sequence_summary.csv"
    manifest_path = output_dir / "coverage_guided_manifest.json"

    print("\nWriting dataset files...")

    write_long_csv(
        path=long_csv_path,
        existing_records=existing_records,
        new_samples=all_new_samples,
        include_existing=args.include_existing_in_output,
        split_seed=args.seed,
    )

    write_sequence_summary_csv(
        path=summary_csv_path,
        existing_records=existing_records,
        new_samples=all_new_samples,
        include_existing=args.include_existing_in_output,
        split_seed=args.seed,
    )

    # Build records for final coverage report.
    report_records: list[SequenceRecord] = []

    if args.include_existing_in_output:
        report_records.extend(existing_records)

    for sample in all_new_samples:
        report_records.append(
            SequenceRecord(
                sequence_id=sample.sequence_id,
                family=sample.family,
                steps=sample.steps,
                source_file=str(long_csv_path),
            )
        )

    print("Computing final coverage report...")
    final_coverage = compute_coverage(report_records, validate=True)
    undercovered = build_undercovered_targets(
        final_coverage,
        min_count=args.coverage_report_min_count,
    )

    coverage_report_dir = output_dir / "coverage_report"
    write_outputs(
        output_dir=coverage_report_dir,
        coverage=final_coverage,
        undercovered=undercovered,
        top_k=args.coverage_report_top_k,
    )

    output_files = {
        "long_csv": str(long_csv_path),
        "sequence_summary_csv": str(summary_csv_path),
        "manifest_json": str(manifest_path),
        "coverage_report_md": str(coverage_report_dir / "coverage_report.md"),
        "coverage_report_json": str(coverage_report_dir / "coverage_report.json"),
        "undercovered_targets_csv": str(coverage_report_dir / "undercovered_targets.csv"),
    }

    write_manifest(
        path=manifest_path,
        args=args,
        existing_records=existing_records,
        new_samples=all_new_samples,
        family_stats=family_stats,
        output_files=output_files,
    )

    print("\nDone.")
    print("\nGenerated files:")
    print(f"  {long_csv_path}")
    print(f"  {summary_csv_path}")
    print(f"  {manifest_path}")
    print(f"  {coverage_report_dir / 'coverage_report.md'}")
    print(f"  {coverage_report_dir / 'undercovered_targets.csv'}")

    print("\nFinal summary:")
    print(f"  existing sequences loaded:     {len(existing_records):,}")
    print(f"  new sequences generated:       {len(all_new_samples):,}")
    print(
        "  final output sequences:        "
        f"{len(all_new_samples) + (len(existing_records) if args.include_existing_in_output else 0):,}"
    )
    print(f"  final valid sequences:         {final_coverage['summary']['valid_sequences']:,}")
    print(f"  final invalid sequences:       {final_coverage['summary']['invalid_sequences']:,}")
    print(f"  unique steps:                  {final_coverage['summary']['unique_steps']:,}")
    print(f"  unique transitions:            {final_coverage['summary']['unique_transitions']:,}")
    print(f"  unique trigrams:               {final_coverage['summary']['unique_trigrams']:,}")
    print(
        f"  unique block transitions:      {final_coverage['summary']['unique_block_transitions']:,}"
    )

    print("\nNext step after this:")
    print("  Inspect the new coverage report.")
    print("  Then we add controlled invalid-sequence generation.")


if __name__ == "__main__":
    main()
