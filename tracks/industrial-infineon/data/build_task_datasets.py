#!/usr/bin/env python3
"""
build_task_datasets.py — Build model-ready datasets from generated semiconductor sequences.

Place this file in:
    tracks/industrial-infineon/data/build_task_datasets.py

Expected inputs after previous pipeline steps:
    coverage_guided_v1/coverage_guided_sequences.csv
    easy_invalid_v1/invalid_sequences.csv
    hard_invalid_v1/hard_invalid_sequences.csv

Outputs:
    task_datasets_v1/next_step_prediction.csv
    task_datasets_v1/sequence_completion.csv
    task_datasets_v1/anomaly_detection.csv
    task_datasets_v1/rule_attribution.csv
    task_datasets_v1/sequence_summary.csv
    task_datasets_v1/task_dataset_manifest.json
    task_datasets_v1/task_dataset_report.md

Recommended smoke test:
    python build_task_datasets.py \
      --valid-input coverage_guided_v1/coverage_guided_sequences.csv \
      --easy-invalid-input easy_invalid_v1/invalid_sequences.csv \
      --hard-invalid-input hard_invalid_v1/hard_invalid_sequences.csv \
      --output-dir task_datasets_test \
      --max-valid-sequences 100 \
      --max-easy-invalid-sequences 100 \
      --max-hard-invalid-sequences 100 \
      --context-window 64 \
      --next-step-stride 1 \
      --seed 42

Recommended production run:
    python build_task_datasets.py \
      --valid-input coverage_guided_v1/coverage_guided_sequences.csv \
      --easy-invalid-input easy_invalid_v1/invalid_sequences.csv \
      --hard-invalid-input hard_invalid_v1/hard_invalid_sequences.csv \
      --output-dir task_datasets_v1 \
      --context-window 64 \
      --next-step-stride 1 \
      --completion-cut-fracs 0.25 0.5 0.75 \
      --completion-target-window 96 \
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


STEP_SEP = " ||| "


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SequenceItem:
    sequence_id: str
    origin_id: str
    family: str
    steps: list[str]
    is_valid: int
    violated_rule: str
    validator_rules: str
    validator_violation_count: int
    source: str
    input_kind: str
    input_file: str
    input_split: str


@dataclass
class BuildStats:
    valid_sequences: int = 0
    easy_invalid_sequences: int = 0
    hard_invalid_sequences: int = 0
    skipped_duplicate_sequences: int = 0
    next_step_examples: int = 0
    completion_examples: int = 0
    anomaly_examples: int = 0
    rule_attribution_examples: int = 0
    combined_long_rows: int = 0


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalise_header(name: str) -> str:
    return name.lstrip("\ufeff").strip().strip('"').strip().upper()


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


def stable_split(key: str, seed: int) -> str:
    """
    Deterministic 80/10/10 split.

    We split by origin_id by default so invalid mutations derived from a valid
    sequence end up in the same train/val/test split as the corresponding
    original sequence. This reduces leakage.
    """
    h = int(hashlib.md5(f"{key}|{seed}".encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def bool_to_int(value: str, default: int) -> int:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return 1
    if value in {"0", "false", "no", "n"}:
        return 0
    return default


def safe_join_steps(steps: list[str]) -> str:
    return STEP_SEP.join(steps)


def get_col(row: dict[str, str], norm_to_raw: dict[str, str], key: str, default: str = "") -> str:
    raw_key = norm_to_raw.get(key)
    if raw_key is None:
        return default
    return (row.get(raw_key) or default).strip()


def infer_family_from_text(text: str) -> str:
    text = text.lower()
    for fam in ("mosfet", "igbt", "ic"):
        if fam in text:
            return fam
    return "unknown"


def sequence_split(item: SequenceItem, seed: int, split_mode: str) -> str:
    if split_mode == "input" and item.input_split:
        return item.input_split
    if split_mode == "sequence":
        return stable_split(item.sequence_id, seed)
    return stable_split(item.origin_id, seed)


# ---------------------------------------------------------------------------
# Streaming CSV reader
# ---------------------------------------------------------------------------

def make_sequence_item(
    path: Path,
    input_kind: str,
    sequence_id: str,
    first_row: dict[str, str],
    norm_to_raw: dict[str, str],
    indexed_steps: list[tuple[int, str]],
) -> SequenceItem:
    indexed_steps = sorted(indexed_steps, key=lambda x: x[0])
    steps = [s for _, s in indexed_steps]

    family = get_col(first_row, norm_to_raw, "FAMILY", "")
    if not family:
        family = infer_family_from_text(f"{path.name} {sequence_id}")

    if input_kind == "valid":
        default_valid = 1
    else:
        default_valid = 0

    is_valid = bool_to_int(get_col(first_row, norm_to_raw, "IS_VALID", ""), default=default_valid)

    violated_rule = get_col(first_row, norm_to_raw, "VIOLATED_RULE", "")
    validator_rules = get_col(first_row, norm_to_raw, "VALIDATOR_RULES", "")
    validator_violation_count_str = get_col(first_row, norm_to_raw, "VALIDATOR_VIOLATION_COUNT", "0")
    source = get_col(first_row, norm_to_raw, "SOURCE", input_kind)
    input_split = get_col(first_row, norm_to_raw, "SPLIT", "")

    try:
        validator_violation_count = int(float(validator_violation_count_str))
    except ValueError:
        validator_violation_count = 0

    original_sequence_id = get_col(first_row, norm_to_raw, "ORIGINAL_SEQUENCE_ID", "")

    if input_kind == "valid":
        # This matches the origin IDs stored by the invalid generators, which
        # used coverage_tracker.SequenceRecord IDs of the form:
        #   coverage_guided_sequences:<SEQUENCE_ID>
        origin_id = f"{path.stem}:{sequence_id}"
        output_sequence_id = f"valid_{clean_id(origin_id)}"
    else:
        origin_id = original_sequence_id or f"{path.stem}:{sequence_id}"
        output_sequence_id = sequence_id

    return SequenceItem(
        sequence_id=output_sequence_id,
        origin_id=origin_id,
        family=family,
        steps=steps,
        is_valid=is_valid,
        violated_rule=violated_rule,
        validator_rules=validator_rules,
        validator_violation_count=validator_violation_count,
        source=source,
        input_kind=input_kind,
        input_file=str(path),
        input_split=input_split,
    )


def iter_sequences_from_long_csv(path: Path, input_kind: str) -> Iterable[SequenceItem]:
    """
    Stream grouped long-format CSVs.

    Assumption:
        Rows are grouped by SEQUENCE_ID.
    This is true for all files generated by the previous scripts.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_fields = reader.fieldnames or []
        norm_to_raw = {normalise_header(h): h for h in raw_fields}

        if "STEP" not in norm_to_raw:
            raise ValueError(f"{path} is missing STEP column. Headers: {raw_fields}")

        seq_key = norm_to_raw.get("SEQUENCE_ID")
        step_key = norm_to_raw["STEP"]
        step_index_key = norm_to_raw.get("STEP_INDEX")

        current_sequence_id: Optional[str] = None
        current_first_row: Optional[dict[str, str]] = None
        current_steps: list[tuple[int, str]] = []

        def flush_current() -> Optional[SequenceItem]:
            if current_sequence_id is None or current_first_row is None:
                return None
            return make_sequence_item(
                path=path,
                input_kind=input_kind,
                sequence_id=current_sequence_id,
                first_row=current_first_row,
                norm_to_raw=norm_to_raw,
                indexed_steps=current_steps,
            )

        for row_number, row in enumerate(reader):
            raw_step = (row.get(step_key) or "").strip().strip('"')
            if not raw_step:
                continue

            sequence_id = (row.get(seq_key) or "seq_0001").strip() if seq_key else "seq_0001"

            if step_index_key:
                try:
                    step_index = int(float((row.get(step_index_key) or "").strip()))
                except ValueError:
                    step_index = len(current_steps)
            else:
                step_index = len(current_steps)

            if current_sequence_id is None:
                current_sequence_id = sequence_id
                current_first_row = dict(row)
                current_steps = []

            elif sequence_id != current_sequence_id:
                item = flush_current()
                if item is not None:
                    yield item

                current_sequence_id = sequence_id
                current_first_row = dict(row)
                current_steps = []

            current_steps.append((step_index, raw_step))

        item = flush_current()
        if item is not None:
            yield item


# ---------------------------------------------------------------------------
# Dataset writers
# ---------------------------------------------------------------------------

def write_next_step_examples(
    writer: csv.writer,
    item: SequenceItem,
    split: str,
    context_window: int,
    stride: int,
    min_prefix_len: int,
    stats: BuildStats,
) -> None:
    if item.is_valid != 1:
        return

    example_idx = 0

    for target_index in range(min_prefix_len, len(item.steps), stride):
        prefix = item.steps[:target_index]
        target = item.steps[target_index]

        if context_window > 0:
            context = prefix[-context_window:]
        else:
            context = prefix

        example_id = f"next_{clean_id(item.sequence_id)}_{target_index:04d}"

        writer.writerow([
            example_id,
            item.sequence_id,
            item.origin_id,
            item.family,
            split,
            target_index,
            len(prefix),
            context_window,
            safe_join_steps(context),
            target,
            item.source,
        ])

        example_idx += 1
        stats.next_step_examples += 1


def write_completion_examples(
    writer: csv.writer,
    item: SequenceItem,
    split: str,
    cut_fracs: list[float],
    context_window: int,
    target_window: int,
    stats: BuildStats,
) -> None:
    if item.is_valid != 1:
        return

    n = len(item.steps)
    used_cut_indices: set[int] = set()

    for frac in cut_fracs:
        if frac <= 0 or frac >= 1:
            continue

        cut_index = int(round(n * frac))
        cut_index = max(1, min(cut_index, n - 1))

        if cut_index in used_cut_indices:
            continue
        used_cut_indices.add(cut_index)

        prefix = item.steps[:cut_index]
        suffix = item.steps[cut_index:]

        if context_window > 0:
            prefix_context = prefix[-context_window:]
        else:
            prefix_context = prefix

        if target_window > 0:
            target_suffix = suffix[:target_window]
            truncated = int(len(suffix) > target_window)
        else:
            target_suffix = suffix
            truncated = 0

        example_id = f"completion_{clean_id(item.sequence_id)}_{cut_index:04d}"

        writer.writerow([
            example_id,
            item.sequence_id,
            item.origin_id,
            item.family,
            split,
            f"{frac:.4f}",
            cut_index,
            n,
            context_window,
            target_window,
            safe_join_steps(prefix_context),
            safe_join_steps(target_suffix),
            truncated,
            item.source,
        ])

        stats.completion_examples += 1


def write_anomaly_example(
    writer: csv.writer,
    item: SequenceItem,
    split: str,
    stats: BuildStats,
) -> None:
    example_id = f"anomaly_{clean_id(item.sequence_id)}"

    writer.writerow([
        example_id,
        item.sequence_id,
        item.origin_id,
        item.family,
        split,
        len(item.steps),
        safe_join_steps(item.steps),
        item.is_valid,
        item.violated_rule,
        item.validator_rules,
        item.validator_violation_count,
        item.input_kind,
        item.source,
    ])

    stats.anomaly_examples += 1


def write_rule_attribution_example(
    writer: csv.writer,
    item: SequenceItem,
    split: str,
    stats: BuildStats,
) -> None:
    if item.is_valid == 1:
        return

    example_id = f"rule_{clean_id(item.sequence_id)}"

    writer.writerow([
        example_id,
        item.sequence_id,
        item.origin_id,
        item.family,
        split,
        len(item.steps),
        safe_join_steps(item.steps),
        item.violated_rule,
        item.validator_rules,
        item.validator_violation_count,
        item.input_kind,
        item.source,
    ])

    stats.rule_attribution_examples += 1


def write_sequence_summary(
    writer: csv.writer,
    item: SequenceItem,
    split: str,
) -> None:
    writer.writerow([
        item.sequence_id,
        item.origin_id,
        item.family,
        split,
        len(item.steps),
        item.is_valid,
        item.violated_rule,
        item.validator_rules,
        item.validator_violation_count,
        item.input_kind,
        item.source,
        item.input_file,
    ])


def write_combined_long(
    writer: csv.writer,
    item: SequenceItem,
    split: str,
    stats: BuildStats,
) -> None:
    for idx, step in enumerate(item.steps):
        writer.writerow([
            item.sequence_id,
            item.origin_id,
            item.family,
            idx,
            step,
            item.is_valid,
            item.violated_rule,
            item.validator_rules,
            item.validator_violation_count,
            split,
            item.input_kind,
            item.source,
            item.input_file,
        ])
        stats.combined_long_rows += 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_None._\n"

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    stats: BuildStats,
    counters: dict[str, Counter],
    output_files: dict[str, str],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []

    lines.append("# Task Dataset Build Report\n")

    lines.append("## Summary\n")
    lines.append(markdown_table(
        ["Metric", "Value"],
        [
            ["Valid sequences", stats.valid_sequences],
            ["Easy invalid sequences", stats.easy_invalid_sequences],
            ["Hard invalid sequences", stats.hard_invalid_sequences],
            ["Skipped duplicate sequences", stats.skipped_duplicate_sequences],
            ["Next-step examples", stats.next_step_examples],
            ["Completion examples", stats.completion_examples],
            ["Anomaly examples", stats.anomaly_examples],
            ["Rule-attribution examples", stats.rule_attribution_examples],
            ["Combined long rows", stats.combined_long_rows],
        ],
    ))

    lines.append("\n## Sequence Counts by Split\n")
    lines.append(markdown_table(
        ["Split", "Count"],
        [[k, v] for k, v in sorted(counters["split_counts"].items())],
    ))

    lines.append("\n## Sequence Counts by Family\n")
    lines.append(markdown_table(
        ["Family", "Count"],
        [[k, v] for k, v in sorted(counters["family_counts"].items())],
    ))

    lines.append("\n## Sequence Counts by Validity\n")
    lines.append(markdown_table(
        ["IS_VALID", "Count"],
        [[k, v] for k, v in sorted(counters["validity_counts"].items())],
    ))

    lines.append("\n## Invalid Counts by Rule\n")
    lines.append(markdown_table(
        ["Violated rule", "Count"],
        [[k, v] for k, v in sorted(counters["rule_counts"].items())],
    ))

    lines.append("\n## Counts by Input Kind\n")
    lines.append(markdown_table(
        ["Input kind", "Count"],
        [[k, v] for k, v in sorted(counters["input_kind_counts"].items())],
    ))

    lines.append("\n## Output Files\n")
    lines.append(markdown_table(
        ["Name", "Path"],
        [[k, v] for k, v in output_files.items()],
    ))

    lines.append("\n## Build Configuration\n")
    lines.append("```json\n")
    lines.append(json.dumps(vars(args), indent=2))
    lines.append("\n```\n")

    lines.append("\n## Notes\n")
    lines.append(
        "- `next_step_prediction.csv` uses only valid sequences.\n"
        "- `sequence_completion.csv` uses only valid sequences.\n"
        "- `anomaly_detection.csv` uses valid, easy-invalid, and hard-invalid sequences.\n"
        "- `rule_attribution.csv` uses only invalid sequences.\n"
        "- By default, the split is based on `origin_id`, so invalid mutations derived from a valid sequence stay in the same split as their source sequence.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    stats: BuildStats,
    counters: dict[str, Counter],
    output_files: dict[str, str],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created_at_unix": time.time(),
        "script": "build_task_datasets.py",
        "description": "Task dataset builder for semiconductor process-sequence learning.",
        "stats": asdict(stats),
        "counters": {
            key: dict(sorted(counter.items()))
            for key, counter in counters.items()
        },
        "output_files": output_files,
        "arguments": vars(args),
    }

    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def process_sequence_stream(
    path: Path,
    input_kind: str,
    max_sequences: Optional[int],
    seen_hashes: set[str],
    args: argparse.Namespace,
    writers: dict[str, Optional[csv.writer]],
    stats: BuildStats,
    counters: dict[str, Counter],
) -> None:
    processed = 0

    for item in iter_sequences_from_long_csv(path, input_kind=input_kind):
        if max_sequences is not None and processed >= max_sequences:
            break

        if args.deduplicate:
            h = sequence_hash(item.steps)
            if h in seen_hashes:
                stats.skipped_duplicate_sequences += 1
                continue
            seen_hashes.add(h)

        split = sequence_split(item, seed=args.seed, split_mode=args.split_mode)

        if input_kind == "valid":
            stats.valid_sequences += 1
        elif input_kind == "easy_invalid":
            stats.easy_invalid_sequences += 1
        elif input_kind == "hard_invalid":
            stats.hard_invalid_sequences += 1

        counters["split_counts"][split] += 1
        counters["family_counts"][item.family] += 1
        counters["validity_counts"][str(item.is_valid)] += 1
        counters["input_kind_counts"][input_kind] += 1

        if item.is_valid == 0 and item.violated_rule:
            counters["rule_counts"][item.violated_rule] += 1

        write_sequence_summary(writers["summary"], item, split)  # type: ignore[arg-type]
        write_anomaly_example(writers["anomaly"], item, split, stats)  # type: ignore[arg-type]
        write_rule_attribution_example(writers["rule"], item, split, stats)  # type: ignore[arg-type]

        if item.is_valid == 1:
            write_next_step_examples(
                writer=writers["next_step"],  # type: ignore[arg-type]
                item=item,
                split=split,
                context_window=args.context_window,
                stride=args.next_step_stride,
                min_prefix_len=args.min_prefix_len,
                stats=stats,
            )

            write_completion_examples(
                writer=writers["completion"],  # type: ignore[arg-type]
                item=item,
                split=split,
                cut_fracs=args.completion_cut_fracs,
                context_window=args.context_window,
                target_window=args.completion_target_window,
                stats=stats,
            )

        if writers.get("combined_long") is not None:
            write_combined_long(writers["combined_long"], item, split, stats)  # type: ignore[arg-type]

        processed += 1

        if args.progress_every > 0 and processed % args.progress_every == 0:
            print(f"  processed {processed:,} sequences from {path}")


def build_datasets(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_valid = [Path(p) for p in args.valid_input]
    paths_easy = [Path(p) for p in args.easy_invalid_input]
    paths_hard = [Path(p) for p in args.hard_invalid_input]

    stats = BuildStats()
    counters: dict[str, Counter] = {
        "split_counts": Counter(),
        "family_counts": Counter(),
        "validity_counts": Counter(),
        "rule_counts": Counter(),
        "input_kind_counts": Counter(),
    }

    seen_hashes: set[str] = set()

    output_files = {
        "next_step_prediction": str(output_dir / "next_step_prediction.csv"),
        "sequence_completion": str(output_dir / "sequence_completion.csv"),
        "anomaly_detection": str(output_dir / "anomaly_detection.csv"),
        "rule_attribution": str(output_dir / "rule_attribution.csv"),
        "sequence_summary": str(output_dir / "sequence_summary.csv"),
        "manifest": str(output_dir / "task_dataset_manifest.json"),
        "report": str(output_dir / "task_dataset_report.md"),
    }

    if args.write_combined_long:
        output_files["combined_long_sequences"] = str(output_dir / "combined_long_sequences.csv")

    with (
        open(output_files["next_step_prediction"], "w", newline="", encoding="utf-8") as f_next,
        open(output_files["sequence_completion"], "w", newline="", encoding="utf-8") as f_completion,
        open(output_files["anomaly_detection"], "w", newline="", encoding="utf-8") as f_anomaly,
        open(output_files["rule_attribution"], "w", newline="", encoding="utf-8") as f_rule,
        open(output_files["sequence_summary"], "w", newline="", encoding="utf-8") as f_summary,
    ):
        writers: dict[str, Optional[csv.writer]] = {
            "next_step": csv.writer(f_next),
            "completion": csv.writer(f_completion),
            "anomaly": csv.writer(f_anomaly),
            "rule": csv.writer(f_rule),
            "summary": csv.writer(f_summary),
            "combined_long": None,
        }

        writers["next_step"].writerow([
            "EXAMPLE_ID",
            "SEQUENCE_ID",
            "ORIGIN_ID",
            "FAMILY",
            "SPLIT",
            "TARGET_INDEX",
            "PREFIX_LENGTH",
            "CONTEXT_WINDOW",
            "PREFIX_CONTEXT",
            "NEXT_STEP",
            "SOURCE",
        ])

        writers["completion"].writerow([
            "EXAMPLE_ID",
            "SEQUENCE_ID",
            "ORIGIN_ID",
            "FAMILY",
            "SPLIT",
            "CUT_FRACTION",
            "CUT_INDEX",
            "SEQUENCE_LENGTH",
            "CONTEXT_WINDOW",
            "TARGET_WINDOW",
            "PREFIX_CONTEXT",
            "TARGET_SUFFIX",
            "TARGET_SUFFIX_TRUNCATED",
            "SOURCE",
        ])

        writers["anomaly"].writerow([
            "EXAMPLE_ID",
            "SEQUENCE_ID",
            "ORIGIN_ID",
            "FAMILY",
            "SPLIT",
            "SEQUENCE_LENGTH",
            "SEQUENCE",
            "IS_VALID",
            "VIOLATED_RULE",
            "VALIDATOR_RULES",
            "VALIDATOR_VIOLATION_COUNT",
            "INPUT_KIND",
            "SOURCE",
        ])

        writers["rule"].writerow([
            "EXAMPLE_ID",
            "SEQUENCE_ID",
            "ORIGIN_ID",
            "FAMILY",
            "SPLIT",
            "SEQUENCE_LENGTH",
            "SEQUENCE",
            "VIOLATED_RULE",
            "VALIDATOR_RULES",
            "VALIDATOR_VIOLATION_COUNT",
            "INPUT_KIND",
            "SOURCE",
        ])

        writers["summary"].writerow([
            "SEQUENCE_ID",
            "ORIGIN_ID",
            "FAMILY",
            "SPLIT",
            "SEQUENCE_LENGTH",
            "IS_VALID",
            "VIOLATED_RULE",
            "VALIDATOR_RULES",
            "VALIDATOR_VIOLATION_COUNT",
            "INPUT_KIND",
            "SOURCE",
            "INPUT_FILE",
        ])

        if args.write_combined_long:
            f_combined = open(output_files["combined_long_sequences"], "w", newline="", encoding="utf-8")
            try:
                writers["combined_long"] = csv.writer(f_combined)
                writers["combined_long"].writerow([
                    "SEQUENCE_ID",
                    "ORIGIN_ID",
                    "FAMILY",
                    "STEP_INDEX",
                    "STEP",
                    "IS_VALID",
                    "VIOLATED_RULE",
                    "VALIDATOR_RULES",
                    "VALIDATOR_VIOLATION_COUNT",
                    "SPLIT",
                    "INPUT_KIND",
                    "SOURCE",
                    "INPUT_FILE",
                ])

                _process_all_inputs(
                    paths_valid, paths_easy, paths_hard, args, writers, stats, counters, seen_hashes
                )
            finally:
                f_combined.close()
        else:
            _process_all_inputs(
                paths_valid, paths_easy, paths_hard, args, writers, stats, counters, seen_hashes
            )

    write_manifest(
        path=Path(output_files["manifest"]),
        stats=stats,
        counters=counters,
        output_files=output_files,
        args=args,
    )

    write_report(
        path=Path(output_files["report"]),
        stats=stats,
        counters=counters,
        output_files=output_files,
        args=args,
    )

    print("\nDone.")
    print("\nGenerated task files:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")

    print("\nSummary:")
    print(f"  valid sequences:              {stats.valid_sequences:,}")
    print(f"  easy invalid sequences:       {stats.easy_invalid_sequences:,}")
    print(f"  hard invalid sequences:       {stats.hard_invalid_sequences:,}")
    print(f"  skipped duplicates:           {stats.skipped_duplicate_sequences:,}")
    print(f"  next-step examples:           {stats.next_step_examples:,}")
    print(f"  completion examples:          {stats.completion_examples:,}")
    print(f"  anomaly examples:             {stats.anomaly_examples:,}")
    print(f"  rule-attribution examples:    {stats.rule_attribution_examples:,}")

    print("\nNext step:")
    print("  Inspect task_datasets_v1/task_dataset_report.md.")
    print("  Then train a baseline next-step / anomaly model.")


def _process_all_inputs(
    paths_valid: list[Path],
    paths_easy: list[Path],
    paths_hard: list[Path],
    args: argparse.Namespace,
    writers: dict[str, Optional[csv.writer]],
    stats: BuildStats,
    counters: dict[str, Counter],
    seen_hashes: set[str],
) -> None:
    for path in paths_valid:
        print(f"\nProcessing valid input: {path}")
        process_sequence_stream(
            path=path,
            input_kind="valid",
            max_sequences=args.max_valid_sequences,
            seen_hashes=seen_hashes,
            args=args,
            writers=writers,
            stats=stats,
            counters=counters,
        )

    for path in paths_easy:
        print(f"\nProcessing easy invalid input: {path}")
        process_sequence_stream(
            path=path,
            input_kind="easy_invalid",
            max_sequences=args.max_easy_invalid_sequences,
            seen_hashes=seen_hashes,
            args=args,
            writers=writers,
            stats=stats,
            counters=counters,
        )

    for path in paths_hard:
        print(f"\nProcessing hard invalid input: {path}")
        process_sequence_stream(
            path=path,
            input_kind="hard_invalid",
            max_sequences=args.max_hard_invalid_sequences,
            seen_hashes=seen_hashes,
            args=args,
            writers=writers,
            stats=stats,
            counters=counters,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_task_datasets.py",
        description="Build model-ready datasets from valid, easy-invalid, and hard-invalid semiconductor sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--valid-input",
        nargs="+",
        required=True,
        help="Valid long-format sequence CSVs.",
    )

    parser.add_argument(
        "--easy-invalid-input",
        nargs="*",
        default=[],
        help="Easy invalid long-format sequence CSVs.",
    )

    parser.add_argument(
        "--hard-invalid-input",
        nargs="*",
        default=[],
        help="Hard invalid long-format sequence CSVs.",
    )

    parser.add_argument(
        "--output-dir",
        default="task_datasets_v1",
        help="Output directory.",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=64,
        help=(
            "Number of previous steps used as model input context. "
            "Use 0 for full prefix, but that can create very large CSV files."
        ),
    )

    parser.add_argument(
        "--next-step-stride",
        type=int,
        default=1,
        help="Create one next-step example every N target positions.",
    )

    parser.add_argument(
        "--min-prefix-len",
        type=int,
        default=1,
        help="Minimum prefix length for next-step prediction examples.",
    )

    parser.add_argument(
        "--completion-cut-fracs",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="Cut fractions for sequence-completion examples.",
    )

    parser.add_argument(
        "--completion-target-window",
        type=int,
        default=96,
        help=(
            "Maximum number of suffix steps in completion target. "
            "Use 0 for full suffix."
        ),
    )

    parser.add_argument(
        "--split-mode",
        choices=["origin", "sequence", "input"],
        default="origin",
        help=(
            "How to assign train/val/test splits. "
            "'origin' keeps invalid mutations with their source sequence."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic splitting.",
    )

    parser.add_argument(
        "--max-valid-sequences",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )

    parser.add_argument(
        "--max-easy-invalid-sequences",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )

    parser.add_argument(
        "--max-hard-invalid-sequences",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Skip duplicate exact sequences across all inputs.",
    )

    parser.add_argument(
        "--no-deduplicate",
        action="store_false",
        dest="deduplicate",
        help="Do not skip duplicate exact sequences.",
    )

    parser.add_argument(
        "--write-combined-long",
        action="store_true",
        help="Also write combined_long_sequences.csv. This can be large.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N sequences per input file. Use 0 to disable.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_datasets(args)


if __name__ == "__main__":
    main()