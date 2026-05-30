"""Adapter that re-exports the organizers' validator from the track folder.

The organizers ship `generate_sequences.py` inside the track. We never modify
it. This adapter exposes `validate_sequence`, `generate_sequence`,
`generate_dataset`, and `read_csv_sequences` under a stable import path, plus
canonical rule indices.
"""

from __future__ import annotations

import sys

from transformer_xlstm.utils.paths import RAW_DATA_DIR

if str(RAW_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(RAW_DATA_DIR))

from generate_sequences import (  # noqa: E402  (path mutation above is intentional)
    Violation,
    generate_dataset,
    generate_sequence,
    read_csv_sequences,
    validate_sequence,
    write_csv,
)

# Canonical rule IDs — these strings are what `Violation.rule` returns.
RULE_IDS: list[str] = [
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
RULE_TO_IDX = {r: i for i, r in enumerate(RULE_IDS)}
VALID_CLASS_IDX = len(RULE_IDS)  # index reserved for "valid" in the rule-ID head
NUM_RULE_CLASSES = len(RULE_IDS) + 1  # 10 rules + valid


def is_valid(steps: list[str]) -> bool:
    return len(validate_sequence(steps)) == 0


def rule_class_index(steps: list[str]) -> int:
    """Return the rule-class index for the multi-task rule-ID head.

    If the sequence is valid → VALID_CLASS_IDX.
    Otherwise → the index of the *first* violation's rule (sequential reading
    order). Sequences with multiple violations report only the first; the head
    is therefore trained to find the earliest violation.
    """
    violations = validate_sequence(steps)
    if not violations:
        return VALID_CLASS_IDX
    return RULE_TO_IDX[violations[0].rule]


__all__ = [
    "Violation",
    "validate_sequence",
    "generate_sequence",
    "generate_dataset",
    "read_csv_sequences",
    "write_csv",
    "is_valid",
    "rule_class_index",
    "RULE_IDS",
    "RULE_TO_IDX",
    "VALID_CLASS_IDX",
    "NUM_RULE_CLASSES",
]
