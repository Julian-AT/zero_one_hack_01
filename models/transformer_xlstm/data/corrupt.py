"""Inject deliberate process-rule violations into valid sequences.

Used during training to produce labeled negatives for the multi-task heads
(validity + rule-ID). Each corrupter targets one of the 10 forbidden patterns
from generation_rules.md §3.

The validator is then run on the corrupted sequence to confirm the injected
violation is detected (and to get the exact `step_index` for PRM labelling
later).
"""

from __future__ import annotations

import random

# Reuse the organizer's vocabulary sets via importing private symbols from the
# adapter location.
import sys
from dataclasses import dataclass

from transformer_xlstm.data.validator import validate_sequence
from transformer_xlstm.utils.paths import RAW_DATA_DIR

if str(RAW_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(RAW_DATA_DIR))
from generate_sequences import (  # noqa: E402
    BACKSIDE_METAL_STEPS,
    CLEAN_STEPS,
    CMP_STEPS,
    DEPOSITION_STEPS,
    ELECTRICAL_TEST_STEPS,
    ETCH_STEPS,
    IMPLANT_STEPS,
    METAL_ETCH_STEPS,
    PAD_WINDOW_STEPS,
)


@dataclass
class Corruption:
    """A corruption applied to a sequence, with metadata for labelling."""

    rule: str  # one of RULE_IDS
    corrupted_steps: list[str]
    inject_index: int  # position where the violation was introduced
    description: str  # human-readable trace


def _find_indices(steps: list[str], targets) -> list[int]:
    """Return indices of steps that are in `targets`."""
    return [i for i, s in enumerate(steps) if s in targets]


def corrupt_dep_no_clean(steps: list[str], rng: random.Random) -> Corruption | None:
    """Remove the clean step preceding a deposition so RULE_DEP_NO_CLEAN fires."""
    dep_indices = _find_indices(steps, DEPOSITION_STEPS)
    rng.shuffle(dep_indices)
    for di in dep_indices:
        # Find the nearest clean step in the previous 12 steps and remove it.
        for j in range(di - 1, max(-1, di - 13), -1):
            if steps[j] in CLEAN_STEPS:
                new = steps[:j] + steps[j + 1 :]
                return Corruption(
                    "RULE_DEP_NO_CLEAN",
                    new,
                    j,
                    f"removed clean step at index {j} ({steps[j]!r}) "
                    f"before deposition at index {di} ({steps[di]!r})",
                )
    return None


def corrupt_metal_etch_no_litho(steps: list[str], rng: random.Random) -> Corruption | None:
    """Remove the EXPOSE LITHO step preceding a metal etch."""
    me_indices = _find_indices(steps, METAL_ETCH_STEPS)
    rng.shuffle(me_indices)
    for mi in me_indices:
        for j in range(mi - 1, max(-1, mi - 16), -1):
            if steps[j].startswith("EXPOSE LITHO LEVEL"):
                new = steps[:j] + steps[j + 1 :]
                return Corruption(
                    "RULE_METAL_ETCH_NO_LITHO",
                    new,
                    j,
                    f"removed EXPOSE LITHO at index {j} before metal etch at index {mi}",
                )
    return None


def corrupt_etch_no_mask(steps: list[str], rng: random.Random) -> Corruption | None:
    """Remove DEVELOP PHOTORESIST preceding an etch."""
    et_indices = _find_indices(steps, ETCH_STEPS - {"ANISOTROPIC ETCH SPACER"})
    rng.shuffle(et_indices)
    for ei in et_indices:
        for j in range(ei - 1, max(-1, ei - 13), -1):
            if steps[j] in ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"):
                new = steps[:j] + steps[j + 1 :]
                return Corruption(
                    "RULE_ETCH_NO_MASK",
                    new,
                    j,
                    f"removed DEVELOP at index {j} before etch at index {ei}",
                )
    return None


def corrupt_litho_level_skip(steps: list[str], rng: random.Random) -> Corruption | None:
    """Drop ALIGN MASK LEVEL N so subsequent ALIGN MASK LEVEL N+1 violates."""
    align_indices = [i for i, s in enumerate(steps) if s.startswith("ALIGN MASK LEVEL ")]
    if len(align_indices) < 2:
        return None
    # Drop a middle level (not the first or last) so the gap is detectable.
    pick = rng.choice(align_indices[:-1])
    new = steps[:pick] + steps[pick + 1 :]
    return Corruption(
        "RULE_LITHO_LEVEL_SKIP", new, pick, f"removed {steps[pick]!r} at index {pick}"
    )


def corrupt_implant_no_mask(steps: list[str], rng: random.Random) -> Corruption | None:
    """Remove an OXIDE ETCH / DEVELOP PHOTORESIST preceding an implant."""
    imp_indices = _find_indices(steps, IMPLANT_STEPS)
    rng.shuffle(imp_indices)
    openers = {
        "OXIDE ETCH",
        "OXIDE ETCH DRY",
        "ETCH SILICON OR OXIDE WINDOW",
        "DEVELOP PHOTORESIST",
    }
    for ii in imp_indices:
        for j in range(ii - 1, max(-1, ii - 16), -1):
            if steps[j] in openers:
                new = steps[:j] + steps[j + 1 :]
                return Corruption(
                    "RULE_IMPLANT_NO_MASK",
                    new,
                    j,
                    f"removed opener at index {j} ({steps[j]!r}) before implant at index {ii}",
                )
    return None


def corrupt_cmp_no_dep(steps: list[str], rng: random.Random) -> Corruption | None:
    """Remove deposition/fill step preceding a CMP."""
    cmp_indices = _find_indices(steps, CMP_STEPS)
    rng.shuffle(cmp_indices)
    fill_set = DEPOSITION_STEPS | {"FILL VIA METAL", "FILL VIA TUNGSTEN"}
    for ci in cmp_indices:
        for j in range(ci - 1, max(-1, ci - 7), -1):
            if steps[j] in fill_set:
                new = steps[:j] + steps[j + 1 :]
                return Corruption(
                    "RULE_CMP_NO_DEP",
                    new,
                    j,
                    f"removed deposition/fill at index {j} ({steps[j]!r}) before CMP at index {ci}",
                )
    return None


def corrupt_pad_open_before_dep(steps: list[str], rng: random.Random) -> Corruption | None:
    """Move a pad-window step to before DEPOSIT PASSIVATION."""
    pad_indices = _find_indices(steps, PAD_WINDOW_STEPS)
    if not pad_indices:
        return None
    pi = pad_indices[0]
    pad_step = steps[pi]
    # Move it to position 5 (well before passivation).
    new = [s for i, s in enumerate(steps) if i != pi]
    insert_at = min(5, len(new))
    new.insert(insert_at, pad_step)
    return Corruption(
        "RULE_PAD_OPEN_BEFORE_DEP",
        new,
        insert_at,
        f"moved {pad_step!r} from index {pi} to {insert_at}",
    )


def corrupt_test_before_passivation(steps: list[str], rng: random.Random) -> Corruption | None:
    """Move an electrical test to before CURE PASSIVATION."""
    test_indices = _find_indices(steps, ELECTRICAL_TEST_STEPS)
    cure_idx = next((i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None)
    if cure_idx is None or not test_indices:
        return None
    ti = test_indices[0]
    test_step = steps[ti]
    if ti < cure_idx:
        return None
    new = [s for i, s in enumerate(steps) if i != ti]
    insert_at = max(0, cure_idx - 1)
    new.insert(insert_at, test_step)
    return Corruption(
        "RULE_TEST_BEFORE_PASSIVATION",
        new,
        insert_at,
        f"moved test {test_step!r} from index {ti} to {insert_at} "
        f"(before CURE PASSIVATION at {cure_idx})",
    )


def corrupt_ship_before_test(steps: list[str], rng: random.Random) -> Corruption | None:
    """Move SHIP LOT before WAFER SORT TEST."""
    ship_idx = next((i for i, s in enumerate(steps) if s == "SHIP LOT"), None)
    sort_idx = next((i for i, s in enumerate(steps) if s == "WAFER SORT TEST"), None)
    if ship_idx is None or sort_idx is None or ship_idx < sort_idx:
        return None
    new = [s for i, s in enumerate(steps) if i != ship_idx]
    new.insert(sort_idx - 1, "SHIP LOT")
    return Corruption(
        "RULE_SHIP_BEFORE_TEST",
        new,
        sort_idx - 1,
        f"moved SHIP LOT from {ship_idx} to before WAFER SORT TEST",
    )


def corrupt_backside_before_passivation(steps: list[str], rng: random.Random) -> Corruption | None:
    """Move DEPOSIT BACKSIDE METAL to before CURE PASSIVATION."""
    bm_indices = _find_indices(steps, BACKSIDE_METAL_STEPS)
    cure_idx = next((i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None)
    if cure_idx is None or not bm_indices:
        return None
    bi = bm_indices[0]
    if bi < cure_idx:
        return None
    new = [s for i, s in enumerate(steps) if i != bi]
    insert_at = max(0, cure_idx - 1)
    new.insert(insert_at, "DEPOSIT BACKSIDE METAL")
    return Corruption(
        "RULE_BACKSIDE_BEFORE_PASSIVATION",
        new,
        insert_at,
        f"moved DEPOSIT BACKSIDE METAL from {bi} to {insert_at}",
    )


CORRUPTERS = {
    "RULE_DEP_NO_CLEAN": corrupt_dep_no_clean,
    "RULE_METAL_ETCH_NO_LITHO": corrupt_metal_etch_no_litho,
    "RULE_ETCH_NO_MASK": corrupt_etch_no_mask,
    "RULE_LITHO_LEVEL_SKIP": corrupt_litho_level_skip,
    "RULE_IMPLANT_NO_MASK": corrupt_implant_no_mask,
    "RULE_CMP_NO_DEP": corrupt_cmp_no_dep,
    "RULE_PAD_OPEN_BEFORE_DEP": corrupt_pad_open_before_dep,
    "RULE_TEST_BEFORE_PASSIVATION": corrupt_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST": corrupt_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": corrupt_backside_before_passivation,
}


def corrupt_random(steps: list[str], rng: random.Random, verify: bool = True) -> Corruption | None:
    """Apply a random corrupter; return the result (or None if none applied).

    If `verify=True`, the validator is run on the result to confirm the
    targeted rule is actually triggered. Some sequences are structurally not
    susceptible to certain corrupters; those return None.
    """
    rule_order = list(CORRUPTERS.keys())
    rng.shuffle(rule_order)
    for rule in rule_order:
        c = CORRUPTERS[rule](steps, rng)
        if c is None:
            continue
        if verify:
            violations = validate_sequence(c.corrupted_steps)
            if not any(v.rule == rule for v in violations):
                # Corruption didn't trigger (e.g. a backup constraint also enforced it).
                continue
        return c
    return None


if __name__ == "__main__":
    # Smoke test: corrupt one sequence per rule and verify the validator catches it.
    from transformer_xlstm.data.validator import generate_sequence

    rng = random.Random(0)
    seed_seq = generate_sequence("mosfet", rng)
    print(
        f"Baseline sequence: {len(seed_seq)} steps, valid={len(validate_sequence(seed_seq)) == 0}"
    )
    print()
    for rule, corrupter in CORRUPTERS.items():
        rng_local = random.Random(0)
        c = corrupter(list(seed_seq), rng_local)
        if c is None:
            print(f"  [{rule}] not applicable to this sequence")
            continue
        violations = validate_sequence(c.corrupted_steps)
        rule_hits = [v.rule for v in violations]
        ok = rule in rule_hits
        print(f"  [{rule}] applied: ok={ok}  caught={set(rule_hits)}")
