"""Build official-format ground-truth + input CSVs from held-out full sequences.

The organizers ship *sample* eval inputs (``eval_input_valid.csv`` etc.) but
those carry **no answers** — there is no ``NEXT_STEP`` or ``FULL_SEQUENCE``
column — so Tasks 1 (next-step) and 2 (completion) cannot be scored against
them locally. This module reconstructs the missing ground-truth from our own
held-out *full* sequences (e.g. a LoFO held-out family), in the exact column
layout that the official ``eval_metrics.py`` consumes:

    next-step   GT : EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, NEXT_STEP
    completion  GT : EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, FULL_SEQUENCE
    model input    : EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
                     (identical schema to the official ``EVAL_INPUT_VALID``)

The input CSV is what ``nspe.predict`` reads; the GT CSVs are what
``nspe.eval`` feeds to the official scorer as ``--ground-truth``.

Cut convention (matches the sample inputs and the scorer's slicing):
    cut   = max(1, min(len-1, round(len * frac)))   # 1 <= cut <= len-1
    partial   = seq[:cut]
    next_step = seq[cut]          # always exists because cut <= len-1
    full      = seq               # the whole sequence

Sequences are pipe-joined (``"A|B|C"``). EXAMPLE_IDs are stable and unique:
``"<prefix>_<family>_<i>_<frac>"`` where *i* is the 0-based sequence index and
*frac* is the cut fraction with its literal text (e.g. ``"0.6"``).

stdlib only — no torch, no numpy.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

PathLike = Union[str, Path]

# Column schemas (kept as module constants so callers / tests can assert them).
INPUT_COLUMNS = ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"]
NEXTSTEP_GT_COLUMNS = [
    "EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE", "NEXT_STEP",
]
COMPLETION_GT_COLUMNS = [
    "EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE", "FULL_SEQUENCE",
]

SeqsByFamily = Mapping[str, Sequence[Sequence[str]]]


def _join(seq: Sequence[str]) -> str:
    """Pipe-join a step sequence the way the official scorer expects."""
    return "|".join(seq)


def _frac_text(frac: float) -> str:
    """Render a cut fraction the way it appears in EXAMPLE_IDs / CSV cells.

    Integers-as-floats (1.0) and trailing-zero floats are normalised so the
    text is stable: 0.60 -> "0.6", 1.0 -> "1.0" stays "1.0" only if passed so.
    We use ``repr``-free formatting: strip a single trailing zero group.
    """
    # ``str(0.6)`` -> "0.6", ``str(0.8)`` -> "0.8"; this is already stable in
    # Python 3 for the fractions we use. We avoid f-string precision so the
    # text matches what a user would write in a config.
    return str(frac)


def _cut_index(n: int, frac: float) -> int:
    """Cut index for a sequence of length *n* at fraction *frac*.

    Clamped to ``1 <= cut <= n-1`` so both the partial (>=1 step) and the
    remainder (>=1 step, hence a valid NEXT_STEP) are always non-empty.
    """
    return max(1, min(n - 1, round(n * frac)))


def make_nextstep_examples(
    seqs_by_family: SeqsByFamily,
    cut_fracs: Sequence[float] = (0.6, 0.8),
    seed: int = 0,
) -> list[dict]:
    """Build next-step ground-truth rows from held-out full sequences.

    One row per (sequence, cut fraction) for which a cut is possible
    (``len >= 2``). ``NEXT_STEP`` is the single step immediately after the cut.

    Returns a list of dicts with keys
    ``EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, NEXT_STEP``.

    *seed* is accepted for API symmetry / future sampling; row generation is
    fully deterministic (no randomness), so it does not affect the output.
    """
    del seed  # deterministic; reserved for future subsampling
    rows: list[dict] = []
    for family in sorted(seqs_by_family):
        for i, seq in enumerate(seqs_by_family[family]):
            seq = list(seq)
            n = len(seq)
            if n < 2:
                continue  # cannot form a (partial, next-step) pair
            for frac in cut_fracs:
                cut = _cut_index(n, frac)
                ftxt = _frac_text(frac)
                rows.append({
                    "EXAMPLE_ID": f"ns_{family}_{i}_{ftxt}",
                    "FAMILY": family,
                    "COMPLETION_FRACTION": ftxt,
                    "PARTIAL_SEQUENCE": _join(seq[:cut]),
                    "NEXT_STEP": seq[cut],
                })
    return rows


def make_completion_examples(
    seqs_by_family: SeqsByFamily,
    cut_fracs: Sequence[float] = (0.6, 0.8),
    seed: int = 0,
) -> list[dict]:
    """Build completion ground-truth rows from held-out full sequences.

    One row per (sequence, cut fraction) for which a cut is possible
    (``len >= 2``). The scorer derives the reference *remaining* steps as
    ``FULL_SEQUENCE[len(PARTIAL_SEQUENCE):]``.

    Returns a list of dicts with keys
    ``EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE, FULL_SEQUENCE``.
    """
    del seed  # deterministic; reserved for future subsampling
    rows: list[dict] = []
    for family in sorted(seqs_by_family):
        for i, seq in enumerate(seqs_by_family[family]):
            seq = list(seq)
            n = len(seq)
            if n < 2:
                continue
            for frac in cut_fracs:
                cut = _cut_index(n, frac)
                ftxt = _frac_text(frac)
                rows.append({
                    "EXAMPLE_ID": f"cp_{family}_{i}_{ftxt}",
                    "FAMILY": family,
                    "COMPLETION_FRACTION": ftxt,
                    "PARTIAL_SEQUENCE": _join(seq[:cut]),
                    "FULL_SEQUENCE": _join(seq),
                })
    return rows


def _write_csv(rows: Iterable[dict], path: PathLike, columns: Sequence[str]) -> Path:
    """Write *rows* to *path* with exactly *columns* (in order). Returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def write_input_csv(rows: Iterable[dict], path: PathLike) -> Path:
    """Write the model-input CSV (EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE).

    Schema is identical to the official ``EVAL_INPUT_VALID``. Accepts either
    next-step or completion GT rows (extra columns are dropped) so a single
    input file can be derived from either example set.
    """
    return _write_csv(rows, path, INPUT_COLUMNS)


def write_nextstep_gt(rows: Iterable[dict], path: PathLike) -> Path:
    """Write next-step ground-truth CSV (adds NEXT_STEP column)."""
    return _write_csv(rows, path, NEXTSTEP_GT_COLUMNS)


def write_completion_gt(rows: Iterable[dict], path: PathLike) -> Path:
    """Write completion ground-truth CSV (adds FULL_SEQUENCE column)."""
    return _write_csv(rows, path, COMPLETION_GT_COLUMNS)


def build_eval_set(
    seqs_by_family: SeqsByFamily,
    out_dir: PathLike,
    prefix: str = "sim",
    cut_fracs: Sequence[float] = (0.6, 0.8),
    seed: int = 0,
) -> dict:
    """Build a full local eval set: one model-input CSV + two GT CSVs.

    Writes three files under *out_dir*:
        <prefix>_input.csv         model input (predict.py reads this)
        <prefix>_nextstep_gt.csv   Task-1 ground truth (eval.py scores this)
        <prefix>_completion_gt.csv Task-2 ground truth (eval.py scores this)

    The next-step input and completion input share the same prefixes, so we
    emit the input file from the *completion* example set (which carries the
    superset of partials at each cut). EXAMPLE_IDs differ by task prefix
    (``ns_`` vs ``cp_``) so the input file uses completion IDs for Task 2; the
    Task-1 input is re-derived from next-step rows. To keep a single, canonical
    input file we write *both* an explicit next-step input and a completion
    input and return all paths.

    Returns a dict of written paths:
        {input, nextstep_gt, completion_gt, nextstep_input, completion_input}
    where ``input`` aliases ``nextstep_input`` for convenience.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ns_rows = make_nextstep_examples(seqs_by_family, cut_fracs=cut_fracs, seed=seed)
    cp_rows = make_completion_examples(seqs_by_family, cut_fracs=cut_fracs, seed=seed)

    nextstep_gt = write_nextstep_gt(ns_rows, out_dir / f"{prefix}_nextstep_gt.csv")
    completion_gt = write_completion_gt(cp_rows, out_dir / f"{prefix}_completion_gt.csv")
    nextstep_input = write_input_csv(ns_rows, out_dir / f"{prefix}_nextstep_input.csv")
    completion_input = write_input_csv(cp_rows, out_dir / f"{prefix}_completion_input.csv")

    return {
        "input": nextstep_input,
        "nextstep_gt": nextstep_gt,
        "completion_gt": completion_gt,
        "nextstep_input": nextstep_input,
        "completion_input": completion_input,
    }


__all__ = [
    "INPUT_COLUMNS", "NEXTSTEP_GT_COLUMNS", "COMPLETION_GT_COLUMNS",
    "make_nextstep_examples", "make_completion_examples",
    "write_input_csv", "write_nextstep_gt", "write_completion_gt",
    "build_eval_set",
]


if __name__ == "__main__":
    # Self-test: build an eval set from 5 mosfet sequences into a temp dir,
    # print row counts + one sample row, read the GT back and confirm the
    # columns match the official eval spec.
    import tempfile

    from nspe.data import load_family

    seqs = list(load_family("mosfet"))[:5]
    seqs_by_family = {"mosfet": seqs}

    with tempfile.TemporaryDirectory() as tmp:
        paths = build_eval_set(seqs_by_family, tmp, prefix="sim")

        ns_rows = make_nextstep_examples(seqs_by_family)
        cp_rows = make_completion_examples(seqs_by_family)
        print(f"5 mosfet seqs, lengths = {[len(s) for s in seqs]}")
        print(f"next-step rows   : {len(ns_rows)} (expect 5 seqs x 2 fracs = 10)")
        print(f"completion rows  : {len(cp_rows)} (expect 10)")
        print(f"files written    : {[p.name for p in paths.values()]}")

        sample = ns_rows[0]
        print("\nsample next-step row:")
        for k in NEXTSTEP_GT_COLUMNS:
            v = sample[k]
            shown = v if len(str(v)) < 70 else str(v)[:67] + "..."
            print(f"  {k:<20s}: {shown}")

        # Verify the cut/next-step invariant against the raw sequence.
        seq0 = list(seqs[0])
        cut0 = _cut_index(len(seq0), 0.6)
        assert sample["PARTIAL_SEQUENCE"] == "|".join(seq0[:cut0])
        assert sample["NEXT_STEP"] == seq0[cut0]
        assert sample["EXAMPLE_ID"] == "ns_mosfet_0_0.6"

        # Read the GT files back and confirm columns match the OFFICIAL spec.
        with open(paths["nextstep_gt"], newline="", encoding="utf-8") as f:
            ns_back = list(csv.DictReader(f))
        with open(paths["completion_gt"], newline="", encoding="utf-8") as f:
            cp_back = list(csv.DictReader(f))
        with open(paths["input"], newline="", encoding="utf-8") as f:
            in_back = csv.DictReader(f)
            in_cols = list(in_back.fieldnames or [])

        ns_cols = list(ns_back[0].keys())
        cp_cols = list(cp_back[0].keys())
        print("\ncolumn checks (must match official eval_metrics.py):")
        print(f"  next-step GT  cols = {ns_cols}")
        print(f"  completion GT cols = {cp_cols}")
        print(f"  input         cols = {in_cols}")

        assert ns_cols == NEXTSTEP_GT_COLUMNS, ns_cols
        assert cp_cols == COMPLETION_GT_COLUMNS, cp_cols
        assert in_cols == INPUT_COLUMNS, in_cols
        # Round-trip a value to confirm pipe-join survives CSV.
        assert ns_back[0]["NEXT_STEP"] == sample["NEXT_STEP"]
        # Unique EXAMPLE_IDs.
        ids = [r["EXAMPLE_ID"] for r in ns_back]
        assert len(ids) == len(set(ids)), "EXAMPLE_IDs not unique"

        print("\nALL SELF-TEST ASSERTIONS PASSED")
