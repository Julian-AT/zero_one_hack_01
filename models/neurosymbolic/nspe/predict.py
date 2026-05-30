"""Turn official eval-input CSVs into the three submission CSVs.

This module is the glue from a (duck-typed) ranker to the organizers' exact
submission formats:

  * Task 1 — next-step : ``EXAMPLE_ID,RANK_1,RANK_2,RANK_3,RANK_4,RANK_5``
  * Task 2 — completion: ``EXAMPLE_ID,PREDICTED_SEQUENCE`` (pipe-joined, ONLY the
    steps AFTER the cut — the partial is never repeated).
  * Task 3 — anomaly   : ``EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE`` (delegated
    to the pure-symbolic ``nspe.anomaly.classify_file`` oracle).

Tasks 1 & 2 read the official valid-eval schema
``EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE`` (columns matched
case-insensitively; extra columns ignored). FAMILY is lower-cased before being
handed to the ranker. Sequences are pipe-joined.

Symbolic core — no torch at module top. ``predict.py`` may *lazily* import
``nspe.model`` only if a caller passes a neural checkpoint path, but the ranker
itself is always duck-typed here, so this module never imports torch.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from nspe import anomaly, decode

PathLike = Union[str, Path]

__all__ = [
    "predict_nextstep",
    "predict_completion",
    "predict_anomaly",
    "make_all_submissions",
]

_RANK_COLUMNS = ["RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _split(raw: Optional[str]) -> List[str]:
    """Split a pipe-joined cell into trimmed, non-empty step strings."""
    if not raw:
        return []
    return [s for s in (p.strip() for p in raw.split("|")) if s]


def _read_input_rows(input_csv: Path):
    """Yield (example_id, family, partial_steps) from a valid-eval-format CSV.

    Column names are matched case-insensitively; FAMILY is lower-cased. Accepts
    ``PARTIAL_SEQUENCE`` (official) or ``SEQUENCE`` as the partial source.
    """
    with input_csv.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fmap = {(name or "").strip().upper(): name for name in (reader.fieldnames or [])}
        id_key = fmap.get("EXAMPLE_ID")
        fam_key = fmap.get("FAMILY")
        seq_key = fmap.get("PARTIAL_SEQUENCE") or fmap.get("SEQUENCE")
        if id_key is None or seq_key is None:
            raise ValueError(
                f"{input_csv} must have EXAMPLE_ID and PARTIAL_SEQUENCE columns; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            eid = (row.get(id_key) or "").strip()
            fam = ((row.get(fam_key) or "").strip().lower()) if fam_key else ""
            partial = _split(row.get(seq_key))
            yield eid, fam, partial


def _write_csv(rows: Iterable[dict], path: Path, columns: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
            n += 1
    return n


# ---------------------------------------------------------------------------
# Task 1 — next-step
# ---------------------------------------------------------------------------

def predict_nextstep(
    input_csv: PathLike,
    out_csv: PathLike,
    ranker,
    candidate_vocab: Iterable[str],
    use_roles: bool = False,
    role_sharpen: bool = True,
) -> int:
    """Write the Task-1 submission (top-5 next steps) for every input row.

    Returns the number of rows written.
    """
    input_csv, out_csv = Path(input_csv), Path(out_csv)
    candidate_vocab = list(candidate_vocab)
    rows_out: List[dict] = []
    for eid, fam, partial in _read_input_rows(input_csv):
        preds = decode.next_step_topk(
            partial, fam, ranker, candidate_vocab, k=5,
            use_roles=use_roles, role_sharpen=role_sharpen,
        )
        # decode returns up to 5 distinct steps (exactly 5 with the full vocab).
        row = {"EXAMPLE_ID": eid}
        for col, step in zip(_RANK_COLUMNS, preds):
            row[col] = step
        rows_out.append(row)
    return _write_csv(rows_out, out_csv, ["EXAMPLE_ID", *_RANK_COLUMNS])


# ---------------------------------------------------------------------------
# Task 2 — completion
# ---------------------------------------------------------------------------

def predict_completion(
    input_csv: PathLike,
    out_csv: PathLike,
    ranker,
    candidate_vocab: Iterable[str],
    use_roles: bool = False,
    beam: int = 1,
    max_len: int = 220,
) -> int:
    """Write the Task-2 submission (predicted remaining steps) for every row.

    ``PREDICTED_SEQUENCE`` is pipe-joined and contains ONLY the steps AFTER the
    cut (``decode.complete`` returns the suffix). Returns rows written.
    """
    input_csv, out_csv = Path(input_csv), Path(out_csv)
    candidate_vocab = list(candidate_vocab)
    rows_out: List[dict] = []
    for eid, fam, partial in _read_input_rows(input_csv):
        suffix = decode.complete(
            partial, fam, ranker, candidate_vocab,
            max_len=max_len, use_roles=use_roles, beam=beam,
        )
        rows_out.append({
            "EXAMPLE_ID": eid,
            "PREDICTED_SEQUENCE": "|".join(suffix),
        })
    return _write_csv(rows_out, out_csv, ["EXAMPLE_ID", "PREDICTED_SEQUENCE"])


# ---------------------------------------------------------------------------
# Task 3 — anomaly (pure symbolic oracle)
# ---------------------------------------------------------------------------

def predict_anomaly(
    input_csv: PathLike,
    out_csv: PathLike,
    ranker=None,
    use_roles: bool = True,
    ppl_threshold: Optional[float] = None,
) -> int:
    """Write the Task-3 submission by delegating to the symbolic oracle.

    The default (``ranker=None``) is the pure-symbolic, role-augmented classifier
    (0 FP / ~100% recall on the unseen family per the OOD probe). An optional
    learned residual is available via ``ranker`` + ``ppl_threshold`` (kept OFF for
    the headline submission). Returns rows written.
    """
    return anomaly.classify_file(
        input_csv, out_csv, ranker=ranker,
        ppl_threshold=ppl_threshold, use_roles=use_roles,
    )


# ---------------------------------------------------------------------------
# All three at once
# ---------------------------------------------------------------------------

def make_all_submissions(
    valid_input: PathLike,
    anomaly_input: PathLike,
    ranker,
    out_dir: PathLike,
    candidate_vocab: Optional[Iterable[str]] = None,
    use_roles: bool = False,
    beam: int = 1,
) -> dict:
    """Produce all three submission CSVs and return their paths.

    Task 1 & 2 are generated from ``valid_input`` (the official valid-eval CSV);
    Task 3 from ``anomaly_input``. If ``candidate_vocab`` is omitted, the full
    three-family training vocabulary is used (``nspe.data.candidate_vocab``).

    Returns ``{"task1": path1, "task2": path2, "task3": path3}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if candidate_vocab is None:
        from nspe.data import candidate_vocab as _cv  # lazy: keep import surface small
        candidate_vocab = _cv()
    candidate_vocab = list(candidate_vocab)

    p1 = out_dir / "submission_task1.csv"
    p2 = out_dir / "submission_task2.csv"
    p3 = out_dir / "submission_task3.csv"

    predict_nextstep(valid_input, p1, ranker, candidate_vocab, use_roles=use_roles)
    predict_completion(valid_input, p2, ranker, candidate_vocab,
                       use_roles=use_roles, beam=beam)
    predict_anomaly(anomaly_input, p3, ranker=None, use_roles=True)

    return {"task1": p1, "task2": p2, "task3": p3}


# ---------------------------------------------------------------------------
# Self-test (PPM ranker): build a tiny eval set, run nextstep + completion,
# assert every completion is rule-valid, and write a Task-3 CSV from a slice of
# the official anomaly input.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    from nspe import rules, simulate_eval
    from nspe.data import candidate_vocab as cand_vocab
    from nspe.data import load_family
    from nspe.official import EVAL_INPUT_ANOMALY
    from nspe.ppm import PPM

    out_dir = Path(os.environ.get(
        "NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("predict.py self-test  (PPM ranker)")
    print("=" * 64)

    # Fit PPM on mosfet+igbt (held out ic).
    train = {
        "mosfet": [list(s) for s in load_family("mosfet")[:200]],
        "igbt": [list(s) for s in load_family("igbt")[:200]],
    }
    ranker = PPM().fit(train)
    cand = cand_vocab(("mosfet", "igbt"))

    # Build a tiny OOD eval set from 3 ic sequences.
    ic = {"ic": [list(s) for s in load_family("ic")[:3]]}
    eval_set = simulate_eval.build_eval_set(ic, out_dir, prefix="predict_selftest")
    ns_input = eval_set["nextstep_input"]
    cp_input = eval_set["completion_input"]

    # ---- Task 1 ----
    t1 = out_dir / "predict_selftest_task1.csv"
    n1 = predict_nextstep(ns_input, t1, ranker, cand)
    with t1.open(encoding="utf-8") as fh:
        head1 = fh.read().splitlines()[:3]
    print(f"\n[Task 1] wrote {n1} rows -> {t1.name}")
    for line in head1:
        print("   ", (line[:100] + "...") if len(line) > 100 else line)
    # Every row has exactly 5 ranks.
    with t1.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ranks = [r[c] for c in _RANK_COLUMNS]
            assert all(ranks) and len(set(ranks)) == 5, f"bad ranks for {r['EXAMPLE_ID']}"
    print("   Task-1: every row has 5 distinct non-empty ranks  OK")

    # ---- Task 2 ----
    t2 = out_dir / "predict_selftest_task2.csv"
    n2 = predict_completion(cp_input, t2, ranker, cand, beam=1)
    print(f"\n[Task 2] wrote {n2} rows -> {t2.name}")

    # Assert: every completion is rule-valid AND the suffix is not the partial.
    # Reconstruct full = partial + predicted-suffix and run validate_with_roles.
    cp_partials = {}
    for eid, fam, partial in _read_input_rows(cp_input):
        cp_partials[eid] = partial
    n_valid = 0
    with t2.open(newline="", encoding="utf-8") as fh:
        comp_rows = list(csv.DictReader(fh))
    for r in comp_rows:
        eid = r["EXAMPLE_ID"]
        suffix = _split(r["PREDICTED_SEQUENCE"])
        partial = cp_partials[eid]
        full = partial + suffix
        viols = rules.validate_with_roles(full)
        ok = len(viols) == 0
        n_valid += ok
        status = "OK" if ok else f"INVALID({viols[0].rule})"
        print(f"   {eid:24s} suffix_len={len(suffix):3d} full_len={len(full):3d}  {status}")
    assert n_valid == len(comp_rows), \
        f"expected all {len(comp_rows)} completions valid, got {n_valid}"
    print("   Task-2: every completion passes validate_with_roles  OK")

    # ---- Task 3 ----
    slice_csv = out_dir / "predict_selftest_anomaly_slice.csv"
    with EVAL_INPUT_ANOMALY.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [next(reader) for _ in range(8)]
    with slice_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    t3 = out_dir / "predict_selftest_task3.csv"
    n3 = predict_anomaly(slice_csv, t3)
    print(f"\n[Task 3] wrote {n3} rows -> {t3.name}")
    with t3.open(encoding="utf-8") as fh:
        for line in fh.read().splitlines()[:4]:
            print("   ", line)

    # ---- make_all_submissions wiring (uses the slice for anomaly) ----
    subs = make_all_submissions(ns_input, slice_csv, ranker, out_dir / "subs_selftest",
                                candidate_vocab=cand)
    print("\n[make_all_submissions] ->")
    for k, v in subs.items():
        assert Path(v).exists(), f"{k} not written"
        print(f"   {k}: {Path(v).name}  (exists)")

    print("\nSELF-TEST PASSED")
