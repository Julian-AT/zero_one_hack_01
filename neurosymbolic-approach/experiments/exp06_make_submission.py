#!/usr/bin/env python3
"""exp06 — produce the three official submission CSVs.

This is the final, end-to-end submission step of NSPE. It reads the organizers'
sample eval inputs (``EVAL_INPUT_VALID`` for Tasks 1 & 2, ``EVAL_INPUT_ANOMALY``
for Task 3) and emits the three submission CSVs in the exact official formats:

    submission_task1.csv  ->  EXAMPLE_ID,RANK_1,RANK_2,RANK_3,RANK_4,RANK_5
    submission_task2.csv  ->  EXAMPLE_ID,PREDICTED_SEQUENCE   (suffix only, pipe-joined)
    submission_task3.csv  ->  EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE

Ranker selection
----------------
* ``--ranker ppm`` (default): fit the pure-symbolic, role-factored PPM ranker on
  all three training families. This runs anywhere — no GPU, no torch — so it is
  the safe default for producing a submission.
* ``--ranker neural``: lazily import :mod:`nspe.model` and reload a trained
  ``NeuralRanker`` checkpoint via ``--ckpt``. torch is imported only on this
  path, keeping the symbolic default torch-free.

Both rankers satisfy the duck-typed RANKER PROTOCOL, so the downstream submission
glue (``nspe.predict.make_all_submissions``) is identical for either.

Task 3 (anomaly) is always the pure-symbolic role-augmented oracle
(``ranker=None`` inside ``make_all_submissions``) — that is the headline NSPE
result (0 false positives / ~100% recall on the unseen family).

After writing, every Task-2 completion is re-validated by reconstructing
``full = partial + predicted_suffix`` and running
:func:`nspe.rules.validate_with_roles`; the count of rule-valid completions is
reported. A constrained decoder should make this 100%.

Run
---
    PYTHONPATH=neurosymbolic-approach python3 \
        neurosymbolic-approach/experiments/exp06_make_submission.py [flags]

Flags: ``--ranker {ppm,neural}`` (default ppm), ``--ckpt PATH`` (neural only),
``--use-roles`` (role-sharpen the candidate support), ``--out DIR`` (override the
output directory; otherwise ``$NSPE_OUT`` or ``neurosymbolic-approach/outputs``).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

from nspe import predict, rules
from nspe.data import candidate_vocab, load_family
from nspe.official import EVAL_INPUT_ANOMALY, EVAL_INPUT_VALID, FAMILIES

__all__ = [
    "build_ranker",
    "make_submission",
    "validate_task2",
    "main",
]


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def _default_out_dir() -> Path:
    """Output dir: env ``NSPE_OUT`` else ``neurosymbolic-approach/outputs``."""
    env = os.environ.get("NSPE_OUT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "outputs"


# ---------------------------------------------------------------------------
# Ranker construction
# ---------------------------------------------------------------------------

def build_ranker(kind: str = "ppm", ckpt: Optional[str] = None):
    """Construct a ranker satisfying the RANKER PROTOCOL.

    ``kind == "ppm"``  -> fit a PPM on all three training families (CPU only).
    ``kind == "neural"`` -> lazily load a ``NeuralRanker`` checkpoint (torch).
    """
    kind = kind.lower()
    if kind == "ppm":
        from nspe.ppm import PPM  # local: keeps top-level torch-free anyway
        train = {f: [list(s) for s in load_family(f)] for f in FAMILIES}
        return PPM().fit(train)
    if kind == "neural":
        if not ckpt:
            raise ValueError("--ranker neural requires --ckpt PATH")
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        # Lazy torch import lives entirely inside nspe.model.load_ranker.
        from nspe.model import load_ranker
        return load_ranker(ckpt)
    raise ValueError(f"unknown ranker kind: {kind!r} (expected 'ppm' or 'neural')")


# ---------------------------------------------------------------------------
# Task-2 validation
# ---------------------------------------------------------------------------

def _split(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [s for s in (p.strip() for p in raw.split("|")) if s]


def validate_task2(valid_input: Path, task2_csv: Path) -> dict:
    """Re-validate every Task-2 completion with ``validate_with_roles``.

    Reconstructs ``full = partial + predicted_suffix`` per EXAMPLE_ID and runs the
    role-augmented validator. Returns a summary dict with the count of valid
    completions, the total, and the first few offending (eid, rule) pairs.
    """
    # Map EXAMPLE_ID -> partial steps from the input CSV.
    partials: dict = {}
    with valid_input.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fmap = {(c or "").strip().upper(): c for c in (reader.fieldnames or [])}
        id_key = fmap.get("EXAMPLE_ID")
        seq_key = fmap.get("PARTIAL_SEQUENCE") or fmap.get("SEQUENCE")
        for row in reader:
            eid = (row.get(id_key) or "").strip()
            partials[eid] = _split(row.get(seq_key))

    total = 0
    n_valid = 0
    offenders: List[tuple] = []
    with task2_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("EXAMPLE_ID") or "").strip()
            suffix = _split(row.get("PREDICTED_SEQUENCE"))
            full = partials.get(eid, []) + suffix
            viols = rules.validate_with_roles(full)
            total += 1
            if not viols:
                n_valid += 1
            elif len(offenders) < 5:
                first = min(viols, key=lambda v: v.step_index)
                offenders.append((eid, first.rule))
    return {"valid": n_valid, "total": total, "offenders": offenders}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_submission(
    ranker_kind: str = "ppm",
    ckpt: Optional[str] = None,
    use_roles: bool = False,
    out_dir: Optional[Path] = None,
    valid_input: Optional[Path] = None,
    anomaly_input: Optional[Path] = None,
) -> dict:
    """Build a ranker, write the three submission CSVs, and validate Task 2.

    Returns ``{"task1","task2","task3"}`` paths plus a ``"task2_validation"``
    summary dict.
    """
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_input = Path(valid_input) if valid_input is not None else EVAL_INPUT_VALID
    anomaly_input = Path(anomaly_input) if anomaly_input is not None else EVAL_INPUT_ANOMALY

    print("=" * 64)
    print("exp06 — make submission")
    print("=" * 64)
    print(f"ranker      : {ranker_kind}" + (f"  (ckpt={ckpt})" if ckpt else ""))
    print(f"use_roles   : {use_roles}")
    print(f"valid_input : {valid_input}")
    print(f"anomaly_in  : {anomaly_input}")
    print(f"out_dir     : {out_dir}")

    ranker = build_ranker(ranker_kind, ckpt)
    cand = candidate_vocab(FAMILIES)
    print(f"ranker built; candidate vocabulary = {len(cand)} steps")

    subs = predict.make_all_submissions(
        valid_input, anomaly_input, ranker, out_dir,
        candidate_vocab=cand, use_roles=use_roles,
    )

    # Summaries -------------------------------------------------------------
    def _row_count(p: Path) -> int:
        with Path(p).open(newline="", encoding="utf-8") as fh:
            return sum(1 for _ in csv.reader(fh)) - 1  # minus header

    print("\nwritten submissions:")
    for key in ("task1", "task2", "task3"):
        p = subs[key]
        print(f"  {key}: {p}  ({_row_count(p)} rows)")

    # Task-2 validity audit.
    v = validate_task2(valid_input, subs["task2"])
    pct = (100.0 * v["valid"] / v["total"]) if v["total"] else 0.0
    print(f"\nTask-2 validity (validate_with_roles): "
          f"{v['valid']}/{v['total']} valid ({pct:.1f}%)")
    if v["offenders"]:
        print("  first offenders (eid, rule):")
        for eid, rule in v["offenders"]:
            print(f"    {eid}: {rule}")

    subs["task2_validation"] = v
    return subs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Produce the 3 NSPE submission CSVs.")
    ap.add_argument("--ranker", choices=["ppm", "neural"], default="ppm",
                    help="ranker backend (default: ppm; runs with no GPU)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path for --ranker neural")
    ap.add_argument("--use-roles", action="store_true",
                    help="role-sharpen the symbolic candidate support")
    ap.add_argument("--out", default=None,
                    help="output dir (default: $NSPE_OUT or .../outputs)")
    args = ap.parse_args(argv)

    make_submission(
        ranker_kind=args.ranker,
        ckpt=args.ckpt,
        use_roles=args.use_roles,
        out_dir=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
