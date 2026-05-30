#!/usr/bin/env python3
"""exp02 — Tasks 1 & 2 with the pure-symbolic PPM ranker (CPU).

This experiment measures next-step (Task 1) and completion (Task 2) performance
of the role-factored variable-order Markov ranker (``nspe.ppm.PPM``) under two
regimes, both scored with the OFFICIAL ``eval_metrics.py`` (via ``nspe.eval``):

  * LoFO (the Task-4 measurement) — ``nspe.eval.lofo`` trains the PPM on two
    families and scores it on BOTH an in-distribution (ID) eval set built from
    the train families and an out-of-distribution (OOD) eval set built from the
    held-out family, recording the ID->OOD drop per metric. The headline
    neurosymbolic claim is that this drop is *flat* because the support
    (grammar) and the ranker's role factor are family-agnostic.

  * ID run — train one PPM on all three families and score it on a held-out
    subset of the same families (a clean in-distribution reference, no family
    shift). This is the upper-reference for the LoFO ID column.

All decoding goes through the symbolic ``nspe.decode`` (constrained next-step
top-k and constrained completion), so it is a *pure symbolic* pipeline — no
torch is imported anywhere in this experiment.

Results are dumped to ``$NSPE_OUT/exp02.json`` (default
``neurosymbolic-approach/outputs``) and Top-1/3/5, MRR, NED, Exact, Token, and
Block tables are printed with the ID-vs-OOD drop.

Flags:
  --limit         subsample the eval sets for speed (sequences per family)
  --n-per-family  alias kept for the spec wording; same meaning as --limit
  --seed          RNG seed
  --use-roles     enable role-augmented constrained decoding
  --beam          completion beam width
  --id-train-cap  cap sequences/family when fitting the ID-run PPM
  --smoke         single LoFO split, tiny n (fast pipeline check)

NOTE: this file is *built and import-checked* here; the full run is launched by
the Integration phase.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nspe import data, eval as nspe_eval, predict, simulate_eval
from nspe.ppm import PPM

__all__ = ["make_ranker", "run_id", "run", "main"]


# --------------------------------------------------------------------------- #
# Output dir
# --------------------------------------------------------------------------- #

def _out_dir() -> Path:
    d = Path(os.environ.get(
        "NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Ranker factory (the exact form the spec asks for)
# --------------------------------------------------------------------------- #

def make_ranker(families: Sequence[str]) -> PPM:
    """Fit a fresh PPM on the given training families.

    ``data.load_families`` returns ``{fam: tuple[tuple[str], ...]}`` which PPM's
    ``fit`` consumes directly (it only iterates over the sequences).
    """
    return PPM().fit(data.load_families(list(families)))


# --------------------------------------------------------------------------- #
# ID run: train on all three families, eval on a held-out subset of the SAME
# families (no family shift — the in-distribution reference).
# --------------------------------------------------------------------------- #

def run_id(
    limit: int,
    seed: int,
    out_dir: Path,
    use_roles: bool = False,
    beam: int = 1,
    train_cap: Optional[int] = None,
) -> dict:
    """Train one PPM on all three families; score on a held-out subset.

    To keep the eval set strictly in-distribution but not memorised, we fit on
    the *first* ``train_cap`` sequences per family and evaluate on a disjoint
    tail slice of ``limit`` sequences per family.
    """
    out_dir = Path(out_dir)
    fams = list(data.FAMILIES)

    train_by_fam: Dict[str, list] = {}
    eval_by_fam: Dict[str, list] = {}
    rng = random.Random(seed)
    for fam in fams:
        seqs = [list(s) for s in data.load_family(fam)]
        rng.shuffle(seqs)
        cap = train_cap if train_cap else max(1, len(seqs) - limit)
        cap = min(cap, len(seqs))
        train_by_fam[fam] = seqs[:cap]
        # Disjoint eval tail (falls back to the train head if the family is tiny).
        tail = seqs[cap:]
        eval_by_fam[fam] = (tail or seqs)[:limit]

    ranker = PPM().fit(train_by_fam)
    cand = list(data.candidate_vocab(fams))

    split_dir = out_dir / "id_run"
    split_dir.mkdir(parents=True, exist_ok=True)
    eset = simulate_eval.build_eval_set(eval_by_fam, split_dir, prefix="id")

    ns_pred = split_dir / "id_nextstep_pred.csv"
    predict.predict_nextstep(eset["nextstep_input"], ns_pred, ranker, cand,
                             use_roles=use_roles)
    ns_metrics = nspe_eval.score("next-step", eset["nextstep_gt"], ns_pred)

    cp_pred = split_dir / "id_completion_pred.csv"
    predict.predict_completion(eset["completion_input"], cp_pred, ranker, cand,
                               use_roles=use_roles, beam=beam)
    cp_metrics = nspe_eval.score("completion", eset["completion_gt"], cp_pred)

    return {
        "train_families": fams,
        "n_train_per_family": {f: len(v) for f, v in train_by_fam.items()},
        "n_eval_per_family": {f: len(v) for f, v in eval_by_fam.items()},
        "next-step": {k: ns_metrics[k] for k in nspe_eval.METRIC_KEYS["next-step"]
                      if k in ns_metrics},
        "completion": {k: cp_metrics[k] for k in nspe_eval.METRIC_KEYS["completion"]
                       if k in cp_metrics},
        "returncode": {"next-step": ns_metrics["returncode"],
                       "completion": cp_metrics["returncode"]},
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(
    limit: int,
    seed: int,
    out_dir: Optional[Path] = None,
    use_roles: bool = False,
    beam: int = 1,
    train_cap: Optional[int] = None,
    smoke: bool = False,
) -> dict:
    """Run the LoFO harness + the ID run and assemble the result dict."""
    out_dir = Path(out_dir) if out_dir is not None else _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # ---- LoFO (Task 4): the spec's exact make_ranker form ----
    lofo_summary = nspe_eval.lofo(
        make_ranker,
        n_per_family=limit,
        out_dir=out_dir / "exp02_lofo",
        smoke=smoke,
        use_roles=use_roles,
        beam=beam,
    )

    # ---- ID run (train all 3, eval held-out subset of same families) ----
    id_run = run_id(limit if not smoke else min(limit, 8), seed, out_dir,
                    use_roles=use_roles, beam=beam, train_cap=train_cap)

    result = {
        "config": {
            "limit": limit, "seed": seed, "use_roles": use_roles, "beam": beam,
            "train_cap": train_cap, "smoke": smoke, "out_dir": str(out_dir),
        },
        "id_run": id_run,
        "lofo": {
            "splits": lofo_summary["splits"],
            "mean_drop": lofo_summary["mean_drop"],
            "n_per_family": lofo_summary["n_per_family"],
            "json_path": lofo_summary.get("json_path"),
        },
    }
    return result


# --------------------------------------------------------------------------- #
# Pretty tables
# --------------------------------------------------------------------------- #

def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def _row(label: str, d: dict, keys: Sequence[str]) -> str:
    return f"  {label:16s} " + " ".join(_fmt(d.get(k)).rjust(9) for k in keys)


def _print_tables(result: dict) -> None:
    sep = "=" * 84
    print(sep)
    print("exp02 — PPM ranker (pure symbolic): Tasks 1 & 2, ID vs OOD (LoFO)")
    print(sep)
    cfg = result["config"]
    print(f"config: limit={cfg['limit']}  seed={cfg['seed']}  "
          f"use_roles={cfg['use_roles']}  beam={cfg['beam']}  "
          f"smoke={cfg['smoke']}")

    ns_keys = nspe_eval.METRIC_KEYS["next-step"]      # top1 top3 top5 mrr
    cp_keys = nspe_eval.METRIC_KEYS["completion"]     # ned exact_match token_acc block_acc

    # ---- ID run reference ----
    idr = result["id_run"]
    print("\n[ID run] train all 3 families, eval disjoint same-family subset")
    print("  next-step  " + " ".join(k.rjust(9) for k in ns_keys))
    print(_row("", idr["next-step"], ns_keys))
    print("  completion " + " ".join(k.rjust(9) for k in cp_keys))
    print(_row("", idr["completion"], cp_keys))

    # ---- LoFO per split ----
    print("\n[LoFO] per held-out family (ID = train families, OOD = held-out)")
    for split in result["lofo"]["splits"]:
        held = split["held_out"]
        print(f"\n  held-out = {held.upper()}  (train {split['train']})")
        print("    next-step   " + " ".join(k.rjust(9) for k in ns_keys))
        print(_row("ID", split["id"]["next-step"], ns_keys))
        print(_row("OOD", split["ood"]["next-step"], ns_keys))
        print(_row("drop(ID-OOD)", split["drop"]["next-step"], ns_keys))
        print("    completion  " + " ".join(k.rjust(9) for k in cp_keys))
        print(_row("ID", split["id"]["completion"], cp_keys))
        print(_row("OOD", split["ood"]["completion"], cp_keys))
        print(_row("drop(ID-OOD)", split["drop"]["completion"], cp_keys))

    # ---- mean drop ----
    md = result["lofo"]["mean_drop"]
    print("\n[LoFO] mean ID->OOD drop across splits (lower = flatter = better)")
    print("  next-step   " + " ".join(k.rjust(9) for k in ns_keys))
    print(_row("mean drop", md.get("next-step", {}), ns_keys))
    print("  completion  " + " ".join(k.rjust(9) for k in cp_keys))
    print(_row("mean drop", md.get("completion", {}), cp_keys))
    print("  (NED drop is ID-OOD; NED is lower-is-better so a NEGATIVE drop is good)")
    print(sep)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100,
                        help="subsample eval sets (sequences per family)")
    parser.add_argument("--n-per-family", type=int, default=None,
                        help="alias for --limit (spec wording)")
    parser.add_argument("--seed", type=int, default=20240530, help="RNG seed")
    parser.add_argument("--use-roles", action="store_true",
                        help="role-augmented constrained decoding")
    parser.add_argument("--beam", type=int, default=1, help="completion beam width")
    parser.add_argument("--id-train-cap", type=int, default=None,
                        help="cap sequences/family for the ID-run PPM fit")
    parser.add_argument("--smoke", action="store_true",
                        help="single LoFO split, tiny n (fast check)")
    parser.add_argument("--out", type=str, default=None,
                        help="output dir (default $NSPE_OUT or outputs/)")
    args = parser.parse_args(argv)

    limit = args.n_per_family if args.n_per_family is not None else args.limit

    out_dir = Path(args.out) if args.out else _out_dir()
    result = run(
        limit, args.seed, out_dir,
        use_roles=args.use_roles, beam=args.beam,
        train_cap=args.id_train_cap, smoke=args.smoke,
    )

    out_json = out_dir / "exp02.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_tables(result)
    print(f"\njson -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
