#!/usr/bin/env python3
"""exp01 — Task-3 anomaly oracle, scored end-to-end with the OFFICIAL scorer.

This experiment exercises the pure-symbolic (role-augmented) Task-3 oracle in
``nspe.anomaly.classify`` and scores it with the organizers' ``eval_metrics.py``
(via ``nspe.eval.score``). It has two parts:

  (a) ID  — sample valid sequences from the three provided families
            (mosfet/igbt/ic), inject all ten rule violations with
            ``nspe.corrupt`` (known vocabulary), classify every sequence with
            ``classify(use_roles=True)``, then build the official forbidden
            ground-truth (``EXAMPLE_ID,VIOLATION_RULE``) + a valid supplement
            (``EXAMPLE_ID``) and score accuracy / precision / recall / F1 / AUC /
            rule-attribution. We also print a per-rule recall + attribution
            table (the rule the oracle attributes vs. the injected rule).

  (b) OOD — generate diode/schottky/sic_mosfet sequences via the official OOD
            family generators (``official.ood.generate_unique``), inject
            violations with ``novel=True`` (the trigger step is renamed to an
            unseen string so the stock validator is blind), and measure recall
            with role-induction ON vs OFF to show the recovery — mirroring
            ``experiments/ood_symbolic_probe.py`` but routed through the same
            official scorer.

Everything is pure symbolic (no torch). Results are dumped to
``$NSPE_OUT/exp01.json`` (default ``neurosymbolic-approach/outputs``) and a set
of human-readable tables is printed.

Flags: ``--n`` (sequences per family pool), ``--seed``.

NOTE: this file is *built and import-checked* here; the full run is launched by
the Integration phase.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nspe import anomaly, corrupt, eval as nspe_eval, rules
from nspe.data import all_sequences
from nspe.official import ood

__all__ = [
    "build_anomaly_csvs",
    "score_anomaly_set",
    "per_rule_table",
    "ood_recall_probe",
    "run",
    "main",
]

OOD_FAMILIES = ("diode", "schottky", "sic_mosfet")


# --------------------------------------------------------------------------- #
# Output dir
# --------------------------------------------------------------------------- #

def _out_dir() -> Path:
    d = Path(os.environ.get(
        "NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# CSV construction for the official anomaly scorer
# --------------------------------------------------------------------------- #

def build_anomaly_csvs(
    anomaly_rows: Sequence[dict],
    out_dir: Path,
    prefix: str,
    use_roles: bool = True,
) -> dict:
    """Classify an anomaly set and write the three CSVs the scorer consumes.

    ``anomaly_rows`` is the output of ``nspe.corrupt.make_anomaly_set`` — each
    dict has ``EXAMPLE_ID``, ``SEQUENCE`` (list[str]), ``IS_VALID`` (1/0),
    ``VIOLATION_RULE`` (str or None).

    Writes (and returns paths to):
      * ``<prefix>_forbidden_gt.csv``  — EXAMPLE_ID,VIOLATION_RULE (the invalid
        rows; label 0 to the scorer).
      * ``<prefix>_valid_supp.csv``    — EXAMPLE_ID (the valid rows; label 1).
      * ``<prefix>_pred.csv``          — EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE
        (the oracle's predictions for *every* row).

    Returns ``{"forbidden_gt", "valid_supp", "pred", "n_invalid", "n_valid"}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forbidden_gt = out_dir / f"{prefix}_forbidden_gt.csv"
    valid_supp = out_dir / f"{prefix}_valid_supp.csv"
    pred = out_dir / f"{prefix}_pred.csv"

    n_invalid = n_valid = 0
    with forbidden_gt.open("w", newline="", encoding="utf-8") as ffh, \
            valid_supp.open("w", newline="", encoding="utf-8") as vfh, \
            pred.open("w", newline="", encoding="utf-8") as pfh:
        fw = csv.DictWriter(ffh, fieldnames=["EXAMPLE_ID", "VIOLATION_RULE"])
        vw = csv.DictWriter(vfh, fieldnames=["EXAMPLE_ID"])
        pw = csv.DictWriter(
            pfh, fieldnames=["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        fw.writeheader(); vw.writeheader(); pw.writeheader()

        for row in anomaly_rows:
            eid = row["EXAMPLE_ID"]
            seq = list(row["SEQUENCE"])
            res = anomaly.classify(seq, use_roles=use_roles)
            pw.writerow({
                "EXAMPLE_ID": eid,
                "IS_VALID": res["is_valid"],
                "SCORE": f"{res['score']:.4f}",
                "PREDICTED_RULE": res["rule"] if res["rule"] is not None else "",
            })
            if row["IS_VALID"] == 0:
                n_invalid += 1
                fw.writerow({
                    "EXAMPLE_ID": eid,
                    "VIOLATION_RULE": row["VIOLATION_RULE"] or "",
                })
            else:
                n_valid += 1
                vw.writerow({"EXAMPLE_ID": eid})

    return {
        "forbidden_gt": forbidden_gt,
        "valid_supp": valid_supp,
        "pred": pred,
        "n_invalid": n_invalid,
        "n_valid": n_valid,
    }


def score_anomaly_set(
    anomaly_rows: Sequence[dict],
    out_dir: Path,
    prefix: str,
    use_roles: bool = True,
) -> dict:
    """Build the CSVs and score with the official anomaly scorer.

    Returns the parsed metric dict from ``nspe.eval.score`` augmented with the
    CSV paths and class counts.
    """
    csvs = build_anomaly_csvs(anomaly_rows, out_dir, prefix, use_roles=use_roles)
    metrics = nspe_eval.score(
        "anomaly",
        csvs["forbidden_gt"],
        csvs["pred"],
        valid_supplement=csvs["valid_supp"],
    )
    metrics["n_invalid"] = csvs["n_invalid"]
    metrics["n_valid"] = csvs["n_valid"]
    metrics["paths"] = {k: str(v) for k, v in csvs.items()
                        if isinstance(v, Path)}
    return metrics


# --------------------------------------------------------------------------- #
# Per-rule recall + attribution table (independent of the scorer aggregate)
# --------------------------------------------------------------------------- #

def per_rule_table(
    anomaly_rows: Sequence[dict],
    use_roles: bool = True,
) -> Dict[str, dict]:
    """Per-injected-rule detection recall and attribution accuracy.

    For each invalid row we re-run the oracle and check (i) was it detected as
    invalid (recall), and (ii) did the attributed rule equal the injected one
    (attribution). Returns ``{rule: {n, detected, attributed}}``.
    """
    agg: Dict[str, dict] = defaultdict(lambda: {"n": 0, "detected": 0, "attributed": 0})
    for row in anomaly_rows:
        if row["IS_VALID"] != 0:
            continue
        rule = row["VIOLATION_RULE"]
        res = anomaly.classify(list(row["SEQUENCE"]), use_roles=use_roles)
        cell = agg[rule]
        cell["n"] += 1
        if res["is_valid"] == 0:
            cell["detected"] += 1
            if res["rule"] == rule:
                cell["attributed"] += 1
    return dict(agg)


# --------------------------------------------------------------------------- #
# OOD recall probe (role-induction ON vs OFF), mirroring ood_symbolic_probe.py
# --------------------------------------------------------------------------- #

def ood_recall_probe(
    n: int,
    seed: int,
    out_dir: Path,
) -> dict:
    """Generate OOD valids, inject NOVEL-vocab violations, measure recovery.

    Builds a novel-vocab anomaly set from the three OOD generators, then scores
    it twice through the official scorer: with role-induction OFF (the stock
    validator, blind to renamed triggers) and ON (recovers them). Also reports
    the false-positive count on OOD valids for both modes.
    """
    valids: List[list] = []
    for i, fam in enumerate(OOD_FAMILIES):
        valids.extend(ood.generate_unique(fam, n, seed=seed + 1000 * i))

    rng = random.Random(seed)
    novel_set = corrupt.make_anomaly_set(valids, rng, novel=True, frac_invalid=0.4)

    result: dict = {
        "n_per_family": n,
        "n_total": len(novel_set),
        "families": list(OOD_FAMILIES),
        "modes": {},
        "novel_capable_rules": sorted(corrupt.NOVEL_CAPABLE),
    }

    for tag, use_roles in (("roles_off", False), ("roles_on", True)):
        metrics = score_anomaly_set(
            novel_set, out_dir, f"ood_{tag}", use_roles=use_roles)
        # False positives on OOD valids: a valid row the oracle calls invalid.
        fp = sum(
            1 for r in novel_set if r["IS_VALID"] == 1
            and anomaly.classify(list(r["SEQUENCE"]), use_roles=use_roles)["is_valid"] == 0
        )
        # Recall restricted to the novel-capable rules (the ones that exercise
        # role-induction), since the ordering rules carry known vocabulary and
        # are caught in both modes.
        novel_rows = [r for r in novel_set if r["IS_VALID"] == 0
                      and r["VIOLATION_RULE"] in corrupt.NOVEL_CAPABLE]
        novel_detected = sum(
            1 for r in novel_rows
            if anomaly.classify(list(r["SEQUENCE"]), use_roles=use_roles)["is_valid"] == 0
        )
        result["modes"][tag] = {
            "recall": metrics.get("recall"),
            "precision": metrics.get("precision"),
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "auc": metrics.get("auc"),
            "rule_attr": metrics.get("rule_attr"),
            "fp_on_valids": fp,
            "novel_capable_n": len(novel_rows),
            "novel_capable_detected": novel_detected,
            "returncode": metrics.get("returncode"),
        }
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(n: int, seed: int, out_dir: Optional[Path] = None) -> dict:
    """Run both parts (ID + OOD) and return the assembled result dict."""
    out_dir = Path(out_dir) if out_dir is not None else _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- (a) ID: pool valid sequences from the three known families ----
    pairs = all_sequences()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    by_family: Dict[str, list] = defaultdict(list)
    for fam, seq in pairs:
        by_family[fam].append(seq)
    id_pool: List[list] = []
    for fam in sorted(by_family):
        id_pool.extend(by_family[fam][:n])
    rng.shuffle(id_pool)

    id_set = corrupt.make_anomaly_set(id_pool, random.Random(seed + 1),
                                      novel=False, frac_invalid=0.4)
    id_metrics = score_anomaly_set(id_set, out_dir, "id", use_roles=True)
    id_per_rule = per_rule_table(id_set, use_roles=True)

    # ---- (b) OOD: novel-vocab injection, roles ON vs OFF ----
    ood_result = ood_recall_probe(n, seed + 2, out_dir)

    result = {
        "config": {"n": n, "seed": seed, "out_dir": str(out_dir)},
        "id": {
            "metrics": {k: id_metrics.get(k) for k in nspe_eval.METRIC_KEYS["anomaly"]},
            "n_invalid": id_metrics["n_invalid"],
            "n_valid": id_metrics["n_valid"],
            "returncode": id_metrics["returncode"],
            "per_rule": id_per_rule,
        },
        "ood": ood_result,
        "rules": list(rules.RULE_IDS),
    }
    return result


def _print_tables(result: dict) -> None:
    sep = "=" * 76
    print(sep)
    print("exp01 — Task-3 symbolic anomaly oracle (official scorer)")
    print(sep)
    cfg = result["config"]
    print(f"config: n={cfg['n']}  seed={cfg['seed']}  out={cfg['out_dir']}")

    # ---- ID aggregate ----
    idm = result["id"]["metrics"]
    print("\n[ID] aggregate (mosfet/igbt/ic, known vocabulary)")
    print(f"  invalid={result['id']['n_invalid']}  valid={result['id']['n_valid']}")
    print(f"  {'accuracy':10s} {'precision':10s} {'recall':10s} "
          f"{'f1':10s} {'auc':10s} {'rule_attr':10s}")

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

    print("  " + " ".join(_fmt(idm.get(k)).ljust(10) for k in
                          ("accuracy", "precision", "recall", "f1", "auc", "rule_attr")))

    # ---- ID per-rule ----
    print("\n[ID] per-injected-rule recall + attribution")
    print(f"  {'rule':34s} {'n':>4s} {'recall':>8s} {'attr_acc':>9s}")
    for rule in result["rules"]:
        cell = result["id"]["per_rule"].get(rule)
        if not cell or cell["n"] == 0:
            print(f"  {rule:34s} {0:>4d} {'  --  ':>8s} {'  --  ':>9s}")
            continue
        rec = cell["detected"] / cell["n"]
        attr = cell["attributed"] / cell["detected"] if cell["detected"] else float("nan")
        print(f"  {rule:34s} {cell['n']:>4d} {rec:>8.3f} {attr:>9.3f}")

    # ---- OOD roles ON vs OFF ----
    o = result["ood"]
    print(f"\n[OOD] novel-vocab injection  (families: {', '.join(o['families'])}, "
          f"n/family={o['n_per_family']}, total={o['n_total']})")
    print(f"  novel-capable rules: {', '.join(o['novel_capable_rules'])}")
    print(f"  {'mode':10s} {'recall':>8s} {'precision':>10s} {'f1':>8s} "
          f"{'fp_valids':>10s} {'novel_det/n':>14s}")
    for tag in ("roles_off", "roles_on"):
        m = o["modes"][tag]
        nd = f"{m['novel_capable_detected']}/{m['novel_capable_n']}"
        print(f"  {tag:10s} {_fmt(m['recall']):>8s} {_fmt(m['precision']):>10s} "
              f"{_fmt(m['f1']):>8s} {m['fp_on_valids']:>10d} {nd:>14s}")
    print(sep)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200,
                        help="sequences per family for the ID/OOD pools")
    parser.add_argument("--seed", type=int, default=20240530, help="RNG seed")
    parser.add_argument("--out", type=str, default=None,
                        help="output dir (default $NSPE_OUT or outputs/)")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    out_dir = Path(args.out) if args.out else _out_dir()
    result = run(args.n, args.seed, out_dir)

    out_json = out_dir / "exp01.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_tables(result)
    print(f"\njson -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
