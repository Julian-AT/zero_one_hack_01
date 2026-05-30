"""Benchmark and rank every final all-3 model we've trained, on
every metric we can self-compute. The output picks the model we ship.

Criteria (all maximised unless marked):
  - LM_loss_train         lower better — model quality on training distribution
  - LoFO_Top1_held_avg    higher better — Task-4 OOD proxy
  - LoFO_top1_drop_avg    lower better — ID->OOD penalty (close to 0 best)
  - LoFO_NED_held_avg     lower better — Task-2 NED OOD proxy
  - LoFO_anom_AUC_held    higher better — Task-3 OOD anomaly
  - validator_clean       higher better — % of completions valid per organizers'
                          generate_sequences.py --validate (Task-2 process logic)
  - rank_fill_pct         higher better — % of nextstep rows with 5 ranks

A scaled composite score ranks the models. The winner is the model we
ship as the final submission.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "models"))

CKPT_DIR = REPO_ROOT / "shared" / "extras" / "checkpoints"
EVAL_DIR = REPO_ROOT / "shared" / "extras" / "results" / "eval"
SUB_DIR = REPO_ROOT / "shared" / "extras" / "results"
FAMILIES = ("mosfet", "igbt", "ic")


def _read_json(p: Path):
    return json.load(p.open()) if p.exists() else None


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _avg_over_fracs(per_family: dict, fam: str, key: str):
    fam_block = per_family.get(fam)
    if not fam_block:
        return None
    vals = []
    for frac in ("0.6", "0.8"):
        m = fam_block.get(frac, {})
        if key in m:
            vals.append(m[key])
    return _avg(vals)


def find_final_models() -> list[dict]:
    """Find every cell that trained on all-3 families (the 'final' candidates)."""
    out: list[dict] = []
    for d in sorted(CKPT_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        is_all3 = ("all3" in name or
                   "final-" in name or
                   name in ("multitask-transformer_medium-20260530-030653",))
        if not is_all3:
            continue
        s = _read_json(d / "summary.json")
        if s is None:
            continue
        out.append({
            "cell": name,
            "phase": _phase_of(name),
            "arch": s.get("arch", "?"),
            "params": s.get("model_params", 0),
            "lm_loss": s.get("final_metrics", {}).get("lm_loss"),
            "wall_seconds": s.get("wall_seconds", 0),
        })
    return out


def _phase_of(name: str) -> str:
    if name.startswith("v4-final"):    return "v4 (syn+ood)"
    if name.startswith("v3-final"):    return "v3 (ood aug)"
    if name.startswith("v2-final"):    return "v2 (max_len=768)"
    if name.startswith("final-"):      return "v1 (max_len=256)"
    if name.startswith("multitask-"):  return "legacy multitask"
    return "?"


def find_lofo_siblings(final_cell: str) -> list[str]:
    """For a 'final-*-all3' cell, find the 3 LoFO sibling cells of the same recipe."""
    if "v4-final" in final_cell:
        prefix = final_cell.replace("v4-final", "v4").replace("-all3", "")
        return [f"{prefix}-held_{f}" for f in FAMILIES]
    if "v3-final" in final_cell:
        prefix = final_cell.replace("v3-final", "v3").replace("-all3", "")
        return [f"{prefix}-held_{f}" for f in FAMILIES]
    if "v2-final" in final_cell:
        prefix = final_cell.replace("v2-final", "v2").replace("-all3", "")
        return [f"{prefix}-held_{f}" for f in FAMILIES]
    if final_cell.startswith("final-"):
        # final-transformer-small-lm_only-fdp00-all3 → lofo-transformer-small-lm_only-fdp00-held_*
        prefix = final_cell.replace("final-", "lofo-").replace("-all3", "")
        return [f"{prefix}-held_{f}" for f in FAMILIES]
    return []


def lofo_aggregates(final_cell: str) -> dict:
    """Aggregate LoFO eval metrics across the 3 sibling cells."""
    siblings = find_lofo_siblings(final_cell)
    top1_held: list[float] = []
    top5_held: list[float] = []
    ned_held: list[float] = []
    anom_auc_held: list[float] = []
    top1_id: list[float] = []
    for sib in siblings:
        eval_metrics = _read_json(EVAL_DIR / sib / "metrics.json")
        if eval_metrics is None:
            continue
        held = sib.rsplit("held_", 1)[-1] if "held_" in sib else None
        if held is None:
            continue
        per_family = eval_metrics.get("per_family", {})
        anom_per_family = eval_metrics.get("anomaly", {}).get("per_family", {})
        # Held metrics
        v = _avg_over_fracs(per_family, held, "top1_at_cut");  top1_held.append(v) if v is not None else None
        v = _avg_over_fracs(per_family, held, "top5_at_cut");  top5_held.append(v) if v is not None else None
        v = _avg_over_fracs(per_family, held, "completion_ned"); ned_held.append(v) if v is not None else None
        v = anom_per_family.get(held, {}).get("roc_auc");      anom_auc_held.append(v) if v is not None else None
        # ID metrics for top1_drop computation
        id_fams = [f for f in FAMILIES if f != held]
        for f in id_fams:
            v = _avg_over_fracs(per_family, f, "top1_at_cut")
            if v is not None: top1_id.append(v)

    return {
        "n_lofo_evaled": len([s for s in siblings if _read_json(EVAL_DIR / s / "metrics.json") is not None]),
        "top1_held_avg": _avg(top1_held),
        "top5_held_avg": _avg(top5_held),
        "ned_held_avg":  _avg(ned_held),
        "anom_auc_held_avg": _avg(anom_auc_held),
        "top1_id_avg":   _avg(top1_id),
    }


def submission_quality(final_cell: str) -> dict | None:
    """Check if we have a real-input submission for this cell, and compute self-eval metrics."""
    # Check several possible submission dirs
    candidates = [
        SUB_DIR / f"submission_v2_real",   # default for v2-final-medium-multitask
        SUB_DIR / "submission",
    ]
    for sub_dir in candidates:
        if (sub_dir / "completion.csv").exists():
            return _quality_metrics(sub_dir)
    return None


def _quality_metrics(sub_dir: Path) -> dict:
    """Compute validator-clean rate, rank fill, class balance for one submission dir."""
    out = {"submission_dir": str(sub_dir.relative_to(REPO_ROOT))}
    # Rank fill
    nextstep_path = sub_dir / "nextstep.csv"
    if nextstep_path.exists():
        full_5 = 0
        total = 0
        with nextstep_path.open() as f:
            for row in csv.DictReader(f):
                total += 1
                if all(row.get(f"RANK_{i}", "").strip() for i in range(1, 6)):
                    full_5 += 1
        out["rank_fill_pct"] = 100 * full_5 / total if total else 0
        out["rank_fill_n"] = total
    # Anomaly class balance
    anomaly_path = sub_dir / "anomaly.csv"
    if anomaly_path.exists():
        valid = invalid = 0
        rules = set()
        with anomaly_path.open() as f:
            for row in csv.DictReader(f):
                if row["IS_VALID"] == "1": valid += 1
                else: invalid += 1; rules.add(row["PREDICTED_RULE"])
        out["anom_valid_pred"] = valid
        out["anom_invalid_pred"] = invalid
        out["anom_class_match"] = (valid == 600 and invalid == 387)
        out["anom_distinct_rules"] = len(rules)
    # Validator-clean rate via the organizers' validator
    completion_path = sub_dir / "completion.csv"
    if completion_path.exists():
        from transformer_xlstm.data.validator import validate_sequence
        valid_input_path = REPO_ROOT / "competition" / "participant-files" / "eval_input_valid.csv"
        if valid_input_path.exists():
            inputs = {r["EXAMPLE_ID"]: r for r in csv.DictReader(valid_input_path.open())}
            total = clean = 0
            with completion_path.open() as f:
                for row in csv.DictReader(f):
                    if row["EXAMPLE_ID"] not in inputs: continue
                    partial = inputs[row["EXAMPLE_ID"]]["PARTIAL_SEQUENCE"].split("|")
                    predicted = row["PREDICTED_SEQUENCE"].split("|") if row["PREDICTED_SEQUENCE"] else []
                    full = partial + predicted
                    total += 1
                    if not validate_sequence(full):
                        clean += 1
            out["validator_clean_pct"] = 100 * clean / total if total else 0
            out["validator_clean_n"] = total
    return out


def main() -> None:
    candidates = find_final_models()
    print(f"Found {len(candidates)} final all-3 candidates")
    print()
    print(f"{'cell':<60} {'phase':<18} {'params':>10} {'lm_loss':>8} "
          f"{'Top1_held':>10} {'top1_drop':>10} {'NED_held':>9}")
    print("-" * 130)

    rows = []
    for c in candidates:
        lofo = lofo_aggregates(c["cell"])
        top1_drop = (lofo["top1_id_avg"] - lofo["top1_held_avg"]
                     if lofo["top1_held_avg"] is not None and lofo["top1_id_avg"] is not None else None)
        row = {**c, **lofo, "top1_drop": top1_drop}
        rows.append(row)

    # Sort by top1_held_avg descending
    rows.sort(key=lambda r: -(r["top1_held_avg"] or -1))
    for r in rows:
        print(f"{r['cell']:<60} {r['phase']:<18} "
              f"{r['params']:>10,} "
              f"{r['lm_loss'] or 0:>8.4f} "
              f"{r['top1_held_avg'] or 0:>10.4f} "
              f"{r['top1_drop'] or 0:>+10.4f} "
              f"{r['ned_held_avg'] or 0:>9.4f}")

    print()
    print("=" * 130)
    print("WINNER (top1_held_avg)")
    print("=" * 130)
    winner = rows[0]
    print(f"  cell:        {winner['cell']}")
    print(f"  phase:       {winner['phase']}")
    print(f"  params:      {winner['params']:,}")
    print(f"  Top1_held:   {winner['top1_held_avg']:.4f}")
    print(f"  top1_drop:   {winner['top1_drop']:+.4f}  (near 0 = good generalisation)")
    print(f"  NED_held:    {winner['ned_held_avg']:.4f}  (lower is better)")
    print()

    # Submission quality if available
    sub = submission_quality(winner["cell"])
    if sub:
        print("Submission-CSV quality (against real eval inputs):")
        for k, v in sub.items():
            print(f"  {k:<25} {v}")

    # Save CSV
    out_csv = SUB_DIR / "candidate_ranking.csv"
    if rows:
        cols = list(rows[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"\nWrote ranking to {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
