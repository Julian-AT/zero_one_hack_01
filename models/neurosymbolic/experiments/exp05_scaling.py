"""exp05 — model-size x data-size scaling sweep (+ a results aggregator).

Two modes.

SWEEP (default)
---------------
Cartesian product of ``d_model in {64, 128, 256}`` x ``data in {2 families,
3 families}`` (6 cells). Each cell trains the small constrained ranker, records
its parameter count and wall-clock train time, and measures next-step (Task 1)
metrics both in-distribution (ID) and out-of-distribution (OOD):

  * ID  — a held slice of the TRAIN families, scored with the constrained decoder.
  * OOD — the genuinely-unseen 4th family. For a 2-family run the OOD family is
    the held-out training family; for a 3-family run there is no held-out training
    family, so we use the organizers' OOD generator (``official.ood`` -> a
    ``diode``/``schottky``/``sic_mosfet`` family) — the same simulated unseen-family
    device used throughout NSPE. The candidate vocab is always the TRAIN-family
    vocab, so OOD step strings the model never saw are correctly absent.

The scaling story is the params/time vs ID/OOD curve, and especially how flat the
ID->OOD drop stays as size and data grow.

AGGREGATE (``--aggregate``)
---------------------------
Merge every ``$NSPE_OUT/*.json`` (exp03/04/05 outputs) into a single
``summary.json`` and a human-readable ``summary.md`` table. This is what the
grid sbatch calls after the LoFO runs finish.

Output: ``$NSPE_OUT/exp05.json`` (sweep) or ``$NSPE_OUT/summary.{json,md}`` (aggregate).
GPU experiment (runs on CPU too); seeded.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from nspe import eval as nspe_eval
from nspe import predict, simulate_eval
from nspe.data import FAMILIES, candidate_vocab, load_family

__all__ = ["run_sweep", "aggregate", "DEFAULT_SWEEP"]

METRIC_KEYS = nspe_eval.METRIC_KEYS

# Default sweep axes (overridable via --config grid.yaml's `sweep` block).
DEFAULT_SWEEP = {
    "d_model": [64, 128, 256],
    "data": [["mosfet", "igbt"], ["mosfet", "igbt", "ic"]],
}
# OOD family used when the run already trains on all three real families.
_OOD_GEN_FAMILY = "diode"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _out_dir() -> Path:
    out = Path(os.environ.get("NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not available but --config was provided")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping")
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _ood_seqs_for(train: Sequence[str], n_per_family: int, seed: int) -> Dict[str, List[List[str]]]:
    """Pick the OOD eval sequences for a run trained on ``train``.

    * If a real family is held out (2-family run) use it.
    * Otherwise generate an unseen 4th family with the organizers' OOD generator.
    """
    train_l = [f.lower() for f in train]
    held = [f for f in FAMILIES if f not in train_l]
    if held:
        fam = held[0]
        return {fam: [list(s) for s in load_family(fam)[:n_per_family]]}
    # All three real families are in train -> simulated unseen 4th family.
    from nspe.official import ood
    gen = ood.generate_unique(_OOD_GEN_FAMILY, n_per_family, seed)
    return {_OOD_GEN_FAMILY: [list(s) for s in gen]}


def _score_next_step(
    ranker, seqs_by_family: Dict[str, List[List[str]]], cand: List[str],
    split_dir: Path, tag: str, use_roles: bool,
) -> Dict[str, float]:
    eset = simulate_eval.build_eval_set(seqs_by_family, split_dir, prefix=tag)
    ns_pred = split_dir / f"{tag}_nextstep_pred.csv"
    predict.predict_nextstep(eset["nextstep_input"], ns_pred, ranker, cand, use_roles=use_roles)
    ns = nspe_eval.score("next-step", eset["nextstep_gt"], ns_pred)
    return {k: ns[k] for k in METRIC_KEYS["next-step"] if k in ns}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(
    config_path: Optional[str] = None,
    smoke: bool = False,
    seed: int = 0,
    n_per_family: int = 60,
    use_roles: bool = True,
) -> Dict:
    """Run the d_model x data sweep and write ``$NSPE_OUT/exp05.json``."""
    _seed_everything(seed)
    out_dir = _out_dir()
    cfg = _load_config(config_path)

    # grid.yaml layout: {base: {...}, sweep: {d_model: [...], data: [[...], ...]}}.
    base_cfg = dict(cfg.get("base", cfg)) if "base" in cfg or "sweep" in cfg else dict(cfg)
    sweep = dict(DEFAULT_SWEEP)
    if isinstance(cfg.get("sweep"), dict):
        sweep.update(cfg["sweep"])
    base_cfg.setdefault("seed", seed)

    d_models = list(sweep["d_model"])
    data_axes = [list(d) for d in sweep["data"]]
    if smoke:
        n_per_family = min(n_per_family, 6)
        d_models = d_models[:2]                       # 64, 128
        data_axes = data_axes[:2]
        base_cfg.setdefault("steps", 60)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    from nspe.model import load_ranker, train_ranker

    t0 = time.time()
    cells: List[Dict] = []
    for train in data_axes:
        train = [f.lower() for f in train]
        cand = list(candidate_vocab(train))
        id_seqs = {f: [list(s) for s in load_family(f)[:n_per_family]] for f in train}
        ood_seqs = _ood_seqs_for(train, n_per_family, seed)
        ood_name = next(iter(ood_seqs))
        for d_model in d_models:
            cfg_cell = dict(base_cfg)
            cfg_cell["d_model"] = int(d_model)
            tag = f"d{d_model}_{'+'.join(train)}"
            cell_dir = out_dir / f"exp05_{tag}"
            cell_dir.mkdir(parents=True, exist_ok=True)

            res = train_ranker(train, config=cfg_cell, out_dir=str(cell_dir), smoke=smoke)
            ranker = load_ranker(res["ckpt_path"])

            id_ns = _score_next_step(ranker, id_seqs, cand, cell_dir, "id", use_roles)
            ood_ns = _score_next_step(ranker, ood_seqs, cand, cell_dir, "ood", use_roles)
            drop = {k: id_ns[k] - ood_ns[k] for k in id_ns if k in ood_ns}

            cell = {
                "d_model": int(d_model),
                "train_families": train,
                "n_train_families": len(train),
                "ood_family": ood_name,
                "n_params": res["metrics"]["n_params"],
                "train_wall_sec": res["metrics"]["wall_sec"],
                "steps_run": res["metrics"]["steps_run"],
                "final_loss": res["metrics"]["final_loss"],
                "id_next_step": id_ns,
                "ood_next_step": ood_ns,
                "drop_next_step": drop,
            }
            cells.append(cell)
            print(f"[d={d_model:3d} | {len(train)}fam] params={cell['n_params']:>8} "
                  f"train={cell['train_wall_sec']:.1f}s "
                  f"ID_top1={id_ns.get('top1')} OOD_top1={ood_ns.get('top1')} "
                  f"(ood={ood_name})")

    result = {
        "experiment": "exp05_scaling",
        "device": device,
        "smoke": smoke,
        "base_config": base_cfg,
        "d_models": d_models,
        "data_axes": data_axes,
        "n_per_family": n_per_family,
        "cells": cells,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = out_dir / "exp05.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    result["json_path"] = str(out_json)

    print("=" * 72)
    print(f"exp05 scaling sweep ({device}): {len(cells)} cells")
    print(f"json -> {out_json}")
    return result


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def aggregate(out_dir: Optional[Path] = None) -> Dict:
    """Merge every ``*.json`` in the output dir into ``summary.{json,md}``.

    Recognises exp03 (neural-vs-PPM comparison), exp04 (ablation cells), and exp05
    (scaling cells); unknown JSONs are still indexed by filename. Returns the
    merged summary dict.
    """
    out_dir = Path(out_dir) if out_dir is not None else _out_dir()
    merged: Dict[str, Dict] = {}
    for jf in sorted(out_dir.glob("*.json")):
        if jf.name in ("summary.json", "lofo.json"):
            continue
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "experiment" in data:
            merged[jf.name] = data

    md_lines: List[str] = ["# NSPE experiment summary", ""]

    # ---- exp05 scaling table ----
    exp05 = [d for d in merged.values() if d.get("experiment") == "exp05_scaling"]
    if exp05:
        md_lines += ["## exp05 — scaling sweep (next-step Top-1)", "",
                     "| d_model | #fam | params | train_s | ID top1 | OOD top1 | drop top1 | OOD fam |",
                     "|--------:|-----:|-------:|--------:|--------:|---------:|----------:|---------|"]
        for d in exp05:
            for c in d.get("cells", []):
                md_lines.append(
                    f"| {c['d_model']} | {c['n_train_families']} | {c['n_params']} | "
                    f"{_fmt(c['train_wall_sec'])} | {_fmt(c['id_next_step'].get('top1'))} | "
                    f"{_fmt(c['ood_next_step'].get('top1'))} | "
                    f"{_fmt(c['drop_next_step'].get('top1'))} | {c.get('ood_family','-')} |")
        md_lines.append("")

    # ---- exp04 ablation table ----
    exp04 = [d for d in merged.values() if d.get("experiment") == "exp04_constraint_loss"]
    if exp04:
        md_lines += ["## exp04 — constraint-loss ablation", "",
                     "| mask_train | sem_w | free invalid_rate | mean_viol | ID top1 | OOD top1 |",
                     "|:----------:|------:|------------------:|----------:|--------:|---------:|"]
        for d in exp04:
            for c in d.get("cells", []):
                fg = c.get("free_generation", {})
                md_lines.append(
                    f"| {c['mask_train']} | {c['sem_w']} | {_fmt(fg.get('invalid_rate'))} | "
                    f"{_fmt(fg.get('mean_violations'))} | {_fmt(c['id_next_step'].get('top1'))} | "
                    f"{_fmt(c['ood_next_step'].get('top1'))} |")
        md_lines.append("")

    # ---- exp03 neural-vs-PPM table ----
    exp03 = [d for d in merged.values() if d.get("experiment") == "exp03_neural_ranker"]
    if exp03:
        md_lines += ["## exp03 — neural ranker vs PPM (next-step Top-1)", "",
                     "| holdout | ranker | ID top1 | OOD top1 | drop top1 |",
                     "|---------|--------|--------:|---------:|----------:|"]
        for d in exp03:
            cmp = d.get("comparison", {})
            for who in ("neural", "ppm"):
                rec = cmp.get(who, {})
                idv = rec.get("id", {}).get("next-step", {}).get("top1")
                oodv = rec.get("ood", {}).get("next-step", {}).get("top1")
                dropv = rec.get("drop", {}).get("next-step", {}).get("top1")
                md_lines.append(
                    f"| {d.get('holdout')} | {who} | {_fmt(idv)} | {_fmt(oodv)} | {_fmt(dropv)} |")
        md_lines.append("")

    summary = {
        "experiments": list(merged.keys()),
        "exp03": [d.get("holdout") for d in exp03],
        "n_exp05_cells": sum(len(d.get("cells", [])) for d in exp05),
        "n_exp04_cells": sum(len(d.get("cells", [])) for d in exp04),
        "merged": merged,
    }
    sj = out_dir / "summary.json"
    with sj.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    sm = out_dir / "summary.md"
    with sm.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    print(f"aggregated {len(merged)} experiment JSON(s) -> {sj.name}, {sm.name}")
    summary["json_path"] = str(sj)
    summary["md_path"] = str(sm)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML grid config (base + sweep axes)")
    p.add_argument("--aggregate", action="store_true",
                   help="merge outputs/*.json into summary.{json,md} (no training)")
    p.add_argument("--smoke", action="store_true", help="tiny fast run for CI / pipeline check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-per-family", type=int, default=60)
    p.add_argument("--no-roles", action="store_true", help="disable role-sharpened decoding")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.aggregate:
        aggregate()
    else:
        run_sweep(
            config_path=args.config,
            smoke=args.smoke,
            seed=args.seed,
            n_per_family=args.n_per_family,
            use_roles=not args.no_roles,
        )
