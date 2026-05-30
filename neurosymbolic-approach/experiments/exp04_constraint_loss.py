"""exp04 — constraint-loss / mask ablation: does the semantic loss reduce invalid mass?

Ablation grid (4 small runs):

    {mask_train: false, true}  x  {sem_w: 0.0, 0.5}

For each cell we train the small constrained ranker, then measure two things on
exactly the same data:

  1. **Invalid-emission rate of FREE (unmasked) generation.** We roll the ranker
     out greedily WITHOUT the symbolic grammar mask (the ranker picks the
     argmax over the full candidate vocab at every step) and validate the
     completed sequence with ``nspe.rules.validate_with_roles``. The fraction of
     completions that contain at least one rule violation is the invalid-emission
     rate, and the mean number of violations per sequence is the "invalid mass".
     The neurosymbolic claim is that the semantic loss (``sem_w > 0``) pushes
     softmax mass into the legal support, so free generation violates less often
     even with no inference-time mask.

  2. **OOD Top-1** on one held-out (LoFO) family, decoded WITH the constrained
     decoder (the production path), to confirm the loss does not hurt — and
     ideally helps — the actual OOD ranking metric.

We also report ID Top-1 for context. Both metrics are scored against
locally-simulated official ground truth via ``nspe.eval`` / ``nspe.simulate_eval``.

Output: ``$NSPE_OUT/exp04.json``. GPU experiment (runs on CPU too); seeded.
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
from nspe import predict, rules, simulate_eval
from nspe.data import LOFO_SPLITS, candidate_vocab, load_family

__all__ = ["run", "free_generate", "invalid_emission_rate"]

METRIC_KEYS = nspe_eval.METRIC_KEYS
_TERMINAL = "SHIP LOT"

# The ablation grid.
GRID = [
    {"mask_train": False, "sem_w": 0.0},
    {"mask_train": False, "sem_w": 0.5},
    {"mask_train": True, "sem_w": 0.0},
    {"mask_train": True, "sem_w": 0.5},
]


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


# ---------------------------------------------------------------------------
# FREE (unmasked) generation + invalid-emission rate
# ---------------------------------------------------------------------------
def free_generate(
    prefix: Sequence[str], family: str, ranker, cand_vocab: Sequence[str],
    max_len: int = 220,
) -> List[str]:
    """Greedy roll-out using ONLY the ranker (no symbolic ``valid_next_set`` mask).

    At each step the ranker scores the full candidate vocab and we append its
    argmax. This is the *un-symbolic* path used to measure how often the learned
    model alone produces a rule-violating continuation. Returns prefix + suffix.
    """
    seq = list(prefix)
    cand_vocab = list(cand_vocab)
    while len(seq) < max_len:
        scores = ranker.predict(seq, family, cand_vocab)  # full vocab, NO grammar mask
        if not scores:
            break
        step = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
        seq.append(step)
        if step == _TERMINAL:
            break
    return seq


def invalid_emission_rate(
    ranker, seqs_by_family: Dict[str, List[List[str]]], cand_vocab: Sequence[str],
    cut_frac: float = 0.5, max_len: int = 220,
) -> Dict[str, float]:
    """Free-generate a completion for each held cut and measure invalidity.

    Returns ``{invalid_rate, mean_violations, n}`` where ``invalid_rate`` is the
    fraction of free completions with >=1 rule violation and ``mean_violations``
    is the average number of violations per completion (the "invalid mass").
    """
    n = 0
    n_invalid = 0
    total_viol = 0
    for fam, seqs in seqs_by_family.items():
        for seq in seqs:
            if len(seq) < 4:
                continue
            cut = max(1, min(len(seq) - 1, round(len(seq) * cut_frac)))
            prefix = seq[:cut]
            gen = free_generate(prefix, fam, ranker, cand_vocab, max_len=max_len)
            viols = rules.validate_with_roles(gen)
            n += 1
            if viols:
                n_invalid += 1
                total_viol += len(viols)
    return {
        "invalid_rate": (n_invalid / n) if n else 0.0,
        "mean_violations": (total_viol / n) if n else 0.0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Constrained Top-1 (production decode) scored via the official scorer
# ---------------------------------------------------------------------------
def _top1(
    ranker, seqs_by_family: Dict[str, List[List[str]]], cand: List[str],
    split_dir: Path, tag: str, use_roles: bool,
) -> Dict[str, float]:
    """Score next-step (constrained decode) on a simulated eval set; return metrics."""
    eset = simulate_eval.build_eval_set(seqs_by_family, split_dir, prefix=tag)
    ns_pred = split_dir / f"{tag}_nextstep_pred.csv"
    predict.predict_nextstep(eset["nextstep_input"], ns_pred, ranker, cand, use_roles=use_roles)
    ns = nspe_eval.score("next-step", eset["nextstep_gt"], ns_pred)
    return {k: ns[k] for k in METRIC_KEYS["next-step"] if k in ns}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run(
    config_path: Optional[str] = None,
    holdout: str = "ic",
    smoke: bool = False,
    seed: int = 0,
    n_per_family: int = 60,
    use_roles: bool = True,
) -> Dict:
    """Run the 4-cell ablation and write ``$NSPE_OUT/exp04.json``."""
    _seed_everything(seed)
    out_dir = _out_dir()
    base_cfg = _load_config(config_path)
    base_cfg.setdefault("seed", seed)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    # Lazy torch-using import (training).
    from nspe.model import load_ranker, train_ranker

    holdout = holdout.lower()
    matches = [h for h in LOFO_SPLITS if h[0] == holdout]
    if not matches:
        raise ValueError(f"--holdout {holdout!r} not in {[h[0] for h in LOFO_SPLITS]}")
    held, train = matches[0][0], list(matches[0][1])

    if smoke:
        n_per_family = min(n_per_family, 6)
        # In smoke mode the heavy semantic mask cache is capped inside the model;
        # still cut the grid to keep it fast but exercise all four cells cheaply.
        base_cfg.setdefault("steps", 60)

    cand = list(candidate_vocab(train))  # OOD-correct candidate vocab (train steps only)

    # Shared eval slices (same data for every cell so cells are comparable).
    id_seqs = {f: [list(s) for s in load_family(f)[:n_per_family]] for f in train}
    ood_seqs = {held: [list(s) for s in load_family(held)[:n_per_family]]}
    # A held-out free-generation slice (use the OOD family — the hard case).
    free_seqs = ood_seqs

    t0 = time.time()
    cells: List[Dict] = []
    for i, ablate in enumerate(GRID):
        cfg = dict(base_cfg)
        cfg.update(ablate)
        cell_dir = out_dir / f"exp04_cell{i}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        res = train_ranker(train, config=cfg, out_dir=str(cell_dir), holdout=held, smoke=smoke)
        ranker = load_ranker(res["ckpt_path"])

        free = invalid_emission_rate(ranker, free_seqs, cand)
        id_ns = _top1(ranker, id_seqs, cand, cell_dir, "id", use_roles)
        ood_ns = _top1(ranker, ood_seqs, cand, cell_dir, "ood", use_roles)

        cell = {
            "mask_train": ablate["mask_train"],
            "sem_w": ablate["sem_w"],
            "n_params": res["metrics"]["n_params"],
            "train_wall_sec": res["metrics"]["wall_sec"],
            "final_loss": res["metrics"]["final_loss"],
            "free_generation": free,                 # invalid-emission rate / mass
            "id_next_step": id_ns,
            "ood_next_step": ood_ns,
        }
        cells.append(cell)
        print(f"[cell {i}] mask_train={ablate['mask_train']} sem_w={ablate['sem_w']} "
              f"-> free_invalid={free['invalid_rate']:.3f} "
              f"mean_viol={free['mean_violations']:.3f} "
              f"ID_top1={id_ns.get('top1')} OOD_top1={ood_ns.get('top1')}")

    # ---- contrast: does sem_w reduce invalid mass (mask_train held fixed)? ----
    def _cell(mt: bool, sw: float):
        return next(c for c in cells if c["mask_train"] == mt and c["sem_w"] == sw)

    contrast = {}
    for mt in (False, True):
        c0 = _cell(mt, 0.0)
        c1 = _cell(mt, 0.5)
        contrast[f"mask_train={mt}"] = {
            "invalid_rate_sem0": c0["free_generation"]["invalid_rate"],
            "invalid_rate_sem0.5": c1["free_generation"]["invalid_rate"],
            "invalid_rate_reduction": c0["free_generation"]["invalid_rate"]
            - c1["free_generation"]["invalid_rate"],
            "mean_violations_sem0": c0["free_generation"]["mean_violations"],
            "mean_violations_sem0.5": c1["free_generation"]["mean_violations"],
            "mean_violations_reduction": c0["free_generation"]["mean_violations"]
            - c1["free_generation"]["mean_violations"],
            "ood_top1_sem0": c0["ood_next_step"].get("top1"),
            "ood_top1_sem0.5": c1["ood_next_step"].get("top1"),
        }

    result = {
        "experiment": "exp04_constraint_loss",
        "holdout": held,
        "train_families": train,
        "device": device,
        "smoke": smoke,
        "base_config": base_cfg,
        "n_per_family": n_per_family,
        "cells": cells,
        "semantic_loss_contrast": contrast,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = out_dir / "exp04.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    result["json_path"] = str(out_json)

    print("=" * 72)
    print(f"exp04 semantic-loss contrast (holdout={held}, device={device}):")
    for k, v in contrast.items():
        print(f"  {k}: invalid_rate {v['invalid_rate_sem0']:.3f} -> "
              f"{v['invalid_rate_sem0.5']:.3f} (reduction {v['invalid_rate_reduction']:+.3f}); "
              f"OOD top1 {v['ood_top1_sem0']} -> {v['ood_top1_sem0.5']}")
    print(f"json -> {out_json}")
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML base config (shared across cells)")
    p.add_argument("--holdout", default="ic", help="the single LoFO family (mosfet|igbt|ic)")
    p.add_argument("--smoke", action="store_true", help="tiny fast run for CI / pipeline check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-per-family", type=int, default=60)
    p.add_argument("--no-roles", action="store_true", help="disable role-sharpened decoding")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        config_path=args.config,
        holdout=args.holdout,
        smoke=args.smoke,
        seed=args.seed,
        n_per_family=args.n_per_family,
        use_roles=not args.no_roles,
    )
