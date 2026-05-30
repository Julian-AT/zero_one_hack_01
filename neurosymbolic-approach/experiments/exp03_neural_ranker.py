"""exp03 — constrained neural ranker on Tasks 1 & 2: ID + LoFO vs the PPM baseline.

This experiment trains the small, role-factored constrained neural ranker
(``nspe.model.train_ranker``) on the training families and evaluates next-step
(Task 1) and completion (Task 2) **on exactly the same locally-simulated eval
sets** as a pure-symbolic PPM baseline (``nspe.ppm.PPM``). It does this both
in-distribution (ID) and, when ``--holdout`` names a family, out-of-distribution
(OOD / Leave-One-Family-Out) on the held-out family. The headline number is the
ID->OOD *drop*: the neurosymbolic thesis is that constrained, role-factored
decoding makes that drop far flatter than a pure-neural scaffold's.

Why a local eval set? The organizers' sample eval inputs carry no answers, so
``nspe.simulate_eval`` reconstructs official-format ground truth from held-out
*full* sequences and ``nspe.eval.score`` scores them with the official
``eval_metrics.py`` subprocess. Both rankers see an identical candidate vocab
(the TRAIN families' steps) and identical eval sets, so the comparison is fair.

Modes
-----
* ``--holdout none`` (default): ID-only. Train on all three families, build an ID
  eval set on a slice of the same families, score neural vs PPM.
* ``--holdout <family>``: LoFO. Train on the other two families, build BOTH an ID
  eval set (train families) and an OOD eval set (the held-out family), score both
  rankers on both, and report the ID->OOD drop per metric for each ranker.

Output: ``$NSPE_OUT/exp03_<holdout>.json`` (``<holdout>`` is ``none`` for ID-only).

GPU experiment (runs on CPU too). Seeded; ``device = cuda if available else cpu``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# yaml is optional at import time; only required if --config is given.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is in the env per the spec
    yaml = None

from nspe import eval as nspe_eval
from nspe import predict, simulate_eval
from nspe.data import LOFO_SPLITS, FAMILIES, candidate_vocab, load_family

__all__ = ["run", "build_neural_make_ranker", "build_ppm_make_ranker"]

METRIC_KEYS = nspe_eval.METRIC_KEYS


# ---------------------------------------------------------------------------
# Config / IO helpers
# ---------------------------------------------------------------------------
def _out_dir() -> Path:
    out = Path(os.environ.get("NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_config(path: Optional[str]) -> Dict:
    """Load a YAML config (the model-hyperparameter dict). Empty dict if none."""
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not available but --config was provided")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping, got {type(cfg)}")
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch  # local: keep the symbolic import surface torch-free
        torch.manual_seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ranker factories (duck-typed: each maps train_families -> ranker)
# ---------------------------------------------------------------------------
def build_ppm_make_ranker(n_per_family: int):
    """Return ``make_ranker(train_families) -> PPM`` fit on the train families."""
    from nspe.ppm import PPM

    def make_ranker(train_families: Sequence[str]):
        seqs = {f: [list(s) for s in load_family(f)[:n_per_family]] for f in train_families}
        return PPM().fit(seqs)

    return make_ranker


def build_neural_make_ranker(
    config: Dict, out_dir: Path, holdout: Optional[str], smoke: bool,
    max_steps: Optional[int],
):
    """Return ``make_ranker(train_families) -> NeuralRanker``.

    Trains a fresh constrained ranker per call (LoFO trains one per split) and
    reloads it through the protocol-compliant ``NeuralRanker`` wrapper. Records
    the training metrics on the returned ranker as ``_train_metrics`` for the
    caller to harvest. torch is imported lazily here so the module stays
    importable without torch present.
    """
    from nspe.model import load_ranker, train_ranker

    cfg = dict(config)
    if max_steps is not None:
        cfg["steps"] = int(max_steps)

    train_log: Dict[str, Dict] = {}

    def make_ranker(train_families: Sequence[str]):
        train_families = [f.lower() for f in train_families]
        res = train_ranker(
            train_families, config=cfg, out_dir=str(out_dir), holdout=holdout,
            smoke=smoke,
        )
        ranker = load_ranker(res["ckpt_path"])
        # Stash training metrics for the experiment report.
        ranker._train_metrics = res["metrics"]  # type: ignore[attr-defined]
        train_log["+".join(train_families)] = res["metrics"]
        return ranker

    make_ranker.train_log = train_log  # type: ignore[attr-defined]
    return make_ranker


# ---------------------------------------------------------------------------
# Paired scoring: both rankers on ONE shared eval set
# ---------------------------------------------------------------------------
def _score_ranker_on_set(
    name: str, ranker, eset: Dict, cand: List[str], split_dir: Path,
    use_roles: bool, beam: int,
) -> Dict:
    """Predict next-step + completion for one ranker on a prebuilt eval set and
    score both via the official scorer. Returns ``{"next-step":{...}, "completion":{...}}``."""
    ns_pred = split_dir / f"{name}_nextstep_pred.csv"
    predict.predict_nextstep(eset["nextstep_input"], ns_pred, ranker, cand, use_roles=use_roles)
    ns = nspe_eval.score("next-step", eset["nextstep_gt"], ns_pred)

    cp_pred = split_dir / f"{name}_completion_pred.csv"
    predict.predict_completion(eset["completion_input"], cp_pred, ranker, cand,
                               use_roles=use_roles, beam=beam)
    cp = nspe_eval.score("completion", eset["completion_gt"], cp_pred)
    return {
        "next-step": {k: ns[k] for k in METRIC_KEYS["next-step"] if k in ns},
        "completion": {k: cp[k] for k in METRIC_KEYS["completion"] if k in cp},
    }


def _drop(id_m: Dict, ood_m: Dict, keys: Sequence[str]) -> Dict:
    return {k: id_m[k] - ood_m[k] for k in keys if k in id_m and k in ood_m}


def _compare_on_split(
    held: Optional[str], train: Sequence[str], neural, ppm,
    out_dir: Path, n_per_family: int, use_roles: bool, beam: int,
) -> Dict:
    """Build ID (+OOD if held) eval sets ONCE and score both rankers on them."""
    tag = held or "none"
    split_dir = out_dir / f"exp03_{tag}_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    cand = list(candidate_vocab(train))  # OOD-correct: only TRAIN-family steps

    id_seqs = {f: [list(s) for s in load_family(f)[:n_per_family]] for f in train}
    id_set = simulate_eval.build_eval_set(id_seqs, split_dir, prefix="id")

    record: Dict = {"held_out": held, "train": list(train), "n_per_family": n_per_family}
    record["neural"] = {"id": _score_ranker_on_set("neural_id", neural, id_set, cand,
                                                    split_dir, use_roles, beam)}
    record["ppm"] = {"id": _score_ranker_on_set("ppm_id", ppm, id_set, cand,
                                                 split_dir, use_roles, beam)}

    if held is not None:
        ood_seqs = {held: [list(s) for s in load_family(held)[:n_per_family]]}
        ood_set = simulate_eval.build_eval_set(ood_seqs, split_dir, prefix="ood")
        record["neural"]["ood"] = _score_ranker_on_set("neural_ood", neural, ood_set, cand,
                                                        split_dir, use_roles, beam)
        record["ppm"]["ood"] = _score_ranker_on_set("ppm_ood", ppm, ood_set, cand,
                                                     split_dir, use_roles, beam)
        for who in ("neural", "ppm"):
            record[who]["drop"] = {
                t: _drop(record[who]["id"][t], record[who]["ood"][t], METRIC_KEYS[t])
                for t in ("next-step", "completion")
            }
    return record


# ---------------------------------------------------------------------------
# Experiment entry point
# ---------------------------------------------------------------------------
def run(
    config_path: Optional[str] = None,
    holdout: str = "none",
    smoke: bool = False,
    max_steps: Optional[int] = None,
    seed: int = 0,
    n_per_family: int = 80,
    use_roles: bool = True,
    beam: int = 1,
) -> Dict:
    """Train the neural ranker and compare to PPM on Tasks 1 & 2 (ID + LoFO).

    Returns the result dict and writes ``$NSPE_OUT/exp03_<holdout>.json``.
    """
    _seed_everything(seed)
    out_dir = _out_dir()
    cfg = _load_config(config_path)
    cfg.setdefault("seed", seed)
    if smoke:
        n_per_family = min(n_per_family, 8)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    make_neural = build_neural_make_ranker(cfg, out_dir, None if holdout == "none" else holdout,
                                           smoke, max_steps)
    make_ppm = build_ppm_make_ranker(n_per_family=1000 if not smoke else 60)

    t0 = time.time()
    if holdout == "none":
        train = list(FAMILIES)
        held = None
    else:
        holdout = holdout.lower()
        matches = [h for h in LOFO_SPLITS if h[0] == holdout]
        if not matches:
            raise ValueError(f"--holdout {holdout!r} not in {[h[0] for h in LOFO_SPLITS]}")
        held, train = matches[0][0], list(matches[0][1])

    neural = make_neural(train)
    ppm = make_ppm(train)
    record = _compare_on_split(held, train, neural, ppm, out_dir, n_per_family,
                               use_roles, beam)

    result = {
        "experiment": "exp03_neural_ranker",
        "holdout": holdout,
        "device": device,
        "smoke": smoke,
        "config": cfg,
        "train_metrics": getattr(neural, "_train_metrics", None),
        "wall_sec": round(time.time() - t0, 2),
        "comparison": record,
    }

    out_json = out_dir / f"exp03_{holdout}.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    result["json_path"] = str(out_json)

    # ---- console summary ----
    print("=" * 72)
    print(f"exp03  holdout={holdout}  device={device}  smoke={smoke}")
    tm = result["train_metrics"] or {}
    print(f"neural params={tm.get('n_params')}  steps={tm.get('steps_run')}  "
          f"final_loss={tm.get('final_loss')}  train_wall={tm.get('wall_sec')}s")
    for who in ("neural", "ppm"):
        print(f"-- {who} --")
        print(f"   ID  next-step  : {record[who]['id']['next-step']}")
        print(f"   ID  completion : {record[who]['id']['completion']}")
        if held is not None:
            print(f"   OOD next-step  : {record[who]['ood']['next-step']}")
            print(f"   OOD completion : {record[who]['ood']['completion']}")
            print(f"   DROP next-step : {record[who]['drop']['next-step']}")
            print(f"   DROP completion: {record[who]['drop']['completion']}")
    print(f"json -> {out_json}")
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML model-hyperparameter config")
    p.add_argument("--holdout", default="none",
                   help="held-out family for LoFO (mosfet|igbt|ic) or 'none' for ID-only")
    p.add_argument("--smoke", action="store_true", help="tiny fast run for CI / pipeline check")
    p.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-per-family", type=int, default=80,
                   help="sequences per family in the eval sets")
    p.add_argument("--no-roles", action="store_true", help="disable role-sharpened decoding")
    p.add_argument("--beam", type=int, default=1, help="completion beam width")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        config_path=args.config,
        holdout=args.holdout,
        smoke=args.smoke,
        max_steps=args.max_steps,
        seed=args.seed,
        n_per_family=args.n_per_family,
        use_roles=not args.no_roles,
        beam=args.beam,
    )
