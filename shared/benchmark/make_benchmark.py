#!/usr/bin/env python3
"""
make_benchmark.py — the unified cross-model benchmark driver.

Runs every model approach through one adapter each, on ONE common eval set, across two
regimes (ID and Leave-One-Family-Out / OOD), scores everything with the OFFICIAL metrics
(`competition/participant-files/eval_metrics.py`, via `score.py`), and writes a tidy
results table + an aggregated ID→OOD summary.

Models under test
    transformer_xlstm   decoder transformer + multitask heads (the submission model)
    self_supervised     SSL hybrid process transformer (semantic-feature embeddings)
    neurosymbolic       PPM + symbolic grammar/oracle (+ role induction for OOD)
    trigram             n-gram-with-backoff baseline (memorization floor)
    grammar             trigram + symbolic grammar mask

Regimes
    ID                  model sees all 3 families; scored on the full common eval set
    LoFO_<family>       model trained on the OTHER two families; scored ONLY on the
                        held-out family's slice (the track's hidden-4th-family proxy)

Prereqs (built by the sibling scripts; see README):
    shared/benchmark/eval_set_v1/                         (make_eval_set.py)
    shared/benchmark/checkpoints/bench-tr-small-*/final.pt        (train, transformer_xlstm)
    shared/benchmark/ssl_checkpoints/ssl-*/checkpoint_best.pt     (train, SSL hybrid)

Usage:
    python shared/benchmark/make_benchmark.py                  # all models, ID + LoFO
    python shared/benchmark/make_benchmark.py --models neurosymbolic trigram
    python shared/benchmark/make_benchmark.py --reuse          # skip adapters if preds exist
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared" / "benchmark"))
import importlib.util

import score as S  # noqa: E402


def _load_validator():
    """validate_sequence from the official generator → completion rule-validity metric."""
    gen = ROOT / "competition" / "track-details" / "training_data" / "generate_sequences.py"
    spec = importlib.util.spec_from_file_location("generate_sequences_val", gen)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return lambda seq: len(mod.validate_sequence(list(seq)))  # violation count


VALIDATOR = _load_validator()

PY = sys.executable
ADAPTERS = ROOT / "shared" / "benchmark" / "adapters"
EVAL_SET = ROOT / "shared" / "benchmark" / "eval_set_v1"
PRED_ROOT = ROOT / "shared" / "benchmark" / "predictions"
TXL_CKPT = ROOT / "shared" / "benchmark" / "checkpoints"
SSL_CKPT = ROOT / "shared" / "benchmark" / "ssl_checkpoints"

FAMILIES = ["mosfet", "igbt", "ic"]
# regime -> (eval_set_dir, train_families, txl_run_suffix, ssl_run_name)
REGIMES = {
    "ID": (EVAL_SET, FAMILIES, "all3", "ssl-all3"),
    "LoFO_mosfet": (EVAL_SET / "by_family" / "mosfet", ["igbt", "ic"], "held_mosfet", "ssl-held_mosfet"),
    "LoFO_igbt": (EVAL_SET / "by_family" / "igbt", ["mosfet", "ic"], "held_igbt", "ssl-held_igbt"),
    "LoFO_ic": (EVAL_SET / "by_family" / "ic", ["mosfet", "igbt"], "held_ic", "ssl-held_ic"),
}

# model -> kind
MODELS = {
    "transformer_xlstm": "txl",
    "self_supervised": "ssl",
    "neurosymbolic": "nspe",
    "trigram": "baseline",
    "grammar": "baseline",
}

TASKS = {
    "next-step": ("predictions_nextstep.csv", "nextstep_gt.csv", S.score_nextstep),
    "completion": ("predictions_completion.csv", "completion_gt.csv", S.score_completion),
    "anomaly": ("predictions_anomaly.csv", "anomaly_gt.csv", S.score_anomaly),
}


def run(cmd: list[str], env_extra: dict | None = None) -> bool:
    env = dict(os.environ)
    env.setdefault("PROCESS_LOGIC_DEVICE", "cpu")
    env.setdefault("OMP_NUM_THREADS", "4")
    if env_extra:
        env.update(env_extra)
    print("    $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env)
    return r.returncode == 0


def adapter_cmd(model: str, regime: str, out_dir: Path, completion_max_len: int):
    """Return (cmd, env_extra) or None if the required checkpoint is missing."""
    eset, fams, txl_suffix, ssl_name = REGIMES[regime]
    kind = MODELS[model]
    if kind == "txl":
        ckpt = TXL_CKPT / f"bench-tr-small-{txl_suffix}" / "final.pt"
        if not ckpt.exists():
            return None
        return ([PY, ADAPTERS / "transformer_xlstm_adapter.py", "--eval-set", eset,
                 "--checkpoint", ckpt, "--out", out_dir, "--device", "cpu",
                 "--completion-max-len", completion_max_len],
                {"PYTHONPATH": str(ROOT / "models")})
    if kind == "ssl":
        d = SSL_CKPT / ssl_name
        ckpt, vocab = d / "checkpoint_best.pt", d / "vocab.json"
        if not ckpt.exists():
            return None
        return ([PY, ADAPTERS / "transformer_adapter.py", "--eval-set", eset,
                 "--out", out_dir, "--checkpoint", ckpt, "--vocab", vocab,
                 "--tasks", "next-step", "completion", "anomaly"],
                {"PYTHONPATH": str(ROOT / "models")})
    if kind == "nspe":
        cmd = [PY, ADAPTERS / "neurosymbolic_adapter.py", "--eval-set", eset,
               "--out", out_dir, "--train-families", *fams,
               "--completion-max-len", completion_max_len]
        if regime != "ID":
            cmd.append("--use-roles")
        return (cmd, {"PYTHONPATH": str(ROOT / "models" / "neurosymbolic")})
    if kind == "baseline":
        return ([PY, ADAPTERS / "baseline_adapter.py", "--eval-set", eset,
                 "--kind", model, "--out", out_dir, "--train-families", *fams,
                 "--completion-max-len", completion_max_len], None)
    return None


def score_dir(pred_dir: Path, gt_dir: Path) -> dict:
    """Score whichever prediction files exist. Returns {task: {metric: val, 'n': n}}."""
    out: dict[str, dict] = {}
    for task, (pred_name, gt_name, fn) in TASKS.items():
        pred_csv, gt_csv = pred_dir / pred_name, gt_dir / gt_name
        if not pred_csv.exists() or not gt_csv.exists():
            continue
        if task == "completion":
            res = fn(str(pred_csv), str(gt_csv), validator=VALIDATOR)
        else:
            res = fn(str(pred_csv), str(gt_csv))
        row = dict(res["overall"])
        row["n"] = res["n"]
        out[task] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument("--reuse", action="store_true", help="skip adapter run if predictions exist")
    ap.add_argument("--completion-max-len", type=int, default=80)
    ap.add_argument("--out", default="shared/benchmark/results_long.csv")
    a = ap.parse_args()

    rows: list[dict] = []          # model, regime, task, metric, value, n
    skipped: list[str] = []
    for model in a.models:
        for regime in a.regimes:
            eset, fams, _, _ = REGIMES[regime]
            out_dir = PRED_ROOT / regime / model
            preds_present = (out_dir / "predictions_nextstep.csv").exists()
            if not (a.reuse and preds_present):
                spec = adapter_cmd(model, regime, out_dir, a.completion_max_len)
                if spec is None:
                    skipped.append(f"{model}/{regime} (missing checkpoint)")
                    continue
                print(f"\n### {model} | {regime}")
                if not run(spec[0], spec[1]):
                    skipped.append(f"{model}/{regime} (adapter failed)")
                    continue
            scored = score_dir(out_dir, eset / "ground_truth")
            for task, metrics in scored.items():
                n = metrics.pop("n", "")
                for metric, val in metrics.items():
                    rows.append({"model": model, "regime": regime, "task": task,
                                 "metric": metric, "value": round(float(val), 4)
                                 if val == val else "nan", "n": n})

    out_csv = ROOT / a.out
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "regime", "task", "metric", "value", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}  ({len(rows)} rows)")
    if skipped:
        print("SKIPPED:")
        for s in skipped:
            print("  -", s)


if __name__ == "__main__":
    main()
