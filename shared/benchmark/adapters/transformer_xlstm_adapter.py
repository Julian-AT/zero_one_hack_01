#!/usr/bin/env python3
"""
transformer_xlstm_adapter.py — run the transformer_xlstm model approach on the common
benchmark eval set and emit organizer-format predictions.

Uses the canonical inference API in `transformer_xlstm.eval.predict`
(load_model / topk_next_step / complete_sequence / anomaly_ensemble), so it works with
any checkpoint produced by `transformer_xlstm.train.trainer` (transformer OR xlstm arch,
step OR compositional tokenization, LM-only OR multitask heads).

Usage (from repo root):
  PYTHONPATH=models python shared/benchmark/adapters/transformer_xlstm_adapter.py \
      --eval-set shared/benchmark/eval_set_v1 \
      --checkpoint shared/extras/checkpoints/bench-transformer-small-all3/final.pt \
      --out shared/benchmark/predictions/transformer_xlstm \
      --device auto
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import torch  # noqa: E402
from transformer_xlstm.eval import predict as P  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="shared/benchmark/eval_set_v1")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="shared/benchmark/predictions/transformer_xlstm")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tasks", nargs="+", default=["next-step", "completion", "anomaly"],
                    choices=["next-step", "completion", "anomaly"])
    ap.add_argument("--completion-max-len", type=int, default=90,
                    help="max steps to greedily decode in completion (bounds cost)")
    ap.add_argument("--no-grammar", action="store_true",
                    help="disable grammar masking (ablation)")
    a = ap.parse_args()

    device = pick_device(a.device)
    print(f"[transformer_xlstm] checkpoint={a.checkpoint}  device={device}")
    lm = P.load_model(Path(a.checkpoint), device=device)
    grammar = not a.no_grammar
    inp = Path(a.eval_set) / "inputs"
    out = Path(a.out)

    if "next-step" in a.tasks:
        rows = C.read_csv(inp / "nextstep_input.csv")
        t0 = time.time()
        results = []
        for i, r in enumerate(rows):
            ranks = P.topk_next_step(lm, r["FAMILY"], C.split_steps(r["PARTIAL_SEQUENCE"]),
                                     k=5, grammar=grammar)
            results.append((r["EXAMPLE_ID"], ranks))
            if i % 200 == 0:
                print(f"  next-step {i}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
        C.write_nextstep(out, results)
        print(f"  wrote next-step ({len(results)} rows, {time.time()-t0:.0f}s)")

    if "completion" in a.tasks:
        rows = C.read_csv(inp / "completion_input.csv")
        t0 = time.time()
        results = []
        for i, r in enumerate(rows):
            comp = P.complete_sequence(lm, r["FAMILY"], C.split_steps(r["PARTIAL_SEQUENCE"]),
                                       max_len=a.completion_max_len, grammar=grammar)
            results.append((r["EXAMPLE_ID"], comp))
            if i % 20 == 0:
                print(f"  completion {i}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
        C.write_completion(out, results)
        print(f"  wrote completion ({len(results)} rows, {time.time()-t0:.0f}s)")

    if "anomaly" in a.tasks:
        rows = C.read_csv(inp / "anomaly_input.csv")
        t0 = time.time()
        results = []
        for r in rows:
            v = P.anomaly_ensemble(lm, r["FAMILY"], C.split_steps(r["SEQUENCE"]))
            results.append((r["EXAMPLE_ID"], v["IS_VALID"], v["SCORE"], v["PREDICTED_RULE"]))
        C.write_anomaly(out, results)
        print(f"  wrote anomaly ({len(results)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
