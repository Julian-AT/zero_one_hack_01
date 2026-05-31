#!/usr/bin/env python3
"""
neurosymbolic_adapter.py — run the NSPE (neurosymbolic) approach on the common
benchmark eval set and emit organizer-format predictions.

Default ranker is the pure-symbolic PPM (variable-order role-factored Markov model,
0 trainable parameters, fits in milliseconds) — fully reproducible, no weights, no GPU.
The symbolic grammar constrains every candidate, so completions are rule-valid by
construction and the anomaly oracle is exact on the 10 known rules.

LoFO: fit the ranker only on `--train-families` (the families NOT held out) and turn on
`--use-roles` so role-induction carries detection/ranking to the unseen family's vocabulary.

Usage (from repo root):
  PYTHONPATH=models/neurosymbolic python shared/benchmark/adapters/neurosymbolic_adapter.py \
      --eval-set shared/benchmark/eval_set_v1 \
      --out shared/benchmark/predictions/neurosymbolic \
      --train-families mosfet igbt ic
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models" / "neurosymbolic"))

from nspe import predict as NP  # noqa: E402
from nspe.data import candidate_vocab, load_family  # noqa: E402
from nspe.ppm import PPM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="shared/benchmark/eval_set_v1")
    ap.add_argument("--out", default="shared/benchmark/predictions/neurosymbolic")
    ap.add_argument("--train-families", nargs="+", default=["mosfet", "igbt", "ic"],
                    help="families the ranker is fit on (exclude the held-out one for LoFO)")
    ap.add_argument("--use-roles", action="store_true",
                    help="enable role-induction (the OOD lever; turn on for LoFO)")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--completion-max-len", type=int, default=160)
    ap.add_argument("--tasks", nargs="+", default=["next-step", "completion", "anomaly"],
                    choices=["next-step", "completion", "anomaly"])
    a = ap.parse_args()

    fams = tuple(f.lower() for f in a.train_families)
    print(f"[neurosymbolic] fit PPM on {fams}  use_roles={a.use_roles}")
    train = {f: [list(s) for s in load_family(f)] for f in fams}
    ranker = PPM().fit(train)
    cand = list(candidate_vocab(fams))

    inp = Path(a.eval_set) / "inputs"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if "next-step" in a.tasks:
        t0 = time.time()
        n = NP.predict_nextstep(inp / "nextstep_input.csv", out / "predictions_nextstep.csv",
                                ranker, cand, use_roles=a.use_roles)
        print(f"  wrote next-step ({n} rows, {time.time()-t0:.1f}s)")

    if "completion" in a.tasks:
        t0 = time.time()
        n = NP.predict_completion(inp / "completion_input.csv", out / "predictions_completion.csv",
                                  ranker, cand, use_roles=a.use_roles, beam=a.beam,
                                  max_len=a.completion_max_len)
        print(f"  wrote completion ({n} rows, {time.time()-t0:.1f}s)")

    if "anomaly" in a.tasks:
        t0 = time.time()
        # Pure-symbolic role-augmented oracle (ranker=None); use_roles on so the
        # rule checker generalizes to held-out-family vocabulary.
        n = NP.predict_anomaly(inp / "anomaly_input.csv", out / "predictions_anomaly.csv",
                               ranker=None, use_roles=True)
        print(f"  wrote anomaly ({n} rows, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
