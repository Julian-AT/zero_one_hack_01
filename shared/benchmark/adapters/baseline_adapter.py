#!/usr/bin/env python3
"""
baseline_adapter.py — run a non-neural reference baseline on the common benchmark eval
set and emit organizer-format predictions. These are the "memorization floor" the
SUBMISSION rubric asks for (baseline-vs-trained comparison): any deep model that cannot
beat the trigram has not learned process logic.

Kinds:
  trigram   trigram-with-backoff (Counter n-gram counts; 0 parameters)
  grammar   the same trigram, but candidates are grammar-masked by the symbolic validator

Both produce next-step + completion only (no anomaly capability — left empty).
Pure CPU, no weights. LoFO: fit only on `--train-families`.

Usage (from repo root):
  python shared/benchmark/adapters/baseline_adapter.py \
      --eval-set shared/benchmark/eval_set_v1 --kind trigram \
      --out shared/benchmark/predictions/trigram --train-families mosfet igbt ic
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared" / "extras" / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import grammar_decoder as G  # noqa: E402
import trigram_baseline as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="shared/benchmark/eval_set_v1")
    ap.add_argument("--kind", choices=["trigram", "grammar"], default="trigram")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-families", nargs="+", default=["mosfet", "igbt", "ic"])
    ap.add_argument("--completion-max-len", type=int, default=140)
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / "shared" / "benchmark" / "predictions" / a.kind
    fams = {f.lower() for f in a.train_families}

    by_fam = T.load_family_sequences()
    train_seqs: list[list[str]] = []
    for fam, seqs in by_fam.items():
        if fam.lower() in fams:
            train_seqs.extend(seqs)
    print(f"[{a.kind}] fit on {sorted(fams)} -> {len(train_seqs)} sequences")
    model = T.TrigramBackoff()
    model.fit(train_seqs)

    inp = Path(a.eval_set) / "inputs"

    # next-step
    rows = C.read_csv(inp / "nextstep_input.csv")
    t0 = time.time()
    ns = []
    for r in rows:
        partial = C.split_steps(r["PARTIAL_SEQUENCE"])
        if a.kind == "grammar":
            ranks = G.grammar_filtered_topk(model, partial, k=5, pool_size=30)
        else:
            ctx2 = partial[-2] if len(partial) >= 2 else None
            ctx1 = partial[-1] if len(partial) >= 1 else None
            ranks = model.rank(ctx2, ctx1, k=5)
        ns.append((r["EXAMPLE_ID"], ranks))
    C.write_nextstep(out, ns)
    print(f"  wrote next-step ({len(ns)} rows, {time.time()-t0:.1f}s)")

    # completion
    rows = C.read_csv(inp / "completion_input.csv")
    t0 = time.time()
    cp = []
    for r in rows:
        partial = C.split_steps(r["PARTIAL_SEQUENCE"])
        if a.kind == "grammar":
            pred = G.grammar_complete(model, partial, max_len=a.completion_max_len)
        else:
            pred = model.complete(partial, max_len=a.completion_max_len)
        cp.append((r["EXAMPLE_ID"], pred))
    C.write_completion(out, cp)
    print(f"  wrote completion ({len(cp)} rows, {time.time()-t0:.1f}s)")
    print(f"  (no anomaly: {a.kind} baseline has no anomaly capability)")


if __name__ == "__main__":
    main()
