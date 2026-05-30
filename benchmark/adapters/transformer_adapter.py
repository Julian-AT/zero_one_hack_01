#!/usr/bin/env python3
"""
transformer_adapter.py — run the SSL hybrid Transformer on the common benchmark eval set
and emit organizer-format predictions for run_benchmark.py.

Reuses the HybridPredictor + validator from participant_files/make_eval_predictions.py, but
decouples the three tasks so each runs on its own input file (next-step on the large
nextstep_input.csv, greedy completion only on the small completion_input.csv).

Requires torch + the trained checkpoint, so run on Leonardo.

Usage (from repo root):
  python benchmark/adapters/transformer_adapter.py \
      --eval-set benchmark/eval_set_v1 \
      --out benchmark/predictions/ssl_transformer \
      --tasks next-step completion anomaly \
      --limit-nextstep 0           # 0 = all rows; set e.g. 5000 for a quick pass
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEP_PATH = ROOT / "participant_files" / "make_eval_predictions.py"


def load_mep():
    spec = importlib.util.spec_from_file_location("make_eval_predictions", MEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # triggers torch + train-script import
    return mod


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="benchmark/eval_set_v1")
    ap.add_argument("--out", default="benchmark/predictions/ssl_transformer")
    ap.add_argument("--tasks", nargs="+", default=["next-step", "completion", "anomaly"],
                    choices=["next-step", "completion", "anomaly"])
    ap.add_argument("--limit-nextstep", type=int, default=0, help="0 = all next-step rows")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint path")
    ap.add_argument("--vocab", default=None, help="override vocab.json path")
    a = ap.parse_args()

    mep = load_mep()
    ckpt = Path(a.checkpoint) if a.checkpoint else mep.CHECKPOINT
    vocab = Path(a.vocab) if a.vocab else mep.VOCAB_JSON
    print(f"checkpoint: {ckpt}")
    predictor = mep.HybridPredictor(ckpt, vocab)

    inp = Path(a.eval_set) / "inputs"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if "next-step" in a.tasks:
        rows = read_csv(inp / "nextstep_input.csv")
        if a.limit_nextstep > 0:
            rows = rows[: a.limit_nextstep]
        with (out / "predictions_nextstep.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
            for i, r in enumerate(rows):
                ranks = predictor.topk_next(r["FAMILY"], mep.split_steps(r["PARTIAL_SEQUENCE"]), k=5)
                ranks = (ranks + ["", "", "", "", ""])[:5]
                w.writerow([r["EXAMPLE_ID"]] + ranks)
                if i % 5000 == 0:
                    print(f"  next-step {i}/{len(rows)}")
        print(f"wrote {out / 'predictions_nextstep.csv'} ({len(rows)} rows)")

    if "completion" in a.tasks:
        rows = read_csv(inp / "completion_input.csv")
        with (out / "predictions_completion.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
            for i, r in enumerate(rows):
                comp = predictor.greedy_completion(r["FAMILY"], mep.split_steps(r["PARTIAL_SEQUENCE"]),
                                                   max_new_steps=160)
                w.writerow([r["EXAMPLE_ID"], "|".join(comp)])
                if i % 200 == 0:
                    print(f"  completion {i}/{len(rows)}")
        print(f"wrote {out / 'predictions_completion.csv'} ({len(rows)} rows)")

    if "anomaly" in a.tasks:
        rows = read_csv(inp / "anomaly_input.csv")
        with (out / "predictions_anomaly.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
            for r in rows:
                viol = mep.gen_mod.validate_sequence(mep.split_steps(r["SEQUENCE"]))
                if viol:
                    w.writerow([r["EXAMPLE_ID"], 0, 0.01, viol[0].rule])
                else:
                    w.writerow([r["EXAMPLE_ID"], 1, 0.99, ""])
        print(f"wrote {out / 'predictions_anomaly.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
