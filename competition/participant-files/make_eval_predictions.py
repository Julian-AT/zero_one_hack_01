#!/usr/bin/env python3
"""
make_eval_predictions.py

Creates organizer-format prediction files:

  competition/participant-files/predictions/predictions_nextstep.csv
  competition/participant-files/predictions/predictions_completion.csv
  competition/participant-files/predictions/predictions_anomaly.csv

Run from repo root:

  source competition/track-details/.venv/bin/activate
  python competition/participant-files/make_eval_predictions.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "models") not in sys.path:
    sys.path.insert(0, str(ROOT / "models"))

from transformer_xlstm.data.sequence_io import norm as norm_step
from transformer_xlstm.data.sequence_io import read_csv, split_steps

TRACK_DIR = ROOT / "competition" / "track-details"
TRAIN_SCRIPT = TRACK_DIR / "scripts" / "train_ssl_hybrid_process_transformer.py"
GEN_SCRIPT = TRACK_DIR / "training_data" / "generate_sequences.py"

DEFAULT_RUN_DIR = TRACK_DIR / "runs" / "ssl_hybrid_new_coverage_guided_v1"

EVAL_VALID = ROOT / "competition" / "participant-files" / "eval_input_valid.csv"
EVAL_ANOMALY = ROOT / "competition" / "participant-files" / "eval_input_anomaly.csv"

OUT_DIR = ROOT / "competition" / "participant-files" / "predictions"
OUT_NEXT = OUT_DIR / "predictions_nextstep.csv"
OUT_COMPLETION = OUT_DIR / "predictions_completion.csv"
OUT_ANOMALY = OUT_DIR / "predictions_anomaly.csv"


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


train_mod = load_module(TRAIN_SCRIPT, "ssl_hybrid_train_mod")
gen_mod = load_module(GEN_SCRIPT, "generate_sequences_mod")


def build_token_feature_table(token_to_id, feature_to_id):
    feature_names = train_mod.FEATURE_NAMES
    table = torch.zeros((len(token_to_id), len(feature_names)), dtype=torch.long)

    for tok, tid in token_to_id.items():
        fd = train_mod.feature_dict_for_token(tok)
        for j, name in enumerate(feature_names):
            unknown_id = feature_to_id[name].get("UNKNOWN", 0)
            table[tid, j] = feature_to_id[name].get(fd[name], unknown_id)

    return table


def maybe_load_vocab_json(vocab_path: Path):
    if not vocab_path.exists():
        return None
    with vocab_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class HybridPredictor:
    def __init__(self, checkpoint_path: Path, vocab_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=self.device)

        vocab_json = maybe_load_vocab_json(vocab_path)

        self.cfg = ckpt["config"]
        self.token_to_id = ckpt["token_to_id"]
        self.family_to_id = ckpt["family_to_id"]
        self.feature_to_id = ckpt["feature_to_id"]

        if vocab_json is not None and "id_to_token" in vocab_json:
            self.id_to_token = {int(k): v for k, v in vocab_json["id_to_token"].items()}
        else:
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}

        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.mask_id = self.token_to_id["<MASK>"]

        feature_vocab_sizes = [len(self.feature_to_id[name]) for name in train_mod.FEATURE_NAMES]

        self.model = train_mod.HybridProcessTransformerLM(
            vocab_size=len(self.token_to_id),
            num_families=len(self.family_to_id),
            feature_vocab_sizes=feature_vocab_sizes,
            max_len=int(self.cfg["max_len"]),
            d_model=int(self.cfg["d_model"]),
            layers=int(self.cfg["layers"]),
            heads=int(self.cfg["heads"]),
            ff_mult=int(self.cfg["ff_mult"]),
            dropout=float(self.cfg["dropout"]),
            pad_id=self.pad_id,
        )

        self.model.load_state_dict(ckpt["model_state"])

        # Important: the training model ties lm_head to step_emb.
        # Re-tie after loading to match the intended architecture.
        self.model.lm_head.weight = self.model.step_emb.weight

        self.model.to(self.device)
        self.model.eval()

        self.token_feature_table = build_token_feature_table(
            self.token_to_id,
            self.feature_to_id,
        ).to(self.device)

        self.max_len = int(self.cfg["max_len"])

        self.special_ids = {
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.unk_id,
            self.mask_id,
        }

    def family_id(self, family: str) -> int:
        fam = norm_step(family)
        return self.family_to_id.get(fam, self.family_to_id.get("<FAM_UNKNOWN>", 0))

    def encode_context(self, steps: list[str]) -> list[int]:
        ids = [self.bos_id]
        ids += [self.token_to_id.get(norm_step(s), self.unk_id) for s in steps]

        if len(ids) > self.max_len:
            ids = [self.bos_id] + ids[-(self.max_len - 1) :]

        return ids

    @torch.no_grad()
    def next_logits(self, family: str, steps: list[str]) -> torch.Tensor:
        ids = self.encode_context(steps)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        feature_ids = self.token_feature_table[input_ids]
        family_ids = torch.tensor([self.family_id(family)], dtype=torch.long, device=self.device)

        logits = self.model(input_ids, feature_ids, family_ids)
        return logits[0, -1].detach().clone()

    def topk_next(self, family: str, steps: list[str], k: int = 5) -> list[str]:
        logits = self.next_logits(family, steps)

        for sid in self.special_ids:
            logits[sid] = -1e9

        k = min(k, logits.numel())
        top_ids = torch.topk(logits, k=k).indices.tolist()
        return [self.id_to_token[int(i)] for i in top_ids]

    def greedy_completion(
        self,
        family: str,
        partial_steps: list[str],
        max_new_steps: int = 160,
    ) -> list[str]:
        generated: list[str] = []
        context = list(partial_steps)

        for _ in range(max_new_steps):
            logits = self.next_logits(family, context)

            # Allow EOS for stopping, but not other special tokens.
            for sid in self.special_ids:
                if sid != self.eos_id:
                    logits[sid] = -1e9

            next_id = int(torch.argmax(logits).item())

            if next_id == self.eos_id:
                break

            step = self.id_to_token[next_id]
            generated.append(step)
            context.append(step)

        return generated


def make_nextstep_and_completion(predictor: HybridPredictor):
    rows = read_csv(EVAL_VALID)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (
        OUT_NEXT.open("w", newline="", encoding="utf-8") as f_next,
        OUT_COMPLETION.open("w", newline="", encoding="utf-8") as f_comp,
    ):
        next_writer = csv.writer(f_next)
        comp_writer = csv.writer(f_comp)

        next_writer.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        comp_writer.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])

        for r in rows:
            eid = r["EXAMPLE_ID"].strip()
            family = r["FAMILY"].strip()
            partial = split_steps(r["PARTIAL_SEQUENCE"])

            ranks = predictor.topk_next(family, partial, k=5)
            while len(ranks) < 5:
                ranks.append("")

            next_writer.writerow([eid] + ranks[:5])

            completion = predictor.greedy_completion(family, partial, max_new_steps=160)
            comp_writer.writerow([eid, "|".join(completion)])

    print(f"Wrote {OUT_NEXT}")
    print(f"Wrote {OUT_COMPLETION}")


def make_anomaly_predictions():
    rows = read_csv(EVAL_ANOMALY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_ANOMALY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])

        for r in rows:
            eid = r["EXAMPLE_ID"].strip()
            seq = split_steps(r["SEQUENCE"])

            violations = gen_mod.validate_sequence(seq)

            if violations:
                is_valid = 0
                score = 0.01
                predicted_rule = violations[0].rule
            else:
                is_valid = 1
                score = 0.99
                predicted_rule = ""

            writer.writerow([eid, is_valid, score, predicted_rule])

    print(f"Wrote {OUT_ANOMALY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory holding checkpoint_best.pt and vocab.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (defaults to <run-dir>/checkpoint_best.pt).",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="vocab.json path (defaults to <run-dir>/vocab.json).",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.run_dir / "checkpoint_best.pt"
    vocab = args.vocab or args.run_dir / "vocab.json"

    print(f"Using checkpoint: {checkpoint}")
    print(f"Using vocab:      {vocab if vocab.exists() else 'not found, reconstructing'}")
    print(f"Eval valid:       {EVAL_VALID}")
    print(f"Eval anomaly:     {EVAL_ANOMALY}")

    predictor = HybridPredictor(checkpoint, vocab)

    make_nextstep_and_completion(predictor)
    make_anomaly_predictions()

    print("\nDone.")
    print("Generated:")
    print(f"  {OUT_NEXT}")
    print(f"  {OUT_COMPLETION}")
    print(f"  {OUT_ANOMALY}")


if __name__ == "__main__":
    main()
