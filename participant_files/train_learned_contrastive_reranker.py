#!/usr/bin/env python3
"""
train_learned_contrastive_reranker.py

Learns a small contrastive reranker on top of the frozen SSL Transformer.

Pipeline:
  SSL model: prefix -> Top-5 next-step candidates
  learned reranker: prefix + candidate -> plausibility/correctness score
  output: reordered Top-5 predictions for official eval

Outputs:
  participant_files/predictions/predictions_nextstep_learned_reranked.csv
  participant_files/predictions/learned_reranker_report.md
  participant_files/predictions/learned_reranker.pt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "tracks" / "industrial-infineon"

NEXT_STEP_DATA = TRACK / "data" / "task_datasets_v1" / "next_step_prediction.csv"
EVAL_VALID = ROOT / "participant_files" / "eval_input_valid.csv"

PRED_MOD_PATH = ROOT / "participant_files" / "make_eval_predictions.py"
RERANK_MOD_PATH = ROOT / "participant_files" / "rerank_nextstep_with_rules.py"

OUT_DIR = ROOT / "participant_files" / "predictions"
OUT_PRED = OUT_DIR / "predictions_nextstep_learned_reranked.csv"
OUT_REPORT = OUT_DIR / "learned_reranker_report.md"
OUT_MODEL = OUT_DIR / "learned_reranker.pt"
OUT_INTERNAL_EXAMPLES = OUT_DIR / "learned_reranker_internal_examples.csv"


FAMILIES = ["MOSFET", "IGBT", "IC"]

BLOCKS = [
    "LITHO",
    "ETCH",
    "DOPING_THERMAL",
    "DEPOSITION",
    "PLANARIZATION",
    "VIA",
    "PASSIVATION",
    "BACKSIDE",
    "METROLOGY_TEST",
    "LOGISTICS",
    "OTHER",
]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pred_mod = load_module(PRED_MOD_PATH, "make_eval_predictions_mod")
rerank_mod = load_module(RERANK_MOD_PATH, "rerank_nextstep_mod")


def norm(x: str) -> str:
    return str(x or "").strip().upper()


def block(step: str) -> str:
    s = norm(step)
    if "LITHO" in s or s.startswith("SPIN COAT PHOTORESIST") or "MASK LEVEL" in s:
        return "LITHO"
    if "ETCH" in s or s.startswith("OPEN PAD WINDOW"):
        return "ETCH"
    if "IMPLANT" in s or "ANNEAL" in s or "DIFFUSION" in s:
        return "DOPING_THERMAL"
    if s.startswith("DEPOSIT") or "OXIDATION" in s or "GROWTH" in s:
        return "DEPOSITION"
    if s.startswith("CMP") or "PLANAR" in s:
        return "PLANARIZATION"
    if "VIA" in s:
        return "VIA"
    if "PASSIVATION" in s:
        return "PASSIVATION"
    if "BACKSIDE" in s or "GRIND" in s:
        return "BACKSIDE"
    if "TEST" in s or "MEASURE" in s or "INSPECT" in s or "ANALYSIS" in s:
        return "METROLOGY_TEST"
    if "LOT" in s or "RELEASE" in s or "SHIP" in s:
        return "LOGISTICS"
    return "OTHER"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def log1p_count(x: int) -> float:
    return math.log1p(max(0, int(x)))


def cget(counter_map, key, cand) -> int:
    return counter_map.get(key, {}).get(cand, 0)


def reciprocal_rank(truth: str, ranks: list[str]) -> float:
    truth = norm(truth)
    ranks = [norm(r) for r in ranks]
    if truth in ranks:
        return 1.0 / (ranks.index(truth) + 1)
    return 0.0


def make_features(
    family: str,
    partial: list[str],
    cand: str,
    rank_idx: int,
    valid_counts,
    bad_counts,
) -> list[float]:
    family = norm(family)
    cand = norm(cand)

    feats: list[float] = []

    # Original SSL model rank.
    feats.append(rank_idx / 5.0)
    feats.append(1.0 / (rank_idx + 1.0))
    feats.append(1.0 if rank_idx == 0 else 0.0)

    # Prefix length.
    feats.append(min(len(partial), 200) / 200.0)

    # Family one-hot.
    for fam in FAMILIES:
        feats.append(1.0 if family == fam else 0.0)

    # Candidate process-block one-hot.
    b = block(cand)
    for bb in BLOCKS:
        feats.append(1.0 if b == bb else 0.0)

    # Global/family priors from valid data.
    prior = valid_counts["prior"].get(cand, 0)
    fam_prior = valid_counts["fam_prior"].get(family, {}).get(cand, 0)
    feats.append(log1p_count(prior))
    feats.append(log1p_count(fam_prior))

    # Valid continuation counts.
    for n in [1, 2, 3]:
        if len(partial) >= n:
            key = partial[-1] if n == 1 else tuple(partial[-n:])
            fam_key = (family, key)
            feats.append(log1p_count(cget(valid_counts[f"ctx{n}"], fam_key, cand)))
            feats.append(log1p_count(cget(valid_counts[f"gctx{n}"], key, cand)))
        else:
            feats.extend([0.0, 0.0])

    # Invalid mutation-context counts.
    for n in [1, 2, 3]:
        if len(partial) >= n:
            key = partial[-1] if n == 1 else tuple(partial[-n:])
            fam_key = (family, key)
            feats.append(log1p_count(cget(bad_counts[f"bad{n}"], fam_key, cand)))
            feats.append(log1p_count(cget(bad_counts[f"gbad{n}"], key, cand)))
        else:
            feats.extend([0.0, 0.0])

    # Heuristic reranker score + validator violation flag.
    try:
        hscore, details = rerank_mod.score_candidate(
            family=family,
            partial=partial,
            candidate=cand,
            rank_index=rank_idx,
            valid_counts=valid_counts,
            bad_counts=bad_counts,
        )
        feats.append(hscore / 30.0)
        feats.append(1.0 if details.get("validator_penalty", 0.0) < 0 else 0.0)
    except Exception:
        feats.extend([0.0, 0.0])

    return feats


class RerankerMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def score_metrics(rows: list[dict], prefix: str) -> dict[str, float]:
    n = max(1, len(rows))
    return {
        "top1": sum(r[f"{prefix}_hit1"] for r in rows) / n,
        "top3": sum(r[f"{prefix}_hit3"] for r in rows) / n,
        "top5": sum(r[f"{prefix}_hit5"] for r in rows) / n,
        "mrr": sum(r[f"{prefix}_rr"] for r in rows) / n,
    }


def sample_split(rows: list[dict[str, str]], split: str, max_examples: int, seed: int):
    selected = [r for r in rows if r.get("SPLIT", "").strip().lower() == split]
    rng = random.Random(seed)
    if max_examples and len(selected) > max_examples:
        selected = rng.sample(selected, max_examples)
    return selected


def build_candidate_dataset(rows, predictor, valid_counts, bad_counts, include_truth: bool):
    X = []
    y = []

    for i, r in enumerate(rows, start=1):
        family = r["FAMILY"]
        partial = pred_mod.split_steps(r["PREFIX_CONTEXT"])
        truth = norm(r["NEXT_STEP"])

        ranks = [norm(x) for x in predictor.topk_next(family, partial, k=5)]

        candidates = list(ranks)
        if include_truth and truth not in candidates:
            candidates.append(truth)

        for rank_idx, cand in enumerate(candidates):
            X.append(make_features(family, partial, cand, min(rank_idx, 5), valid_counts, bad_counts))
            y.append(1.0 if cand == truth else 0.0)

        if i % 1000 == 0:
            print(f"  built candidates for {i:,}/{len(rows):,} examples", flush=True)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@torch.no_grad()
def evaluate_rows(rows, predictor, reranker, valid_counts, bad_counts, device, write_examples: bool = False):
    results = []
    changed_top1 = 0
    changed_order = 0
    example_rows = []

    for i, r in enumerate(rows, start=1):
        eid = r["EXAMPLE_ID"]
        family = r["FAMILY"]
        partial = pred_mod.split_steps(r["PREFIX_CONTEXT"])
        truth = norm(r["NEXT_STEP"])

        model_ranks = [norm(x) for x in predictor.topk_next(family, partial, k=5)]

        feats = [
            make_features(family, partial, cand, rank_idx, valid_counts, bad_counts)
            for rank_idx, cand in enumerate(model_ranks)
        ]
        x = torch.tensor(feats, dtype=torch.float32, device=device)
        scores = reranker(x).detach().cpu().tolist()

        scored = sorted(zip(scores, model_ranks), key=lambda z: z[0], reverse=True)
        learned_ranks = [cand for _, cand in scored]

        if model_ranks[0] != learned_ranks[0]:
            changed_top1 += 1
        if model_ranks != learned_ranks:
            changed_order += 1

        model_rr = reciprocal_rank(truth, model_ranks)
        learned_rr = reciprocal_rank(truth, learned_ranks)

        row_result = {
            "model_hit1": int(model_ranks[0] == truth),
            "model_hit3": int(truth in model_ranks[:3]),
            "model_hit5": int(truth in model_ranks[:5]),
            "model_rr": model_rr,
            "learned_hit1": int(learned_ranks[0] == truth),
            "learned_hit3": int(truth in learned_ranks[:3]),
            "learned_hit5": int(truth in learned_ranks[:5]),
            "learned_rr": learned_rr,
        }
        results.append(row_result)

        if write_examples:
            example_rows.append([
                eid,
                family,
                truth,
                *(model_ranks + [""] * 5)[:5],
                *(learned_ranks + [""] * 5)[:5],
                row_result["model_hit1"],
                row_result["learned_hit1"],
                f"{model_rr:.4f}",
                f"{learned_rr:.4f}",
            ])

        if i % 1000 == 0:
            print(f"  evaluated {i:,}/{len(rows):,}", flush=True)

    return results, changed_top1, changed_order, example_rows


def write_official_predictions(predictor, reranker, valid_counts, bad_counts, device):
    rows = read_csv(EVAL_VALID)
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)

    changed_top1 = 0
    changed_order = 0

    with OUT_PRED.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])

        for r in rows:
            eid = r["EXAMPLE_ID"].strip()
            family = r["FAMILY"].strip()
            partial = pred_mod.split_steps(r["PARTIAL_SEQUENCE"])

            model_ranks = [norm(x) for x in predictor.topk_next(family, partial, k=5)]

            feats = [
                make_features(family, partial, cand, rank_idx, valid_counts, bad_counts)
                for rank_idx, cand in enumerate(model_ranks)
            ]
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            scores = reranker(x).detach().cpu().tolist()

            scored = sorted(zip(scores, model_ranks), key=lambda z: z[0], reverse=True)
            learned_ranks = [cand for _, cand in scored]

            if model_ranks[0] != learned_ranks[0]:
                changed_top1 += 1
            if model_ranks != learned_ranks:
                changed_order += 1

            while len(learned_ranks) < 5:
                learned_ranks.append("")

            writer.writerow([eid] + learned_ranks[:5])

    return len(rows), changed_top1, changed_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=50000)
    parser.add_argument("--max-val", type=int, default=8000)
    parser.add_argument("--max-test", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SSL predictor checkpoint...", flush=True)
    predictor = pred_mod.HybridPredictor(pred_mod.CHECKPOINT, pred_mod.VOCAB_JSON)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Reranker device:", device, flush=True)

    print("Building valid n-gram counts...", flush=True)
    valid_counts = rerank_mod.build_valid_ngram_counts(rerank_mod.VALID_TRAIN)

    print("Building invalid-context counts...", flush=True)
    bad_counts = rerank_mod.build_invalid_bad_context_counts([
        rerank_mod.EASY_INVALID,
        rerank_mod.HARD_INVALID,
    ])

    print("Loading next-step task dataset...", flush=True)
    all_rows = read_csv(NEXT_STEP_DATA)

    train_rows = sample_split(all_rows, "train", args.max_train, args.seed)
    val_rows = sample_split(all_rows, "val", args.max_val, args.seed + 1)
    test_rows = sample_split(all_rows, "test", args.max_test, args.seed + 2)

    print(f"Train rows: {len(train_rows):,}", flush=True)
    print(f"Val rows:   {len(val_rows):,}", flush=True)
    print(f"Test rows:  {len(test_rows):,}", flush=True)

    print("Building train candidate dataset...", flush=True)
    X_train, y_train = build_candidate_dataset(
        train_rows,
        predictor,
        valid_counts,
        bad_counts,
        include_truth=True,
    )

    print("Train candidate rows:", len(y_train), flush=True)
    print("Positive rate:", float(y_train.mean()), flush=True)

    in_dim = X_train.shape[1]
    model = RerankerMLP(in_dim).to(device)

    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print("Training learned reranker...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(yb)
            n += len(yb)

        print(f"Epoch {epoch:03d} | loss {total_loss / max(1, n):.5f}", flush=True)

    model.eval()

    print("Evaluating validation split...", flush=True)
    val_results, val_changed_top1, val_changed_order, _ = evaluate_rows(
        val_rows,
        predictor,
        model,
        valid_counts,
        bad_counts,
        device,
        write_examples=False,
    )

    print("Evaluating test split...", flush=True)
    test_results, test_changed_top1, test_changed_order, example_rows = evaluate_rows(
        test_rows,
        predictor,
        model,
        valid_counts,
        bad_counts,
        device,
        write_examples=True,
    )

    val_model = score_metrics(val_results, "model")
    val_learned = score_metrics(val_results, "learned")
    test_model = score_metrics(test_results, "model")
    test_learned = score_metrics(test_results, "learned")

    print("Writing official learned-reranked predictions...", flush=True)
    official_n, official_changed_top1, official_changed_order = write_official_predictions(
        predictor,
        model,
        valid_counts,
        bad_counts,
        device,
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": in_dim,
            "families": FAMILIES,
            "blocks": BLOCKS,
            "args": vars(args),
            "val_model": val_model,
            "val_learned": val_learned,
            "test_model": test_model,
            "test_learned": test_learned,
        },
        OUT_MODEL,
    )

    with OUT_INTERNAL_EXAMPLES.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "EXAMPLE_ID",
            "FAMILY",
            "TRUTH",
            "MODEL_RANK_1",
            "MODEL_RANK_2",
            "MODEL_RANK_3",
            "MODEL_RANK_4",
            "MODEL_RANK_5",
            "LEARNED_RANK_1",
            "LEARNED_RANK_2",
            "LEARNED_RANK_3",
            "LEARNED_RANK_4",
            "LEARNED_RANK_5",
            "MODEL_HIT1",
            "LEARNED_HIT1",
            "MODEL_RR",
            "LEARNED_RR",
        ])
        writer.writerows(example_rows)

    report = f"""# Learned Contrastive Reranker Report

This trains a small learned reranker on top of the frozen SSL Transformer candidate generator.

The reranker scores candidate next steps using:
- original model rank,
- valid n-gram continuation evidence,
- invalid mutation-context penalty features,
- process-block/family features,
- validator violation features.

## Validation Split

| Metric | Model only | Learned reranker | Delta |
|---|---:|---:|---:|
| Top-1 Accuracy | {val_model["top1"]:.4f} | {val_learned["top1"]:.4f} | {val_learned["top1"] - val_model["top1"]:+.4f} |
| Top-3 Accuracy | {val_model["top3"]:.4f} | {val_learned["top3"]:.4f} | {val_learned["top3"] - val_model["top3"]:+.4f} |
| Top-5 Accuracy | {val_model["top5"]:.4f} | {val_learned["top5"]:.4f} | {val_learned["top5"] - val_model["top5"]:+.4f} |
| MRR | {val_model["mrr"]:.4f} | {val_learned["mrr"]:.4f} | {val_learned["mrr"] - val_model["mrr"]:+.4f} |

Validation Top-1 changed: {val_changed_top1} / {len(val_results)} = {val_changed_top1 / max(1, len(val_results)):.2%}

## Test Split

| Metric | Model only | Learned reranker | Delta |
|---|---:|---:|---:|
| Top-1 Accuracy | {test_model["top1"]:.4f} | {test_learned["top1"]:.4f} | {test_learned["top1"] - test_model["top1"]:+.4f} |
| Top-3 Accuracy | {test_model["top3"]:.4f} | {test_learned["top3"]:.4f} | {test_learned["top3"] - test_model["top3"]:+.4f} |
| Top-5 Accuracy | {test_model["top5"]:.4f} | {test_learned["top5"]:.4f} | {test_learned["top5"] - test_model["top5"]:+.4f} |
| MRR | {test_model["mrr"]:.4f} | {test_learned["mrr"]:.4f} | {test_learned["mrr"] - test_model["mrr"]:+.4f} |

Test Top-1 changed: {test_changed_top1} / {len(test_results)} = {test_changed_top1 / max(1, len(test_results)):.2%}

## Official Eval Prediction Activity

| Metric | Value |
|---|---:|
| Official eval examples | {official_n} |
| Official Top-1 changed vs model-only | {official_changed_top1} |
| Official Top-1 changed % | {official_changed_top1 / max(1, official_n):.2%} |
| Official any order changed | {official_changed_order} |
| Official any order changed % | {official_changed_order / max(1, official_n):.2%} |

## Output Files

| File | Path |
|---|---|
| Learned-reranked official next-step predictions | `{OUT_PRED}` |
| Learned reranker checkpoint | `{OUT_MODEL}` |
| Internal example dump | `{OUT_INTERNAL_EXAMPLES}` |

## Decision Rule

Use learned-reranked official predictions only if validation and test Top-1 or MRR improve.
If validation improves but test drops, keep the model-only predictions.
"""

    OUT_REPORT.write_text(report, encoding="utf-8")

    print(report, flush=True)
    print("Wrote:", OUT_PRED, flush=True)
    print("Wrote:", OUT_REPORT, flush=True)
    print("Wrote:", OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()
