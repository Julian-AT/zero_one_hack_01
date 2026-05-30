"""Trigram-with-backoff baseline for the Industrial AI track.

Self-contained: no project imports, only stdlib + the organizers' CSV reader.

Builds a count-based language model over step strings:
    P(step | step_{-2}, step_{-1})  with backoff to bigram and unigram.

Reports the four organizer-scored metrics for Task 1 (next-step prediction):
    Top-1, Top-3, Top-5 accuracy, MRR

Also runs three eval splits to characterize generalization:
    1. Held-out 20% inside each family ("in-distribution") at 60% and 80%
       truncation, matching the organizers' eval_input_valid.csv schema.
    2. Leave-one-family-out (LoFO): train on 2 families, eval on the 3rd.
       This is our self-reported Task 4 OOD proxy.
    3. All-data smoothed re-prediction (matches the EDA result of 0.993 Top-5).

Outputs are written to shared/extras/results/baselines/ for the report.

Usage
-----
    python shared/extras/baselines/trigram_baseline.py
    python shared/extras/baselines/trigram_baseline.py --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "competition" / "track-details" / "training_data"
OUT_DIR = REPO / "shared" / "extras" / "results" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_FILES = {
    "mosfet": DATA_DIR / "MOSFET_variants.csv",
    "igbt": DATA_DIR / "IGBT_variants.csv",
    "ic": DATA_DIR / "IC_variants.csv",
}

# Reuse the organizers' robust CSV reader (handles BOM, quoted headers, etc.).
sys.path.insert(0, str(DATA_DIR))
from generate_sequences import read_csv_sequences  # noqa: E402

if str(REPO / "models") not in sys.path:
    sys.path.insert(0, str(REPO / "models"))
from transformer_xlstm.eval.metrics import normalized_edit_distance  # noqa: E402


class TrigramBackoff:
    """Trigram with Katz-style fall-through backoff.

    Ranking strategy (for Top-K next-step prediction):
      1. Emit candidates ranked by trigram counts at (ctx2, ctx1).
      2. If fewer than K, fill remaining slots from bigram counts at ctx1.
      3. If still fewer, fill from unigram counts (frequent steps overall).

    This is mutually-exclusive backoff: the higher order's ranking wins for
    its candidates, and the lower order is only consulted when slots are
    still empty. Matches the EDA logic that yielded Top-5 = 0.993.
    """

    def __init__(self) -> None:
        self.tri: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.bi: dict[str, Counter[str]] = defaultdict(Counter)
        self.uni: Counter[str] = Counter()

    def fit(self, sequences: Iterable[list[str]]) -> None:
        for s in sequences:
            for i, w in enumerate(s):
                self.uni[w] += 1
                if i >= 1:
                    self.bi[s[i - 1]][w] += 1
                if i >= 2:
                    self.tri[(s[i - 2], s[i - 1])][w] += 1

    def rank(self, ctx2: str | None, ctx1: str | None, k: int = 5) -> list[str]:
        """Return top-k next-step candidates with Katz-style fall-through."""
        out: list[str] = []
        seen: set[str] = set()

        def add_from(counter: Counter[str]) -> bool:
            for w, _ in counter.most_common():
                if w in seen:
                    continue
                out.append(w)
                seen.add(w)
                if len(out) >= k:
                    return True
            return False

        # 1. Trigram
        if ctx2 is not None and ctx1 is not None and (ctx2, ctx1) in self.tri:
            if add_from(self.tri[(ctx2, ctx1)]):
                return out
        # 2. Bigram (fill remaining slots)
        if ctx1 is not None and ctx1 in self.bi:
            if add_from(self.bi[ctx1]):
                return out
        # 3. Unigram (fill remaining slots)
        add_from(self.uni)
        return out

    def complete(self, prefix: list[str], max_len: int = 200) -> list[str]:
        """Greedy completion until SHIP LOT or max_len."""
        out: list[str] = []
        ctx2 = prefix[-2] if len(prefix) >= 2 else None
        ctx1 = prefix[-1] if len(prefix) >= 1 else None
        for _ in range(max_len):
            top = self.rank(ctx2, ctx1, k=1)
            if not top:
                break
            nxt = top[0]
            out.append(nxt)
            if nxt == "SHIP LOT":
                break
            ctx2, ctx1 = ctx1, nxt
        return out


def topk_metrics(model: TrigramBackoff, sequences: list[list[str]]) -> dict[str, float]:
    """Compute Top-1/3/5 accuracy and MRR over all (prefix, gold) positions
    in `sequences`, starting from position 2 (so trigram context is defined).
    """
    n = c1 = c3 = c5 = 0
    rr_sum = 0.0
    for s in sequences:
        for i in range(2, len(s)):
            ctx2, ctx1, gold = s[i - 2], s[i - 1], s[i]
            ranked = model.rank(ctx2, ctx1, k=5)
            n += 1
            if not ranked:
                continue
            if ranked[0] == gold:
                c1 += 1
            if gold in ranked[:3]:
                c3 += 1
            if gold in ranked[:5]:
                c5 += 1
            if gold in ranked:
                rr_sum += 1.0 / (ranked.index(gold) + 1)
    return {
        "n": n,
        "top1": c1 / n if n else 0.0,
        "top3": c3 / n if n else 0.0,
        "top5": c5 / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
    }


def truncated_completion_metrics(
    model: TrigramBackoff,
    sequences: list[list[str]],
    completion_fraction: float,
) -> dict[str, float]:
    """Simulate the organizer's eval_input_valid.csv setup:
    truncate each sequence at `completion_fraction` and ask the model to predict
    the next step from that truncation point (Task 1 style at the cut).

    Reports Top-1/3/5 + MRR over the single 'next step after cut' per sequence,
    plus an Edit-Distance-based metric over the full completion (Task 2 style).
    """
    n = c1 = c3 = c5 = 0
    rr_sum = 0.0
    edit_sum = 0.0
    exact_match = 0
    for s in sequences:
        if len(s) < 5:
            continue
        cut = max(2, int(len(s) * completion_fraction))
        if cut >= len(s):
            continue
        # Task 1 style at the cut
        ctx2, ctx1, gold = s[cut - 2], s[cut - 1], s[cut]
        ranked = model.rank(ctx2, ctx1, k=5)
        n += 1
        if ranked and ranked[0] == gold:
            c1 += 1
        if gold in ranked[:3]:
            c3 += 1
        if gold in ranked[:5]:
            c5 += 1
        if gold in ranked:
            rr_sum += 1.0 / (ranked.index(gold) + 1)
        # Task 2 style: full completion from the cut
        gold_completion = s[cut:]
        pred_completion = model.complete(s[:cut], max_len=len(s) - cut + 30)
        edit_sum += normalized_edit_distance(pred_completion, gold_completion)
        if pred_completion == gold_completion:
            exact_match += 1
    return {
        "n": n,
        "top1_at_cut": c1 / n if n else 0.0,
        "top3_at_cut": c3 / n if n else 0.0,
        "top5_at_cut": c5 / n if n else 0.0,
        "mrr_at_cut": rr_sum / n if n else 0.0,
        "completion_exact_match": exact_match / n if n else 0.0,
        "completion_normalized_edit_distance": edit_sum / n if n else 0.0,
    }


def family_split(
    family_seqs: dict[str, list[list[str]]], seed: int = 42, test_fraction: float = 0.2
) -> tuple[dict, dict]:
    """Per-family 80/20 train/test split."""
    rng = random.Random(seed)
    train, test = {}, {}
    for fam, seqs in family_seqs.items():
        idx = list(range(len(seqs)))
        rng.shuffle(idx)
        n_test = int(len(idx) * test_fraction)
        test[fam] = [seqs[i] for i in idx[:n_test]]
        train[fam] = [seqs[i] for i in idx[n_test:]]
    return train, test


def lofo_split(
    family_seqs: dict[str, list[list[str]]], held_out: str
) -> tuple[list[list[str]], list[list[str]]]:
    """Leave-one-family-out: train on the two other families, eval on `held_out`."""
    train: list[list[str]] = []
    for fam, seqs in family_seqs.items():
        if fam != held_out:
            train.extend(seqs)
    return train, list(family_seqs[held_out])


def flatten(d: dict[str, list[list[str]]]) -> list[list[str]]:
    out: list[list[str]] = []
    for v in d.values():
        out.extend(v)
    return out


def load_family_sequences() -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for fam, path in FAMILY_FILES.items():
        raw = read_csv_sequences(path)
        out[fam] = list(raw.values())
        print(
            f"  loaded {fam}: {len(out[fam])} sequences, "
            f"mean len {sum(len(s) for s in out[fam]) / len(out[fam]):.1f}"
        )
    return out


def write_topk_submission_csv(
    model: TrigramBackoff,
    eval_examples: list[tuple[str, list[str]]],
    out_path: Path,
) -> None:
    """Write `nextstep.csv` in the organizers' submission format.

    Args:
        eval_examples: list of (example_id, partial_sequence)
    """
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for example_id, partial in eval_examples:
            ctx2 = partial[-2] if len(partial) >= 2 else None
            ctx1 = partial[-1] if len(partial) >= 1 else None
            ranked = model.rank(ctx2, ctx1, k=5)
            # Pad if fewer than 5 (very unlikely with backoff to unigram)
            while len(ranked) < 5:
                ranked.append("")
            w.writerow([example_id, *ranked[:5]])
    print(f"  wrote {out_path}")


def write_completion_submission_csv(
    model: TrigramBackoff,
    eval_examples: list[tuple[str, list[str], int]],
    out_path: Path,
) -> None:
    """Write `completion.csv` in the organizers' submission format.

    Args:
        eval_examples: list of (example_id, partial_sequence, target_max_len)
    """
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for example_id, partial, target_len in eval_examples:
            pred = model.complete(partial, max_len=target_len + 30)
            w.writerow([example_id, "|".join(pred)])
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    print("─" * 72)
    print(f"Loading sequences ... (seed={args.seed})")
    family_seqs = load_family_sequences()

    results: dict = {"seed": args.seed, "test_fraction": args.test_fraction}

    print("\n[1] EDA reproduction (train on all 3000, eval on all — memorization upper bound)")
    model_all = TrigramBackoff()
    model_all.fit(flatten(family_seqs))
    m = topk_metrics(model_all, flatten(family_seqs))
    print(
        f"    n={m['n']:,}  Top-1={m['top1']:.4f}  Top-3={m['top3']:.4f}  "
        f"Top-5={m['top5']:.4f}  MRR={m['mrr']:.4f}"
    )
    results["eda_reproduction"] = m

    print(f"\n[2] In-distribution held-out ({int(args.test_fraction * 100)}% per family)")
    train_id, test_id = family_split(family_seqs, seed=args.seed, test_fraction=args.test_fraction)
    model_id = TrigramBackoff()
    model_id.fit(flatten(train_id))
    m = topk_metrics(model_id, flatten(test_id))
    print(
        f"    n={m['n']:,}  Top-1={m['top1']:.4f}  Top-3={m['top3']:.4f}  "
        f"Top-5={m['top5']:.4f}  MRR={m['mrr']:.4f}"
    )
    results["id_holdout_overall"] = m

    # Per-family ID breakdown
    print("    per-family:")
    results["id_holdout_per_family"] = {}
    for fam, seqs in test_id.items():
        m = topk_metrics(model_id, seqs)
        print(
            f"      {fam.upper():>6}: n={m['n']:,}  Top-1={m['top1']:.4f}  "
            f"Top-5={m['top5']:.4f}  MRR={m['mrr']:.4f}"
        )
        results["id_holdout_per_family"][fam] = m

    print(
        f"\n[3] Truncation metrics on the {int(args.test_fraction * 100)}% held-out "
        f"(simulates eval_input_valid.csv format)"
    )
    results["truncation"] = {}
    for frac in (0.6, 0.8):
        print(f"    completion_fraction = {frac}")
        per_fam = {}
        for fam, seqs in test_id.items():
            m = truncated_completion_metrics(model_id, seqs, frac)
            print(
                f"      {fam.upper():>6}: n={m['n']:,}  "
                f"Top-1@cut={m['top1_at_cut']:.4f}  Top-5@cut={m['top5_at_cut']:.4f}  "
                f"ExactMatch={m['completion_exact_match']:.4f}  "
                f"NED={m['completion_normalized_edit_distance']:.4f}"
            )
            per_fam[fam] = m
        results["truncation"][f"{frac}"] = per_fam

    print("\n[4] Leave-one-family-out (LoFO) — our Task 4 OOD proxy")
    results["lofo"] = {}
    for held_out in family_seqs:
        train_seqs, test_seqs = lofo_split(family_seqs, held_out)
        m_lofo = TrigramBackoff()
        m_lofo.fit(train_seqs)
        m = topk_metrics(m_lofo, test_seqs)
        print(
            f"    held_out={held_out.upper():>6}  "
            f"Top-1={m['top1']:.4f}  Top-3={m['top3']:.4f}  "
            f"Top-5={m['top5']:.4f}  MRR={m['mrr']:.4f}"
        )
        results["lofo"][held_out] = m

    # Truncation under LoFO too (this is the realistic Task 4 number)
    print("    truncation metrics under LoFO:")
    results["lofo_truncation"] = {}
    for held_out in family_seqs:
        train_seqs, test_seqs = lofo_split(family_seqs, held_out)
        m_lofo = TrigramBackoff()
        m_lofo.fit(train_seqs)
        for frac in (0.6, 0.8):
            m = truncated_completion_metrics(m_lofo, test_seqs, frac)
            key = f"{held_out}_{frac}"
            results["lofo_truncation"][key] = m
            print(
                f"      held_out={held_out.upper():>6}  frac={frac}  "
                f"Top-1@cut={m['top1_at_cut']:.4f}  Top-5@cut={m['top5_at_cut']:.4f}  "
                f"ExactMatch={m['completion_exact_match']:.4f}  "
                f"NED={m['completion_normalized_edit_distance']:.4f}"
            )

    #         (so once the real eval_input_valid.csv lands, the format is verified)
    print("\n[5] Writing sample submission-format CSVs (using held-out test as proxy)")
    eval_nextstep: list[tuple[str, list[str]]] = []
    eval_completion: list[tuple[str, list[str], int]] = []
    for fam, seqs in test_id.items():
        for i, s in enumerate(seqs):
            if len(s) < 5:
                continue
            for frac in (0.6, 0.8):
                cut = max(2, int(len(s) * frac))
                if cut >= len(s):
                    continue
                example_id = f"{fam}_{i:04d}_f{int(frac * 100)}"
                eval_nextstep.append((example_id, s[:cut]))
                eval_completion.append((example_id, s[:cut], len(s) - cut))

    write_topk_submission_csv(model_id, eval_nextstep, OUT_DIR / "trigram_nextstep_sample.csv")
    write_completion_submission_csv(
        model_id, eval_completion, OUT_DIR / "trigram_completion_sample.csv"
    )

    out_json = OUT_DIR / "trigram_metrics.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")

    md = [
        "# Trigram-with-backoff baseline\n",
        f"Seed: {args.seed}.  Test fraction: {args.test_fraction}.\n",
        "## EDA reproduction (memorization upper bound)\n",
        "Train and eval on all 3000 sequences. Reproduces the EDA number.\n",
        f"- Top-1 = **{results['eda_reproduction']['top1']:.4f}**",
        f"- Top-3 = **{results['eda_reproduction']['top3']:.4f}**",
        f"- Top-5 = **{results['eda_reproduction']['top5']:.4f}**",
        f"- MRR   = **{results['eda_reproduction']['mrr']:.4f}**\n",
        "## In-distribution held-out (honest baseline)\n",
        "Train on 80% per family, eval on 20% held-out per family.\n",
        "| split | Top-1 | Top-3 | Top-5 | MRR |",
        "|---|--:|--:|--:|--:|",
        f"| ALL  | {results['id_holdout_overall']['top1']:.4f} | "
        f"{results['id_holdout_overall']['top3']:.4f} | "
        f"{results['id_holdout_overall']['top5']:.4f} | "
        f"{results['id_holdout_overall']['mrr']:.4f} |",
    ]
    for fam, m in results["id_holdout_per_family"].items():
        md.append(
            f"| {fam.upper()} | {m['top1']:.4f} | {m['top3']:.4f} | "
            f"{m['top5']:.4f} | {m['mrr']:.4f} |"
        )
    md.append("\n## LoFO (Task 4 OOD proxy)\n")
    md.append(
        "Train on 2 families, eval on the held-out 3rd. This is the number "
        "we expect to roughly predict performance on the hidden family 4.\n"
    )
    md.append("| held_out | Top-1 | Top-3 | Top-5 | MRR |")
    md.append("|---|--:|--:|--:|--:|")
    for fam, m in results["lofo"].items():
        md.append(
            f"| {fam.upper()} | {m['top1']:.4f} | {m['top3']:.4f} | "
            f"{m['top5']:.4f} | {m['mrr']:.4f} |"
        )
    md.append("\n## Truncation @ 0.6 / 0.8 (simulates eval_input_valid.csv)\n")
    md.append("In-distribution; Task 1 (Top-K at cut) and Task 2 (full completion).\n")
    md.append("| frac | family | Top-1@cut | Top-5@cut | ExactMatch | NormEditDist |")
    md.append("|---|---|--:|--:|--:|--:|")
    for frac, per_fam in results["truncation"].items():
        for fam, m in per_fam.items():
            md.append(
                f"| {frac} | {fam.upper()} | "
                f"{m['top1_at_cut']:.4f} | {m['top5_at_cut']:.4f} | "
                f"{m['completion_exact_match']:.4f} | "
                f"{m['completion_normalized_edit_distance']:.4f} |"
            )
    md.append("\n## LoFO + truncation (the realistic Task 4 number)\n")
    md.append("| held_out | frac | Top-1@cut | Top-5@cut | ExactMatch | NormEditDist |")
    md.append("|---|---|--:|--:|--:|--:|")
    for key, m in results["lofo_truncation"].items():
        held, frac = key.rsplit("_", 1)
        md.append(
            f"| {held.upper()} | {frac} | "
            f"{m['top1_at_cut']:.4f} | {m['top5_at_cut']:.4f} | "
            f"{m['completion_exact_match']:.4f} | "
            f"{m['completion_normalized_edit_distance']:.4f} |"
        )
    (OUT_DIR / "trigram_metrics.md").write_text("\n".join(md))
    print(f"Wrote {OUT_DIR / 'trigram_metrics.md'}")


if __name__ == "__main__":
    main()
