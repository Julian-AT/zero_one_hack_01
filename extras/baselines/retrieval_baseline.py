"""k-NN retrieval baseline for Task 2 (sequence completion).

For each test partial sequence at 60% or 80% completion, we find the most
similar training sequence (Jaccard similarity over step multisets restricted
to the partial slice), then output the corresponding training sequence's
remaining steps — adjusted to the same length as the gold completion.

This baseline has no parameters and no training; it's a memory-based lookup.
The hypothesis is that our highly structured, low-entropy data means a
nearest-neighbor copy is a strong-and-interpretable Task 2 reference.

Also serves as a smoke test for the eval pipeline against the documented
submission format in generation_rules.md §5.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "tracks" / "industrial-infineon" / "training_data"
OUT_DIR = REPO / "extras" / "results" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_DIR))

sys.path.insert(0, str(REPO / "extras" / "baselines"))
from trigram_baseline import (  # noqa: E402
    family_split,
    load_family_sequences,
    lofo_split,
    normalized_edit_distance,
)


def _bag(seq: list[str]) -> Counter[str]:
    return Counter(seq)


def jaccard_count(a: Counter[str], b: Counter[str]) -> float:
    """Weighted Jaccard over multisets (sum-min / sum-max)."""
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 0.0


def _last_k(seq: list[str], k: int) -> list[str]:
    return seq[-k:] if len(seq) >= k else seq


def retrieve_completion(
    test_prefix: list[str],
    train_seqs: list[list[str]],
    target_len: int,
    last_k: int = 20,
) -> tuple[list[str], int]:
    """Find the best matching training sequence and return its continuation.

    Match strategy:
      1. Heavily weight the LAST `last_k` steps of the prefix (recent context
         matters most).
      2. Among training sequences that share the prefix's last step exactly,
         pick by Jaccard over the whole prefix.

    Returns:
        (completion, best_idx)
    """
    prefix_bag = _bag(test_prefix)
    tail = test_prefix[-1] if test_prefix else None

    best_score = -1.0
    best_idx = 0
    best_cont: list[str] = []
    # First pass: insist on tail-step match where possible.
    candidates = [(i, s) for i, s in enumerate(train_seqs) if tail is None or tail in s]
    if not candidates:
        candidates = list(enumerate(train_seqs))
    for i, s in candidates:
        # Align: find positions in s where tail step appears; use the latest
        # occurrence as a candidate split point.
        if tail is None:
            split = len(test_prefix)
        else:
            try:
                split = len(s) - 1 - s[::-1].index(tail)
            except ValueError:
                continue
        if split + 1 >= len(s):
            continue
        score = jaccard_count(prefix_bag, _bag(s[: split + 1]))
        if score > best_score:
            best_score = score
            best_idx = i
            best_cont = s[split + 1 :]
    # Trim/pad continuation to target_len.
    if len(best_cont) >= target_len:
        best_cont = best_cont[:target_len]
    return best_cont, best_idx


def evaluate(
    test_seqs: list[list[str]],
    train_seqs: list[list[str]],
    completion_fraction: float,
) -> dict[str, float]:
    n = exact = 0
    ned_sum = 0.0
    for s in test_seqs:
        if len(s) < 5:
            continue
        cut = max(2, int(len(s) * completion_fraction))
        if cut >= len(s):
            continue
        prefix = s[:cut]
        gold = s[cut:]
        pred, _ = retrieve_completion(prefix, train_seqs, target_len=len(gold))
        n += 1
        if pred == gold:
            exact += 1
        ned_sum += normalized_edit_distance(pred, gold)
    return {
        "n": n,
        "exact_match": exact / n if n else 0.0,
        "normalized_edit_distance": ned_sum / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("─" * 72)
    print(f"Loading sequences... (seed={args.seed})")
    family_seqs = load_family_sequences()

    results: dict = {"seed": args.seed}

    train_id, test_id = family_split(family_seqs, seed=args.seed, test_fraction=0.2)
    all_train = [s for seqs in train_id.values() for s in seqs]

    print("\n[1] In-distribution retrieval (Task 2 completion)")
    results["id_retrieval"] = {}
    for frac in (0.6, 0.8):
        per_fam = {}
        for fam, seqs in test_id.items():
            m = evaluate(seqs, all_train, frac)
            print(
                f"  frac={frac}  {fam.upper():>6}: n={m['n']:,}  "
                f"ExactMatch={m['exact_match']:.4f}  NED={m['normalized_edit_distance']:.4f}"
            )
            per_fam[fam] = m
        results["id_retrieval"][f"{frac}"] = per_fam

    print("\n[2] LoFO retrieval (Task 4 OOD proxy)")
    results["lofo_retrieval"] = {}
    for held_out in family_seqs:
        train_seqs_lofo, test_seqs_lofo = lofo_split(family_seqs, held_out)
        for frac in (0.6, 0.8):
            m = evaluate(test_seqs_lofo, train_seqs_lofo, frac)
            key = f"{held_out}_{frac}"
            results["lofo_retrieval"][key] = m
            print(
                f"  held_out={held_out.upper():>6}  frac={frac}: "
                f"ExactMatch={m['exact_match']:.4f}  NED={m['normalized_edit_distance']:.4f}"
            )

    out = OUT_DIR / "retrieval_metrics.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
