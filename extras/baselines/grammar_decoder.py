"""Validator-masked next-step decoder + grammar-constrained completion.

The organizers' `validate_sequence` already encodes the 10 process-logic
rules. We exploit it at *inference* time: for any candidate next-step `c`
given prefix `p`, we run `validate_sequence(p + [c])` and reject `c` if
appending it creates a *new* violation involving position `len(p)`.

This wraps any base next-step model. We use the trigram baseline here, but
the same wrapper works on a Transformer or xLSTM at inference time.

Outputs metrics in the same format as `trigram_baseline.py` so the report
can show a clean side-by-side: trigram vs grammar-constrained-trigram.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "tracks" / "industrial-infineon" / "training_data"
OUT_DIR = REPO / "extras" / "results" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DATA_DIR))
from generate_sequences import validate_sequence  # noqa: E402

# Reuse the trigram from the existing baseline.
sys.path.insert(0, str(REPO / "extras" / "baselines"))
from trigram_baseline import (  # noqa: E402
    TrigramBackoff,
    family_split,
    flatten,
    load_family_sequences,
    lofo_split,
    normalized_edit_distance,
)


def candidate_immediately_violates(prefix: list[str], candidate: str) -> bool:
    """True iff appending `candidate` to `prefix` introduces a NEW violation
    located at position `len(prefix)` (the just-added step).

    This is much cheaper than running a full lookahead — we just check whether
    THIS step caused a violation right now.
    """
    if not prefix and candidate == "SHIP LOT":
        return True  # SHIP LOT can never be first
    new_prefix = prefix + [candidate]
    new_idx = len(prefix)
    for v in validate_sequence(new_prefix):
        if v.step_index == new_idx:
            return True
    return False


def grammar_filtered_topk(
    model: TrigramBackoff,
    prefix: list[str],
    k: int = 5,
    pool_size: int = 30,
) -> list[str]:
    """Get top-`pool_size` candidates from the base model, drop any that
    immediately violate the grammar at this position, return the first `k`.

    If the filter removes all candidates, fall back to the base model's
    unfiltered top-k (so we never produce an empty prediction).
    """
    ctx2 = prefix[-2] if len(prefix) >= 2 else None
    ctx1 = prefix[-1] if len(prefix) >= 1 else None
    raw = model.rank(ctx2, ctx1, k=pool_size)
    kept: list[str] = []
    for cand in raw:
        if candidate_immediately_violates(prefix, cand):
            continue
        kept.append(cand)
        if len(kept) >= k:
            break
    if not kept:
        # All grammar-invalid: fall back so we still emit something.
        return raw[:k]
    # Pad to k from the raw list (preserving original ranking) if needed.
    while len(kept) < k:
        for c in raw:
            if c not in kept:
                kept.append(c)
                break
    return kept[:k]


def grammar_complete(
    model: TrigramBackoff,
    prefix: list[str],
    max_len: int = 200,
) -> list[str]:
    """Greedy completion with grammar masking applied at every step."""
    out: list[str] = []
    cur = list(prefix)
    for _ in range(max_len):
        top = grammar_filtered_topk(model, cur, k=1)
        if not top:
            break
        nxt = top[0]
        out.append(nxt)
        cur.append(nxt)
        if nxt == "SHIP LOT":
            break
    return out


def topk_metrics_grammar(model: TrigramBackoff, sequences: list[list[str]]) -> dict[str, float]:
    n = c1 = c3 = c5 = 0
    rr_sum = 0.0
    for s in sequences:
        for i in range(2, len(s)):
            gold = s[i]
            prefix = s[:i]
            ranked = grammar_filtered_topk(model, prefix, k=5)
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


def truncated_completion_grammar(
    model: TrigramBackoff,
    sequences: list[list[str]],
    completion_fraction: float,
) -> dict[str, float]:
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
        prefix = s[:cut]
        # Task 1 style: top-K at the cut, grammar-filtered
        ranked = grammar_filtered_topk(model, prefix, k=5)
        gold_next = s[cut]
        n += 1
        if ranked and ranked[0] == gold_next:
            c1 += 1
        if gold_next in ranked[:3]:
            c3 += 1
        if gold_next in ranked[:5]:
            c5 += 1
        if gold_next in ranked:
            rr_sum += 1.0 / (ranked.index(gold_next) + 1)
        # Task 2 style: full completion with grammar mask
        gold_completion = s[cut:]
        pred_completion = grammar_complete(model, prefix, max_len=len(s) - cut + 30)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("─" * 72)
    print(f"Loading sequences... (seed={args.seed})")
    family_seqs = load_family_sequences()

    results: dict = {"seed": args.seed}

    print("\n[1] Training trigram on 80% per family…")
    train_id, test_id = family_split(family_seqs, seed=args.seed, test_fraction=0.2)
    base = TrigramBackoff()
    base.fit(flatten(train_id))

    print("\n[2] In-distribution held-out: top-K (grammar-filtered)")
    m_g = topk_metrics_grammar(base, flatten(test_id))
    print(
        f"    n={m_g['n']:,}  Top-1={m_g['top1']:.4f}  Top-3={m_g['top3']:.4f}  "
        f"Top-5={m_g['top5']:.4f}  MRR={m_g['mrr']:.4f}"
    )
    results["id_grammar_topk"] = m_g

    print("\n[3] Truncation metrics (grammar-constrained completion)")
    results["truncation"] = {}
    for frac in (0.6, 0.8):
        print(f"    completion_fraction = {frac}")
        per_fam = {}
        for fam, seqs in test_id.items():
            m = truncated_completion_grammar(base, seqs, frac)
            print(
                f"      {fam.upper():>6}: n={m['n']:,}  "
                f"Top-1@cut={m['top1_at_cut']:.4f}  Top-5@cut={m['top5_at_cut']:.4f}  "
                f"ExactMatch={m['completion_exact_match']:.4f}  "
                f"NED={m['completion_normalized_edit_distance']:.4f}"
            )
            per_fam[fam] = m
        results["truncation"][f"{frac}"] = per_fam

    print("\n[4] LoFO + grammar-constrained completion (Task 4 OOD proxy)")
    results["lofo_truncation"] = {}
    for held_out in family_seqs:
        train_seqs, test_seqs = lofo_split(family_seqs, held_out)
        m_lofo = TrigramBackoff()
        m_lofo.fit(train_seqs)
        for frac in (0.6, 0.8):
            m = truncated_completion_grammar(m_lofo, test_seqs, frac)
            key = f"{held_out}_{frac}"
            results["lofo_truncation"][key] = m
            print(
                f"      held_out={held_out.upper():>6}  frac={frac}  "
                f"Top-1@cut={m['top1_at_cut']:.4f}  Top-5@cut={m['top5_at_cut']:.4f}  "
                f"ExactMatch={m['completion_exact_match']:.4f}  "
                f"NED={m['completion_normalized_edit_distance']:.4f}"
            )

    out_json = OUT_DIR / "grammar_decoder_metrics.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
