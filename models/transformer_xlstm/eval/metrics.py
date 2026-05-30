"""Shared metric helpers for evaluation and reranking.

Stdlib-only on purpose (no torch), so the prediction post-processing scripts and
baselines can import these without a heavy dependency. Previously
`normalized_edit_distance` lived in four places and `reciprocal_rank` in two.
"""

from __future__ import annotations


def normalized_edit_distance(a: list[str], b: list[str]) -> float:
    """Length-normalized Levenshtein distance between two step lists (0..1)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb] / max(la, lb)


def reciprocal_rank(truth: str, ranks: list[str]) -> float:
    """Reciprocal rank of `truth` within `ranks` (case-insensitive); 0 if absent."""
    truth = str(truth or "").strip().upper()
    ranks = [str(r or "").strip().upper() for r in ranks]
    if truth in ranks:
        return 1.0 / (ranks.index(truth) + 1)
    return 0.0
