"""The prefix automaton: `valid_next_set(prefix)` — the symbolically-legal
support over which the ranker chooses. This is the heart of the symbolic-first
approach: learning is confined to ranking inside this set.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Set

from nspe.official import gs
from nspe.rules import would_violate
from nspe.roles import induce_role

# Trigger universe for the fast incremental checker (known-vocab steps only).
_DEP = gs.DEPOSITION_STEPS
_METAL_ETCH = gs.METAL_ETCH_STEPS
_ETCH = gs.ETCH_STEPS
_IMPLANT = gs.IMPLANT_STEPS
_OPENER = gs.IMPLANT_OPENER_STEPS
_CMP = gs.CMP_STEPS
_FILL = gs.FILL_STEPS
_CLEAN = gs.CLEAN_STEPS
_PAD = gs.PAD_WINDOW_STEPS
_ETEST = gs.ELECTRICAL_TEST_STEPS
_DEVELOP = frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"})
_TRIGGERS = (_DEP | _METAL_ETCH | _ETCH | _IMPLANT | _CMP | _PAD | _ETEST
             | {"SHIP LOT", "DEPOSIT BACKSIDE METAL"})


def legal_next_sets(steps, candidate_vocab) -> List[Set[str]]:
    """Fast incremental equivalent of
    ``[valid_next_set(steps[:t], candidate_vocab, use_roles=False) for t in 0..len]``.

    Walks the sequence once, maintaining the running state the 10 rules need, and
    applies each rule's check locally to the ~40 *trigger* candidates per position
    (non-triggers are always legal). Verified to match the official-validator path
    exactly on known-vocab sequences; this is the path used to build the training
    semantic-loss mask (which is otherwise O(positions × |vocab| × len) in Python).
    """
    cv = list(candidate_vocab)
    base = {c for c in cv if c not in _TRIGGERS and not c.startswith("ALIGN MASK LEVEL ")}
    trig = [c for c in cv if c in _TRIGGERS or c.startswith("ALIGN MASK LEVEL ")]

    out: List[Set[str]] = []
    prefix: List[str] = []
    cure = dep_pass = sort = False
    last_align: Optional[int] = None

    for t in range(len(steps) + 1):
        w15 = prefix[-15:]
        w12 = prefix[-12:]
        w6 = prefix[-6:]
        has_clean12 = any(s in _CLEAN for s in w12)
        has_dev12 = any(s in _DEVELOP for s in w12)
        has_dev15 = any(s in _DEVELOP for s in w15)
        has_expose15 = any(s.startswith("EXPOSE LITHO LEVEL") for s in w15)
        has_opener15 = any(s in _OPENER for s in w15)
        has_fill6 = any(s in _FILL for s in w6)

        legal = set(base)
        for c in trig:
            if c in _DEP and not has_clean12:
                continue
            if c in _METAL_ETCH and not (has_expose15 and has_dev15):
                continue
            if c in _ETCH and not has_dev12:
                continue
            if c in _IMPLANT and not has_opener15:
                continue
            if c in _CMP and not has_fill6:
                continue
            if c in _PAD and not (dep_pass and cure):
                continue
            if c in _ETEST and not cure:
                continue
            if c == "SHIP LOT" and not sort:
                continue
            if c == "DEPOSIT BACKSIDE METAL" and not cure:
                continue
            if c.startswith("ALIGN MASK LEVEL "):
                tail = c.split("ALIGN MASK LEVEL ")[1]
                if tail.isdigit() and last_align is not None:
                    L = int(tail)
                    if L > last_align + 1 or L < last_align:
                        continue
            legal.add(c)
        out.append(legal)

        if t < len(steps):
            s = steps[t]
            prefix.append(s)
            if s == "CURE PASSIVATION":
                cure = True
            elif s in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"):
                dep_pass = True
            elif s == "WAFER SORT TEST":
                sort = True
            elif s.startswith("ALIGN MASK LEVEL "):
                tail = s.split("ALIGN MASK LEVEL ")[1]
                if tail.isdigit():
                    last_align = int(tail)
    return out


def valid_next_set(prefix, candidate_vocab: Iterable[str], use_roles: bool = False,
                   allowed_roles: Optional[set] = None) -> set:
    """Rule-legal next steps for an already-valid `prefix`.

    Parameters
    ----------
    prefix : list[str]
        An already rule-valid prefix.
    candidate_vocab : iterable[str]
        Step inventory to consider (e.g. all training steps, optionally family-filtered).
    use_roles : bool
        If True, use the role-augmented validator (for unseen-vocab robustness).
    allowed_roles : set[str] | None
        Optional sharpening: restrict candidates to these induced roles (e.g. the
        ranker's top-r predicted roles). The legality filter is always applied.

    Returns
    -------
    set[str] of legal next steps. The gold next step of any valid sequence is
    always included (legality never excludes it), so masking cannot reduce Top-k
    recall — it can only help.
    """
    out = set()
    for c in candidate_vocab:
        if allowed_roles is not None and induce_role(c) not in allowed_roles:
            continue
        if not would_violate(prefix, c, use_roles):
            out.add(c)
    return out


__all__ = ["valid_next_set", "legal_next_sets"]
