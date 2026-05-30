"""The prefix automaton: `valid_next_set(prefix)` — the symbolically-legal
support over which the ranker chooses. This is the heart of the symbolic-first
approach: learning is confined to ranking inside this set.
"""
from __future__ import annotations

from typing import Iterable, Optional

from nspe.rules import would_violate
from nspe.roles import induce_role


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


__all__ = ["valid_next_set"]
