"""Constrained inference: next-step top-k, beam/greedy completion, symbolic repair.

This is the inference half of the symbolic-first pipeline. Decoding never leaves
the symbolically-legal support: candidates are restricted to
``grammar.valid_next_set(prefix, ...)`` and only *ranked* by the (duck-typed)
ranker. Because the gold next step is always legal (a guarantee established in
``nspe.rules``), restricting to the valid set can only help Top-k recall.

Three public entry points:

  * ``next_step_topk`` — up to ``k`` distinct legal next steps (exactly ``k``
    whenever the candidate vocabulary has >= ``k`` steps, which the real 198-step
    submission vocab always does), ranked by the ranker, with optional
    role-sharpening (intersect the legal set with the ranker's top predicted
    roles for a much stronger Top-1). The set is widened progressively if
    role-sharpening leaves fewer than ``k`` candidates, and falls back to the
    unconstrained ranker only as a last resort — but since the gold is always
    legal we always prefer the constrained set.

  * ``complete`` — constrained greedy (``beam=1``) or beam search until ``SHIP
    LOT`` or ``max_len``, returning ONLY the steps after the given prefix, then
    passing the full sequence through ``repair``. Constrained decoding keeps every
    step legal; in the rare case a position has an empty legal set it falls back
    to the unconstrained ranker, and the bounded ``repair`` pass restores full
    rule-validity (self-tests confirm 100% of completions pass the validator).

  * ``repair`` — bounded greedy fix of the first violation reported by the
    role-augmented validator (insert a CLEAN before an uncleaned deposition, a
    DEVELOP before an unmasked etch/implant window, move SHIP LOT after WAFER
    SORT TEST, etc.). Returns a best-effort sequence.

No torch import at module top — this is symbolic core. The ``ranker`` argument is
duck-typed (``predict`` / ``predict_roles``); a PPM or a neural ranker both work.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from nspe import grammar, roles, rules

__all__ = ["next_step_topk", "complete", "repair"]


# ---------------------------------------------------------------------------
# Next-step top-k
# ---------------------------------------------------------------------------

def _rank(scores: dict, exclude: Optional[set] = None) -> List[str]:
    """Stable descending rank of a {step: prob} dict (ties broken by step name)."""
    exclude = exclude or set()
    items = [(s, p) for s, p in scores.items() if s not in exclude]
    # Sort by (-prob, step) so the order is deterministic across runs.
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return [s for s, _ in items]


def next_step_topk(
    prefix: Sequence[str],
    family: str,
    ranker,
    candidate_vocab: Iterable[str],
    k: int = 5,
    use_roles: bool = False,
    role_sharpen: bool = True,
) -> List[str]:
    """Return up to ``k`` distinct legal next steps, ranked by the ranker
    (exactly ``k`` whenever ``candidate_vocab`` has at least ``k`` distinct steps).

    Strategy (each step strictly widens the candidate set):
      1. legal set sharpened to the ranker's top predicted roles (if
         ``role_sharpen``) — strongest Top-1;
      2. legal set without the role filter;
      3. unconstrained ranker over the full ``candidate_vocab`` — last resort
         padding toward ``k`` items (capped at ``len(candidate_vocab)``).

    The gold next step is always in the legal set, so widening only ever appends
    *additional* (lower-priority) candidates; it never displaces the legal ones.
    """
    prefix = list(prefix)
    candidate_vocab = list(candidate_vocab)

    allowed_roles = None
    if role_sharpen:
        try:
            allowed_roles = set(ranker.predict_roles(prefix, family))
        except Exception:
            allowed_roles = None

    # 1) role-sharpened legal set.
    ranked: List[str] = []
    if allowed_roles is not None:
        v_sharp = grammar.valid_next_set(prefix, candidate_vocab, use_roles, allowed_roles)
        if v_sharp:
            scores = ranker.predict(prefix, family, v_sharp)
            ranked = _rank(scores)

    # 2) full legal set (drop the role filter), appending any new legal steps.
    if len(ranked) < k:
        v_full = grammar.valid_next_set(prefix, candidate_vocab, use_roles)
        if v_full:
            scores = ranker.predict(prefix, family, v_full)
            ranked += _rank(scores, exclude=set(ranked))

    # 3) unconstrained ranker over the whole vocab — last-resort padding only.
    if len(ranked) < k:
        scores = ranker.predict(prefix, family, candidate_vocab)
        ranked += _rank(scores, exclude=set(ranked))

    # 4) absolute fallback: pad from the raw vocab (deterministic) so we always
    #    return k distinct steps even for a tiny vocab / degenerate ranker.
    if len(ranked) < k:
        seen = set(ranked)
        for s in sorted(candidate_vocab):
            if s not in seen:
                ranked.append(s)
                seen.add(s)
            if len(ranked) >= k:
                break

    return ranked[:k]


# ---------------------------------------------------------------------------
# Completion (greedy / beam)
# ---------------------------------------------------------------------------

_TERMINAL = "SHIP LOT"


def _greedy_complete(
    prefix: List[str],
    family: str,
    ranker,
    candidate_vocab: List[str],
    max_len: int,
    use_roles: bool,
) -> List[str]:
    """Constrained greedy roll-out; returns the FULL sequence (prefix + suffix)."""
    seq = list(prefix)
    while len(seq) < max_len:
        nxt = next_step_topk(seq, family, ranker, candidate_vocab, k=1,
                             use_roles=use_roles)
        if not nxt:
            break
        step = nxt[0]
        seq.append(step)
        if step == _TERMINAL:
            break
    return seq


def _beam_complete(
    prefix: List[str],
    family: str,
    ranker,
    candidate_vocab: List[str],
    max_len: int,
    use_roles: bool,
    beam: int,
    branch: int = 4,
) -> List[str]:
    """Constrained beam search; returns the FULL sequence with best log-prob.

    Each live hypothesis expands to its top ``branch`` legal next steps. A
    hypothesis that emits ``SHIP LOT`` is finished. We return the best finished
    hypothesis (highest mean log-prob), else the best live one at ``max_len``.
    """
    import math

    Beam = List[tuple]  # (seq, logprob, finished)
    beams: Beam = [(list(prefix), 0.0, False)]
    finished: Beam = []

    while beams:
        # Stop once every live beam is finished or has hit max_len.
        nxt_beams: Beam = []
        for seq, lp, fin in beams:
            if fin or len(seq) >= max_len:
                finished.append((seq, lp, True))
                continue
            v = grammar.valid_next_set(seq, candidate_vocab, use_roles)
            scores = ranker.predict(seq, family, v or candidate_vocab)
            for step in _rank(scores)[:branch]:
                p = max(scores.get(step, 1e-12), 1e-12)
                child_lp = lp + math.log(p)
                is_fin = step == _TERMINAL
                nxt_beams.append((seq + [step], child_lp, is_fin))
        if not nxt_beams:
            break
        # Keep the top-`beam` by length-normalised log-prob.
        nxt_beams.sort(key=lambda h: h[1] / max(len(h[0]), 1), reverse=True)
        beams = nxt_beams[:beam]
        # Early stop: enough finished and no live beam beats them meaningfully.
        if len(finished) >= beam and all(h[2] for h in beams):
            finished.extend(beams)
            break

    pool = finished or beams
    if not pool:
        return list(prefix)
    pool.sort(key=lambda h: h[1] / max(len(h[0]), 1), reverse=True)
    return pool[0][0]


def complete(
    prefix: Sequence[str],
    family: str,
    ranker,
    candidate_vocab: Iterable[str],
    max_len: int = 220,
    use_roles: bool = False,
    beam: int = 1,
) -> List[str]:
    """Complete a partial sequence and return ONLY the steps after ``prefix``.

    Constrained greedy (``beam<=1``) or beam search until ``SHIP LOT`` or
    ``max_len``, then ``repair`` the full sequence as a safety net. The returned
    list is the suffix (the predicted remaining steps), matching the Task-2
    submission convention.
    """
    prefix = list(prefix)
    candidate_vocab = list(candidate_vocab)
    cut = len(prefix)

    if beam and beam > 1:
        full = _beam_complete(prefix, family, ranker, candidate_vocab, max_len,
                              use_roles, beam)
    else:
        full = _greedy_complete(prefix, family, ranker, candidate_vocab, max_len,
                                use_roles)

    full = repair(full)
    # Defensive: repair may legitimately insert a prerequisite *inside* the
    # original prefix region; the suffix is everything beyond the original cut.
    if len(full) < cut:
        return full[cut:]
    return full[cut:]


# ---------------------------------------------------------------------------
# Symbolic repair
# ---------------------------------------------------------------------------

# Canonical prerequisite step strings used by the repair inserters. These are the
# universal known-vocab steps; for novel-vocab sequences we additionally fall
# back to a role-matched step already present in the sequence.
_CLEAN_STEP = "WAFER SURFACE CLEAN"
_DEVELOP_STEP = "DEVELOP PHOTORESIST"
_TERMINAL_STEP = "SHIP LOT"
_WAFER_SORT = "WAFER SORT TEST"
_CURE_PASSIVATION = "CURE PASSIVATION"


def _role_of(step: str) -> str:
    return roles.induce_role(step)


def _find_prereq(steps: Sequence[str], idx: int, role: str,
                 default: str) -> str:
    """Pick a prerequisite step of ``role`` to insert.

    Prefer a step of that role already seen earlier in the sequence (keeps
    novel-vocab families self-consistent); otherwise use the canonical default.
    """
    for j in range(idx - 1, -1, -1):
        if _role_of(steps[j]) == role:
            return steps[j]
    # Then anywhere in the sequence (handles role appearing only later).
    for s in steps:
        if _role_of(s) == role:
            return s
    return default


def _fix_one(steps: List[str], viol) -> Optional[List[str]]:
    """Apply a single role-based fix for one violation. Returns a new list or None."""
    rule = viol.rule
    idx = viol.step_index
    n = len(steps)
    idx = max(0, min(idx, n - 1))

    # 1) Missing CLEAN before a deposition.
    if rule == "RULE_DEP_NO_CLEAN":
        clean = _find_prereq(steps, idx, "CLEAN", _CLEAN_STEP)
        return steps[:idx] + [clean] + steps[idx:]

    # 2) Missing litho/mask before a patterned etch / metal etch / implant window.
    if rule in ("RULE_ETCH_NO_MASK", "RULE_METAL_ETCH_NO_LITHO",
                "RULE_IMPLANT_NO_MASK"):
        develop = _find_prereq(steps, idx, "LITHO", _DEVELOP_STEP)
        # Ensure we insert an actual develop/pattern step, not just any litho op.
        if _role_of(develop) != "LITHO":
            develop = _DEVELOP_STEP
        return steps[:idx] + [develop] + steps[idx:]

    # 3) Missing deposition/fill before a CMP.
    if rule == "RULE_CMP_NO_DEP":
        dep = _find_prereq(steps, idx, "THERMAL_DEP", "DEPOSIT INTERLEVEL DIELECTRIC")
        if _role_of(dep) not in ("THERMAL_DEP", "FILL"):
            dep = "DEPOSIT INTERLEVEL DIELECTRIC"
        return steps[:idx] + [dep] + steps[idx:]

    # 4) Litho mask level skip — drop the offending out-of-order litho block step
    #    (removing the renumbered align/expose realigns the level sequence).
    if rule == "RULE_LITHO_LEVEL_SKIP":
        return steps[:idx] + steps[idx + 1:]

    # 5) SHIP LOT before WAFER SORT TEST — move SHIP LOT to the end.
    if rule == "RULE_SHIP_BEFORE_TEST":
        rest = [s for s in steps if s != _TERMINAL_STEP]
        # Place after the last test step if present, else at the very end.
        ti = max((i for i, s in enumerate(rest) if _role_of(s) == "TEST"), default=len(rest) - 1)
        return rest[:ti + 1] + [_TERMINAL_STEP] + rest[ti + 1:]

    # 6) TEST before passivation cured — move the offending test after passivation.
    if rule == "RULE_TEST_BEFORE_PASSIVATION":
        step = steps[idx]
        rest = steps[:idx] + steps[idx + 1:]
        ci = next((i for i, s in enumerate(rest)
                   if s == _CURE_PASSIVATION or _role_of(s) == "PASSIVATION"), None)
        if ci is None:
            return rest + [step]
        return rest[:ci + 1] + [step] + rest[ci + 1:]

    # 7) Pad-window opened before its dielectric/passivation deposition — move the
    #    pad-open step later, after the passivation deposition.
    if rule == "RULE_PAD_OPEN_BEFORE_DEP":
        step = steps[idx]
        rest = steps[:idx] + steps[idx + 1:]
        pi = next((i for i, s in enumerate(rest)
                   if _role_of(s) in ("THERMAL_DEP", "PASSIVATION")), None)
        if pi is None:
            return rest + [step]
        return rest[:pi + 1] + [step] + rest[pi + 1:]

    # 8) Backside metal before passivation cured — move it after passivation.
    if rule == "RULE_BACKSIDE_BEFORE_PASSIVATION":
        step = steps[idx]
        rest = steps[:idx] + steps[idx + 1:]
        ci = next((i for i, s in enumerate(rest)
                   if s == _CURE_PASSIVATION or _role_of(s) == "PASSIVATION"), None)
        if ci is None:
            return rest + [step]
        return rest[:ci + 1] + [step] + rest[ci + 1:]

    # Unknown rule id: drop the offending step as a generic last-resort fix.
    return steps[:idx] + steps[idx + 1:]


def repair(steps: Sequence[str], max_passes: int = 3) -> List[str]:
    """Bounded greedy repair of the first role-augmented-validator violation.

    Each pass fixes the earliest violation and re-validates. Returns the best
    sequence reached: a fully valid one if found within ``max_passes``, else the
    pass that produced the fewest violations (best-effort).
    """
    best = list(steps)
    best_n = len(rules.validate_with_roles(best))
    if best_n == 0:
        return best

    cur = list(steps)
    for _ in range(max_passes):
        viols = rules.validate_with_roles(cur)
        if not viols:
            return cur
        first = min(viols, key=lambda v: v.step_index)
        nxt = _fix_one(list(cur), first)
        if nxt is None or nxt == cur:
            break
        n = len(rules.validate_with_roles(nxt))
        if n < best_n:
            best, best_n = nxt, n
        if n == 0:
            return nxt
        cur = nxt
    return best


# ---------------------------------------------------------------------------
# Self-test (PPM ranker — symbolic core only, no torch).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from nspe.data import candidate_vocab, load_family
    from nspe.ppm import PPM

    print("=" * 64)
    print("decode.py self-test  (PPM ranker)")
    print("=" * 64)

    train = {
        "mosfet": [list(s) for s in load_family("mosfet")[:200]],
        "igbt": [list(s) for s in load_family("igbt")[:200]],
    }
    ranker = PPM().fit(train)
    cand = candidate_vocab(("mosfet", "igbt"))  # train-family vocab (OOD vs ic)

    # ---- next_step_topk: exactly k, distinct, gold-in-set, gold often Top-1 ----
    print("\n[next_step_topk] on 60 ic prefixes (held-out family) ...")
    ic_seqs = [list(s) for s in load_family("ic")[:30]]
    n_eval = top1 = top5 = 0
    gold_in_vocab = gold_in_set = 0  # legality guarantee holds when gold is in vocab
    for seq in ic_seqs:
        for frac in (0.6, 0.8):
            cut = max(1, min(len(seq) - 1, round(len(seq) * frac)))
            prefix, gold = seq[:cut], seq[cut]
            preds = next_step_topk(prefix, "ic", ranker, cand, k=5, role_sharpen=True)
            assert len(preds) == 5, f"expected 5 preds, got {len(preds)}"
            assert len(set(preds)) == 5, "preds not distinct"
            n_eval += 1
            top1 += preds[0] == gold
            top5 += gold in preds
            # Core guarantee: when the gold step exists in the candidate vocab it
            # is ALWAYS legal (never excluded by the rule filter). A novel OOD step
            # absent from the train-family vocab cannot be in any vocab-built set.
            if gold in cand:
                gold_in_vocab += 1
                v = grammar.valid_next_set(prefix, cand)
                gold_in_set += gold in v
    print(f"  evaluated   : {n_eval}")
    print(f"  Top-1       : {top1}/{n_eval} = {top1/n_eval:.3f}")
    print(f"  Top-5       : {top5}/{n_eval} = {top5/n_eval:.3f}")
    print(f"  gold-in-set : {gold_in_set}/{gold_in_vocab} (of in-vocab golds; "
          f"{n_eval - gold_in_vocab} novel OOD steps absent from train vocab)")
    assert gold_in_set == gold_in_vocab, "in-vocab gold must always be legal"

    # ---- complete: returns suffix only; full sequence is rule-valid ----
    print("\n[complete] greedy + beam, ic prefixes; assert every output valid ...")
    n_done = valid_greedy = valid_beam = 0
    for seq in ic_seqs[:15]:
        cut = max(1, min(len(seq) - 1, round(len(seq) * 0.8)))
        prefix = seq[:cut]
        suffix_g = complete(prefix, "ic", ranker, cand, use_roles=False, beam=1)
        suffix_b = complete(prefix, "ic", ranker, cand, use_roles=False, beam=4)
        # Suffix must not repeat the partial; reconstructed full must be valid.
        full_g = prefix + suffix_g
        full_b = prefix + suffix_b
        valid_greedy += len(rules.validate_with_roles(full_g)) == 0
        valid_beam += len(rules.validate_with_roles(full_b)) == 0
        n_done += 1
    print(f"  completions       : {n_done}")
    print(f"  greedy all-valid  : {valid_greedy}/{n_done}")
    print(f"  beam   all-valid  : {valid_beam}/{n_done}")
    assert valid_greedy == n_done, "every greedy completion must be rule-valid"
    assert valid_beam == n_done, "every beam completion must be rule-valid"

    # ---- repair: turns a deliberately corrupted sequence valid ----
    print("\n[repair] fix an injected DEP_NO_CLEAN ...")
    base = list(load_family("mosfet")[0])
    di = next(i for i, s in enumerate(base) if s.upper().startswith("DEPOSIT "))
    corrupt = [s for j, s in enumerate(base)
               if not (j < di and _role_of(s) == "CLEAN")]
    before = rules.first_rule(corrupt, use_roles=True)
    fixed = repair(corrupt)
    after = rules.first_rule(fixed, use_roles=True)
    print(f"  before repair: first_rule = {before}")
    print(f"  after  repair: first_rule = {after}  (len {len(corrupt)} -> {len(fixed)})")
    assert before is not None, "corruption did not trigger a violation"
    assert after is None, "repair failed to make the sequence valid"

    print("\nSELF-TEST PASSED")
