"""Inject each of the 10 process-logic rule violations into a valid sequence.

This module manufactures *anomalous* sequences for the Task-3 (anomaly) study,
both in-distribution (known vocabulary) and out-of-distribution (a `novel=True`
variant that renames the trigger step to an unseen string, to test whether
role-induction in `nspe.rules.validate_with_roles` recovers the violation that
the stock validator — keyed on exact step strings — would miss).

Design contract for every corruptor
------------------------------------
* It takes a *valid* sequence and an RNG and returns a corrupted **copy** (the
  input list is never mutated) that triggers exactly ONE intended rule, or
  ``None`` if the source sequence lacks the trigger structure to corrupt.
* The hard correctness gate (enforced by ``_verify``) is that, on known
  vocabulary, ``nspe.rules.first_rule(result, use_roles=False) == rule_id``. The
  earliest-positioned violation must be the intended one — important because the
  official validator may report several violations and ``first_rule`` returns the
  one with the smallest ``step_index``.

The DEP_NO_CLEAN corruptor reuses the robust, confound-free trick validated in
``experiments/ood_symbolic_probe.py``: insert the deposition step just before
``SHIP LOT`` after blanking any clean-like step in the prior-12 window. That
position is past every other rule trigger, so the injected deposition is the
sole — hence earliest — violation.

No torch. Pure stdlib + the symbolic core (official validator via nspe.rules).
"""
from __future__ import annotations

import random
import uuid
from typing import Callable, Optional

from nspe.official import gs
from nspe.roles import induce_role
from nspe.rules import first_rule, validate, validate_with_roles

# A benign filler that is neither a clean, deposition, etch, implant, litho,
# test, nor logistics trigger — safe to overwrite any step with.
_FILLER = "MEASURE THICKNESS"

# Unseen (family-4) trigger/anchor names used by the ``novel=True`` path.
#
# None of these is a member of any official frozenset (verified against
# ``rules.known_vocab()``), so the stock validator — which keys on exact step
# strings / hardcoded literals — is BLIND to them. Each is, however,
# KEYWORD-PRESERVING in exactly the way ``rules._canonical_landmark`` recognizes,
# so ``validate_with_roles`` canonicalizes it back to its landmark and recovers
# the violation. This is the surface-rename failure mode a renaming 4th family
# induces, and the pair (stock-blind, role-caught) is what each novel corruptor
# demonstrates.
#
#   novel string                     canonicalizes back to        rule(s)
#   -------------------------------  ---------------------------  ---------------
_NOVEL_DEP = "DEPOSIT SIC POLY"          # DEPOSIT POLYSILICON     DEP_NO_CLEAN
_NOVEL_ETCH = "OXIDE DRY ETCH STEP"      # OXIDE ETCH              ETCH_NO_MASK
_NOVEL_METAL_ETCH = "METAL DRY ETCH STEP"  # METAL ETCH            METAL_ETCH_NO_LITHO
_NOVEL_IMPLANT = "IMPLANT WELL REGION"   # IMPLANT WELL            IMPLANT_NO_MASK
_NOVEL_CMP = "CMP PLANARIZE STEP"        # CMP METAL               CMP_NO_DEP
_NOVEL_PAD = "OPEN BOND PAD WINDOW AREA"  # OPEN PAD WINDOW         PAD_OPEN_BEFORE_DEP
_NOVEL_ELECTRICAL_TEST = "LEAKAGE SCREEN TEST"  # LEAKAGE TEST      TEST_BEFORE_PASSIVATION
_NOVEL_SHIP = "SHIP PRODUCT LOT"         # SHIP LOT                SHIP_BEFORE_TEST
_NOVEL_BACKSIDE = "DEPOSIT BACKSIDE CONTACT METAL"  # DEPOSIT BACKSIDE METAL  BACKSIDE_BEFORE_PASSIVATION


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _clean_like(step: str) -> bool:
    """Conservative clean detector matching both the validator frozenset and the
    role-induction surface patterns (so blanking works for ID *and* novel)."""
    return step in gs.CLEAN_STEPS or induce_role(step) == "CLEAN"


def _first_index(steps: list[str], targets) -> Optional[int]:
    for i, s in enumerate(steps):
        if s in targets:
            return i
    return None


def _last_index(steps: list[str], targets) -> Optional[int]:
    found = None
    for i, s in enumerate(steps):
        if s in targets:
            found = i
    return found


def _verify(result: Optional[list[str]], rule_id: str,
            novel: bool = False) -> Optional[list[str]]:
    """Gate a candidate corruption.

    Known-vocab (novel=False): the stock validator's earliest violation must be
    exactly `rule_id`.

    Novel-vocab (novel=True): the stock validator must MISS it (be blind), while
    the role-augmented validator's earliest violation must be `rule_id`.
    """
    if result is None:
        return None
    if not novel:
        return result if first_rule(result, use_roles=False) == rule_id else None
    # novel: stock validator blind, role-augmented validator catches it.
    if first_rule(result, use_roles=False) == rule_id:
        return None  # not actually exercising the novel/role path
    if first_rule(result, use_roles=True) == rule_id:
        return result
    return None


# --------------------------------------------------------------------------- #
# Corruptors — one per rule. Each returns a corrupted copy or None.
# --------------------------------------------------------------------------- #

def corrupt_dep_no_clean(seq: list[str], rng: random.Random,
                         novel: bool = False) -> Optional[list[str]]:
    """Insert a deposition just before SHIP LOT, after blanking clean-like steps
    in the prior-12 window. Robust trick from ood_symbolic_probe.py."""
    steps = list(seq)
    j = _first_index(steps, ("SHIP LOT",))
    if j is None:
        return None
    for k in range(max(0, j - 12), j):
        if _clean_like(steps[k]):
            steps[k] = _FILLER
    dep = _NOVEL_DEP if novel else "DEPOSIT POLYSILICON"
    steps.insert(j, dep)
    return _verify(steps, "RULE_DEP_NO_CLEAN", novel)


def corrupt_metal_etch_no_litho(seq: list[str], rng: random.Random,
                                novel: bool = False) -> Optional[list[str]]:
    """Blank the EXPOSE LITHO LEVEL and DEVELOP steps in the prior-15 window of a
    metal etch, so the metal etch loses its photoresist mask.

    Novel variant: in addition to blanking the litho prereqs, rename the metal
    etch trigger itself to ``_NOVEL_METAL_ETCH`` (``METAL DRY ETCH STEP``). The
    stock validator no longer recognizes it as a metal-etch step (nor as any etch
    step — the string is in neither ``METAL_ETCH_STEPS`` nor ``ETCH_STEPS``) so it
    is blind, whereas ``validate_with_roles`` canonicalizes it back to
    ``METAL ETCH`` and fires RULE_METAL_ETCH_NO_LITHO."""
    steps = list(seq)
    mi = _last_index(steps, gs.METAL_ETCH_STEPS)
    if mi is None:
        return None
    # Blank EXPOSE LITHO LEVEL* and DEVELOP* in the prior-15 window. To avoid
    # tripping RULE_ETCH_NO_MASK first at an *earlier* patterned etch, we only
    # touch the window of THIS metal etch and replace litho steps with filler.
    changed = False
    for k in range(max(0, mi - 15), mi):
        s = steps[k]
        if s.startswith("EXPOSE LITHO LEVEL") or s in ("DEVELOP PHOTORESIST",
                                                        "DEVELOP PAD WINDOW"):
            steps[k] = _FILLER
            changed = True
    if not changed:
        return None
    if novel:
        steps[mi] = _NOVEL_METAL_ETCH
    return _verify(steps, "RULE_METAL_ETCH_NO_LITHO", novel)


def corrupt_etch_no_mask(seq: list[str], rng: random.Random,
                         novel: bool = False) -> Optional[list[str]]:
    """Blank the DEVELOP step preceding a *non-metal* patterned etch so it has no
    mask. We pick the EARLIEST patterned etch and clear DEVELOP in its prior-12,
    making it the earliest violation. For novel=True we additionally rename that
    etch step to an unseen string so only role-induction sees it as an etch."""
    steps = list(seq)
    # candidate non-metal patterned etches (RULE_ETCH_NO_MASK triggers).
    etch_targets = gs.ETCH_STEPS - gs.METAL_ETCH_STEPS
    ei = _first_index(steps, etch_targets)
    if ei is None:
        return None
    develop_window = ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW")
    cleared = False
    for k in range(max(0, ei - 12), ei):
        if steps[k] in develop_window:
            steps[k] = _FILLER
            cleared = True
    if not cleared:
        return None
    if novel:
        steps[ei] = _NOVEL_ETCH
    return _verify(steps, "RULE_ETCH_NO_MASK", novel)


def corrupt_litho_level_skip(seq: list[str], rng: random.Random,
                             novel: bool = False) -> Optional[list[str]]:
    """Renumber an ``ALIGN MASK LEVEL n`` -> ``n+2`` to skip a level.

    STRUCTURAL rule — intentionally KNOWN-VOCAB ONLY (no novel variant). Unlike
    every other rule, RULE_LITHO_LEVEL_SKIP does not key on a renamable surface
    anchor: it keys on the *integer ordering* of the ``ALIGN MASK LEVEL n``
    levels. There is no surface-rename failure mode for a 4th family to induce
    here — the level integers are structural, not vocabulary — so canonicalization
    has nothing to recover and ``validate_with_roles`` would behave identically to
    ``validate``. We therefore exclude it from ``NOVEL_CAPABLE`` and return None on
    the novel path; ``make_anomaly_set(novel=True)`` still covers this rule via its
    standard known-vocab injection."""
    if novel:
        return None
    steps = list(seq)
    aligns = [i for i, s in enumerate(steps)
              if s.startswith("ALIGN MASK LEVEL ")
              and s.split("ALIGN MASK LEVEL ")[1].isdigit()]
    if len(aligns) < 2:
        return None
    # Bump the SECOND align level by +2 so the jump from the first is > +1 while
    # keeping every later align >= this one (they were ascending originally, so
    # bumping the earliest non-first by +2 yields the earliest skip-violation).
    idx = aligns[1]
    lvl = int(steps[idx].split("ALIGN MASK LEVEL ")[1])
    steps[idx] = f"ALIGN MASK LEVEL {lvl + 2}"
    return _verify(steps, "RULE_LITHO_LEVEL_SKIP", novel)


def corrupt_implant_no_mask(seq: list[str], rng: random.Random,
                            novel: bool = False) -> Optional[list[str]]:
    """Blank every implant-opener (oxide etch / develop) in the prior-15 window of
    the EARLIEST implant, so the implant has no open window. For novel=True we
    rename that implant to an unseen string (role-induction recovers IMPLANT)."""
    steps = list(seq)
    ii = _first_index(steps, gs.IMPLANT_STEPS)
    if ii is None:
        return None
    cleared = False
    for k in range(max(0, ii - 15), ii):
        if steps[k] in gs.IMPLANT_OPENER_STEPS:
            steps[k] = _FILLER
            cleared = True
    if not cleared:
        return None
    if novel:
        steps[ii] = _NOVEL_IMPLANT
    return _verify(steps, "RULE_IMPLANT_NO_MASK", novel)


def corrupt_cmp_no_dep(seq: list[str], rng: random.Random,
                       novel: bool = False) -> Optional[list[str]]:
    """Blank every deposition/fill step in the prior-6 window of the EARLIEST CMP,
    so there is nothing to planarize.

    Novel variant: blank the feeding deposition/fill exactly as in the ID case,
    then rename the CMP TRIGGER itself to ``_NOVEL_CMP`` (``CMP PLANARIZE STEP``).
    The stock validator no longer sees a CMP step (the string is not in
    ``CMP_STEPS``), so it is blind; ``validate_with_roles`` canonicalizes it back
    to ``CMP METAL`` and fires RULE_CMP_NO_DEP. (Renaming the *deposition* instead
    would be self-defeating — role-induction would re-classify it as FILL and the
    violation would vanish — so the trigger, not the prereq, carries the novel
    string here.)"""
    steps = list(seq)
    ci = _first_index(steps, gs.CMP_STEPS)
    if ci is None:
        return None
    cleared = False
    for k in range(max(0, ci - 6), ci):
        if steps[k] in gs.FILL_STEPS:
            steps[k] = _FILLER
            cleared = True
    if not cleared:
        return None
    if novel:
        steps[ci] = _NOVEL_CMP
    return _verify(steps, "RULE_CMP_NO_DEP", novel)


def corrupt_pad_open_before_dep(seq: list[str], rng: random.Random,
                                novel: bool = False) -> Optional[list[str]]:
    """Move a pad-window step to before DEPOSIT PASSIVATION (ordering rule).

    Novel variant: rename the pad-window anchor to ``_NOVEL_PAD``
    (``OPEN BOND PAD WINDOW AREA``) before moving it. The stock validator does not
    recognize it as a pad-window step (string absent from ``PAD_WINDOW_STEPS``), so
    the misorder is invisible; ``validate_with_roles`` canonicalizes it back to
    ``OPEN PAD WINDOW`` and fires RULE_PAD_OPEN_BEFORE_DEP. (The ``_canonical``
    rule checks DEVELOP before PAD+WINDOW, so this AREA string — having no DEVELOP
    keyword — maps to the pad-window opener, not a develop step.)"""
    steps = list(seq)
    dep_idx = _first_index(steps, ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"))
    if dep_idx is None:
        return None
    pad_idx = _first_index(steps, gs.PAD_WINDOW_STEPS)
    if pad_idx is None or pad_idx <= dep_idx:
        return None
    pad = steps.pop(pad_idx)
    if novel:
        pad = _NOVEL_PAD
    steps.insert(dep_idx, pad)  # now the pad window precedes passivation deposition
    return _verify(steps, "RULE_PAD_OPEN_BEFORE_DEP", novel)


def corrupt_test_before_passivation(seq: list[str], rng: random.Random,
                                    novel: bool = False) -> Optional[list[str]]:
    """Move an electrical test to before CURE PASSIVATION (ordering rule).

    Novel variant: rename the electrical-test anchor to ``_NOVEL_ELECTRICAL_TEST``
    (``LEAKAGE SCREEN TEST``) before moving it. The stock validator does not
    recognize it as an electrical test (string absent from
    ``ELECTRICAL_TEST_STEPS``) so the misorder is invisible; the keyword-gated
    ``_canonical_landmark`` (``* TEST`` carrying an electrical keyword such as
    LEAKAGE) maps it back to ``LEAKAGE TEST`` and fires
    RULE_TEST_BEFORE_PASSIVATION."""
    steps = list(seq)
    cure_idx = _first_index(steps, ("CURE PASSIVATION",))
    if cure_idx is None:
        return None
    test_idx = next((i for i, s in enumerate(steps)
                     if s in gs.ELECTRICAL_TEST_STEPS and i > cure_idx), None)
    if test_idx is None:
        return None
    test = steps.pop(test_idx)
    if novel:
        test = _NOVEL_ELECTRICAL_TEST
    steps.insert(cure_idx, test)  # test now precedes CURE PASSIVATION
    return _verify(steps, "RULE_TEST_BEFORE_PASSIVATION", novel)


def corrupt_ship_before_test(seq: list[str], rng: random.Random,
                             novel: bool = False) -> Optional[list[str]]:
    """Move SHIP LOT to before WAFER SORT TEST (ordering rule).

    Novel variant: rename the ship anchor to ``_NOVEL_SHIP``
    (``SHIP PRODUCT LOT``) before moving it. The stock validator finds no literal
    ``SHIP LOT`` (its ``ship_idx`` is None) so the rule cannot fire — it is blind;
    ``validate_with_roles`` canonicalizes it back to ``SHIP LOT`` and, now placed
    before the (still-known) ``WAFER SORT TEST``, fires RULE_SHIP_BEFORE_TEST."""
    steps = list(seq)
    ship_idx = _first_index(steps, ("SHIP LOT",))
    sort_idx = _first_index(steps, ("WAFER SORT TEST",))
    if ship_idx is None or sort_idx is None or ship_idx < sort_idx:
        return None
    ship = steps.pop(ship_idx)
    if novel:
        ship = _NOVEL_SHIP
    steps.insert(sort_idx, ship)  # SHIP LOT now precedes WAFER SORT TEST
    return _verify(steps, "RULE_SHIP_BEFORE_TEST", novel)


def corrupt_backside_before_passivation(seq: list[str], rng: random.Random,
                                        novel: bool = False) -> Optional[list[str]]:
    """Move DEPOSIT BACKSIDE METAL to before CURE PASSIVATION (ordering rule).

    We insert the backside metal at ``cure_idx`` (just before the first
    CURE PASSIVATION) so a clean still sits in its prior-12 window in the original
    flow; the verifier confirms the earliest violation is backside (not an
    incidental RULE_DEP_NO_CLEAN on the moved deposition).

    Novel variant: rename the backside-metal trigger to ``_NOVEL_BACKSIDE``
    (``DEPOSIT BACKSIDE CONTACT METAL``) before moving it. The stock validator
    does not recognize it as a backside-metal step (string absent from
    ``BACKSIDE_METAL_STEPS``) so the misorder is invisible; the
    DEPOSIT+BACKSIDE+METAL canonicalization maps it back to
    ``DEPOSIT BACKSIDE METAL`` and fires RULE_BACKSIDE_BEFORE_PASSIVATION."""
    steps = list(seq)
    cure_idx = _first_index(steps, ("CURE PASSIVATION",))
    if cure_idx is None:
        return None
    bk_idx = next((i for i, s in enumerate(steps)
                   if s in gs.BACKSIDE_METAL_STEPS and i > cure_idx), None)
    if bk_idx is None:
        return None
    bk = steps.pop(bk_idx)
    if novel:
        bk = _NOVEL_BACKSIDE
    steps.insert(cure_idx, bk)  # backside metal now precedes CURE PASSIVATION
    return _verify(steps, "RULE_BACKSIDE_BEFORE_PASSIVATION", novel)


# --------------------------------------------------------------------------- #
# Registry + anomaly-set builder
# --------------------------------------------------------------------------- #

CORRUPTORS: dict[str, Callable[[list[str], random.Random], Optional[list[str]]]] = {
    "RULE_DEP_NO_CLEAN": corrupt_dep_no_clean,
    "RULE_METAL_ETCH_NO_LITHO": corrupt_metal_etch_no_litho,
    "RULE_ETCH_NO_MASK": corrupt_etch_no_mask,
    "RULE_LITHO_LEVEL_SKIP": corrupt_litho_level_skip,
    "RULE_IMPLANT_NO_MASK": corrupt_implant_no_mask,
    "RULE_CMP_NO_DEP": corrupt_cmp_no_dep,
    "RULE_PAD_OPEN_BEFORE_DEP": corrupt_pad_open_before_dep,
    "RULE_TEST_BEFORE_PASSIVATION": corrupt_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST": corrupt_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": corrupt_backside_before_passivation,
}

# Rules whose corruptors expose a meaningful novel-vocab variant — i.e. the rule
# keys on a renamable SURFACE anchor (a trigger or an ordering landmark) that a
# 4th family could rename to an unseen string. For each of these the novel
# corruptor renames that anchor to a keyword-preserving novel string, so the
# stock validator (exact-string keyed) is BLIND while `validate_with_roles`
# canonicalizes the anchor back and recovers the violation.
#
# 9 of 10 rules qualify. The sole exception is RULE_LITHO_LEVEL_SKIP, which is
# STRUCTURAL (keyed on ALIGN MASK LEVEL integer ordering, not on a renamable
# string) and therefore has no surface-rename failure mode — see
# `corrupt_litho_level_skip`. It stays known-vocab only.
NOVEL_CAPABLE: frozenset = frozenset({
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
})


def make_anomaly_set(valid_seqs, rng: random.Random, novel: bool = False,
                     frac_invalid: float = 0.4) -> list[dict]:
    """Build a shuffled mix of valid and corrupted sequences.

    Parameters
    ----------
    valid_seqs : iterable of list[str]
        Pool of valid sequences to draw from.
    rng : random.Random
        Seeded RNG.
    novel : bool
        If True, corrupt using novel-vocab variants for the windowed rules that
        support them (DEP / ETCH / IMPLANT / CMP per the spec) so the resulting
        anomalies are invisible to the stock validator but caught by the
        role-augmented one. Ordering rules have no novel variant and fall back to
        their standard (known-vocab) injection so the set still spans all rules.
    frac_invalid : float
        Target fraction of the output that is invalid (corrupted).

    Returns
    -------
    list[dict] with keys EXAMPLE_ID, SEQUENCE (list[str]), IS_VALID (1/0),
    VIOLATION_RULE (str or None), shuffled.
    """
    pool = [list(s) for s in valid_seqs]
    rng.shuffle(pool)
    n = len(pool)
    n_invalid = int(round(n * frac_invalid))

    rules = list(CORRUPTORS.keys())
    if novel:
        # Prioritise novel-capable rules so the set actually exercises the role
        # path; ordering rules remain (known-vocab) to keep coverage.
        rules = sorted(rules, key=lambda r: (r not in NOVEL_CAPABLE, r))

    out: list[dict] = []
    made_invalid = 0
    ri = 0
    for seq in pool:
        want_invalid = made_invalid < n_invalid
        produced = None
        rule_used = None
        if want_invalid:
            # Try rules in round-robin until one corrupts this sequence.
            for _ in range(len(rules)):
                rule = rules[ri % len(rules)]
                ri += 1
                use_novel = novel and (rule in NOVEL_CAPABLE)
                produced = CORRUPTORS[rule](list(seq), rng, novel=use_novel)
                if produced is not None:
                    rule_used = rule
                    break
        if produced is not None:
            made_invalid += 1
            out.append({
                "EXAMPLE_ID": uuid.uuid4().hex[:12],
                "SEQUENCE": produced,
                "IS_VALID": 0,
                "VIOLATION_RULE": rule_used,
            })
        else:
            out.append({
                "EXAMPLE_ID": uuid.uuid4().hex[:12],
                "SEQUENCE": list(seq),
                "IS_VALID": 1,
                "VIOLATION_RULE": None,
            })
    rng.shuffle(out)
    return out


__all__ = [
    "CORRUPTORS", "NOVEL_CAPABLE", "make_anomaly_set",
    "corrupt_dep_no_clean", "corrupt_metal_etch_no_litho", "corrupt_etch_no_mask",
    "corrupt_litho_level_skip", "corrupt_implant_no_mask", "corrupt_cmp_no_dep",
    "corrupt_pad_open_before_dep", "corrupt_test_before_passivation",
    "corrupt_ship_before_test", "corrupt_backside_before_passivation",
]


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    from collections import defaultdict

    from nspe.data import all_sequences

    rng = random.Random(20240530)
    pairs = all_sequences()
    rng.shuffle(pairs)

    # Draw a pool that spans all 3 families for each rule attempt.
    by_family: dict[str, list] = defaultdict(list)
    for fam, seq in pairs:
        by_family[fam].append(seq)

    def draw(n: int) -> list[list[str]]:
        out = []
        fams = list(by_family)
        i = 0
        while len(out) < n:
            fam = fams[i % len(fams)]
            lst = by_family[fam]
            out.append(lst[(i // len(fams)) % len(lst)])
            i += 1
        return out

    print("=" * 72)
    print("Per-rule ID corruption success (40 attempts each, all 3 families)")
    print("=" * 72)
    all_ok = True
    for rule, fn in CORRUPTORS.items():
        sample = draw(40)
        produced = 0
        correct = 0
        for s in sample:
            r = fn(list(s), rng, novel=False)
            if r is None:
                continue
            produced += 1
            if first_rule(r, use_roles=False) == rule:
                correct += 1
        frac = correct / produced if produced else 0.0
        status = "OK " if (produced > 0 and correct == produced) else "!! "
        if not (produced > 0 and correct == produced):
            all_ok = False
        print(f"  {status}{rule:32s} produced={produced:2d}/40 "
              f"exact-rule={correct:2d}/{produced:2d} frac={frac:.3f}")

    print("\n" + "=" * 72)
    print("Novel-vocab corruption: stock validator MISSES, role-aug CATCHES")
    print("=" * 72)
    for rule in sorted(NOVEL_CAPABLE):
        fn = CORRUPTORS[rule]
        sample = draw(40)
        produced = 0
        stock_blind = 0
        role_catches = 0
        for s in sample:
            r = fn(list(s), rng, novel=True)
            if r is None:
                continue
            produced += 1
            if first_rule(r, use_roles=False) != rule:
                stock_blind += 1
            if first_rule(r, use_roles=True) == rule:
                role_catches += 1
        print(f"  {rule:32s} produced={produced:2d}/40 "
              f"stock-blind={stock_blind:2d} role-catches={role_catches:2d}")

    print("\n" + "=" * 72)
    print("make_anomaly_set sanity")
    print("=" * 72)
    valids = draw(200)
    # ID set
    aset = make_anomaly_set(valids, random.Random(1), novel=False, frac_invalid=0.4)
    n_inv = sum(1 for r in aset if r["IS_VALID"] == 0)
    # every invalid row's stock first_rule must equal its recorded VIOLATION_RULE
    id_attr_ok = all(
        first_rule(r["SEQUENCE"], use_roles=False) == r["VIOLATION_RULE"]
        for r in aset if r["IS_VALID"] == 0
    )
    # every valid row must actually validate clean
    id_valids_clean = all(
        not validate(r["SEQUENCE"]) for r in aset if r["IS_VALID"] == 1
    )
    print(f"  ID set: n={len(aset)} invalid={n_inv} "
          f"({n_inv/len(aset):.2f})  attribution_ok={id_attr_ok} "
          f"valids_clean={id_valids_clean}")

    # Novel set: invalid windowed rows should be MISSED by stock, CAUGHT by role-aug.
    nset = make_anomaly_set(valids, random.Random(2), novel=True, frac_invalid=0.4)
    n_ninv = sum(1 for r in nset if r["IS_VALID"] == 0)
    novel_rows = [r for r in nset if r["IS_VALID"] == 0
                  and r["VIOLATION_RULE"] in NOVEL_CAPABLE]
    # how many novel-capable invalid rows are stock-blind but role-caught
    stock_blind = sum(1 for r in novel_rows
                      if first_rule(r["SEQUENCE"], use_roles=False) is None
                      or first_rule(r["SEQUENCE"], use_roles=False) != r["VIOLATION_RULE"])
    role_caught = sum(1 for r in novel_rows
                      if first_rule(r["SEQUENCE"], use_roles=True) == r["VIOLATION_RULE"])
    print(f"  Novel set: n={len(nset)} invalid={n_ninv} "
          f"novel-capable-invalid={len(novel_rows)} "
          f"stock-blind={stock_blind} role-caught={role_caught}")
    novel_demo_ok = (len(novel_rows) > 0 and stock_blind == len(novel_rows)
                     and role_caught == len(novel_rows))

    print("\n" + "=" * 72)
    if all_ok and id_attr_ok and id_valids_clean and novel_demo_ok:
        print("SELF-TEST PASSED")
        sys.exit(0)
    else:
        print("SELF-TEST FAILED "
              f"(per_rule={all_ok} id_attr={id_attr_ok} "
              f"valids_clean={id_valids_clean} novel_demo={novel_demo_ok})")
        sys.exit(1)
