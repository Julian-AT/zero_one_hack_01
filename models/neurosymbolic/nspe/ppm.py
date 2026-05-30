"""Symbolic role-factored variable-order Markov ranker (CPU, stdlib only).

This is the primary OOD-robust ranker and a core differentiator of NSPE. It is a
pure counting model — no torch, no training loop — built from two factored,
variable-order (PPM/Katz-style) Markov components:

  (a) ROLE model    P(role_t | role_{t-k..t-1}, family)
  (b) STEP model    P(step_t | role_t, step_{t-j..t-1}, family)

and a step is scored as the product of the two factors:

    score(c) = P(role(c) | role-context, family)            # which role comes next
             * P(c       | role(c), step-context, family)   # which step of that role

renormalized over the supplied candidate set (typically the symbolic
`valid_next_set`). This factoring is what makes the ranker transfer to an unseen
4th family: the ROLE context is shared across families (roles are induced via
`nspe.roles.induce_role`, which works on novel step strings), so the *next role*
is predictable even when the concrete step strings are entirely new. The STEP
factor then backs off, when the family-specific step context is unseen, to a
family-agnostic `P(step | role)` distribution, which keeps step selection sane on
novel vocabulary instead of collapsing.

Smoothing & backoff (documented choices)
-----------------------------------------
Both factors use *interpolation* (Jelinek–Mercer-style) across context orders
rather than hard Katz cut-off, because it is smoother and parameter-light:

  * ROLE model. For role-context length k = 4, 3, 2, 1, 0 we form a per-order
    estimate P_k(role | last-k-roles, family) with add-alpha smoothing, and mix
    them with weights that favour longer contexts that were actually *observed*:

        lambda_k(ctx) = COUNT(ctx) / (COUNT(ctx) + BACKOFF_BETA)

    i.e. the longer context gets a share proportional to how much evidence it has;
    the leftover (1 - lambda_k) mass recurses to the shorter context. Order 0
    (the family unigram over roles) is the final interpolation anchor, and a tiny
    add-alpha floor guarantees every role keeps non-zero probability.

  * STEP model. Identical interpolation scheme over step-context length
    j = 3, 2, 1, 0, but every distribution is *conditioned on the predicted role*
    so it only ever ranks steps of that role. The order-0 anchor is the
    family-specific P(step | role, family); if that role was never seen in this
    family (an OOD situation) it falls through to the family-agnostic
    P(step | role) pooled over all training families, and finally to a uniform
    distribution over the role's steps. Add-alpha smoothing again floors the
    probabilities.

Tunables (module constants): ROLE_ORDER=4, STEP_ORDER=3, ROLE_ALPHA, STEP_ALPHA,
ROLE_BETA, STEP_BETA. These were chosen to be robust defaults; the model is cheap
to refit so they can be swept by a caller.

Implements the RANKER PROTOCOL (duck-typed): predict / predict_roles / perplexity,
plus fit / save / load.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from nspe.roles import ROLES, induce_role

# ---------------------------------------------------------------------------
# Hyper-parameters (documented in the module docstring).
# ---------------------------------------------------------------------------
ROLE_ORDER = 4          # max role-context length (k = 4 .. 0)
STEP_ORDER = 3          # max step-context length (j = 3 .. 0)
ROLE_ALPHA = 0.1        # add-alpha smoothing for the role distributions
STEP_ALPHA = 0.05       # add-alpha smoothing for the step distributions
ROLE_BETA = 2.0         # interpolation confidence pivot for the role model
STEP_BETA = 1.0         # interpolation confidence pivot for the step model

_FAMILY_AGNOSTIC = "*"  # pseudo-family key for pooled (cross-family) counts

__all__ = ["PPM", "ROLE_ORDER", "STEP_ORDER"]


def _ctx(tokens: Sequence[str], i: int, order: int) -> tuple:
    """The `order`-length context ending just before position `i`."""
    lo = max(0, i - order)
    return tuple(tokens[lo:i])


class PPM:
    """Role-factored variable-order Markov ranker.

    Counts are keyed by family (with a pooled `*` family for cross-family
    backoff). All public methods accept a `family` string; an unknown family
    transparently uses the pooled counts, which is exactly the desired OOD path.
    """

    def __init__(self) -> None:
        # ROLE model: role_ctx_counts[fam][order][ctx_tuple] -> Counter(role->n)
        self._role_counts: Dict[str, Dict[int, Dict[tuple, Counter]]] = {}
        # STEP model: step_ctx_counts[fam][role][order][ctx_tuple] -> Counter(step->n)
        self._step_counts: Dict[str, Dict[str, Dict[int, Dict[tuple, Counter]]]] = {}
        # Inventory of steps observed for each role (for uniform final backoff).
        self._role_steps: Dict[str, set] = defaultdict(set)
        # Full training step vocabulary (used when predict() gets an empty candset).
        self._vocab: List[str] = []
        # Per-token negative log-prob accumulators are not stored; perplexity is
        # computed on demand from the fitted distributions.
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, seqs_by_family: Mapping[str, Iterable[Sequence[str]]]) -> "PPM":
        """Build the role and step models by counting on the training families.

        Parameters
        ----------
        seqs_by_family : mapping family -> iterable of step sequences.
        """
        self.__init__()
        vocab: set = set()

        for family, seqs in seqs_by_family.items():
            fam = family.lower()
            self._role_counts.setdefault(fam, {o: defaultdict(Counter)
                                               for o in range(ROLE_ORDER + 1)})
            self._role_counts.setdefault(_FAMILY_AGNOSTIC,
                                         {o: defaultdict(Counter)
                                          for o in range(ROLE_ORDER + 1)})
            self._step_counts.setdefault(fam, defaultdict(
                lambda: {o: defaultdict(Counter) for o in range(STEP_ORDER + 1)}))
            self._step_counts.setdefault(_FAMILY_AGNOSTIC, defaultdict(
                lambda: {o: defaultdict(Counter) for o in range(STEP_ORDER + 1)}))

            for seq in seqs:
                steps = list(seq)
                roles = [induce_role(s) for s in steps]
                for s, r in zip(steps, roles):
                    vocab.add(s)
                    self._role_steps[r].add(s)

                for i in range(len(steps)):
                    step_t, role_t = steps[i], roles[i]
                    # ---- role model: contexts of length 0..ROLE_ORDER ----
                    for o in range(ROLE_ORDER + 1):
                        ctx = _ctx(roles, i, o)
                        self._role_counts[fam][o][ctx][role_t] += 1
                        self._role_counts[_FAMILY_AGNOSTIC][o][ctx][role_t] += 1
                    # ---- step model: conditioned on role_t, contexts 0..STEP_ORDER
                    for o in range(STEP_ORDER + 1):
                        ctx = _ctx(steps, i, o)
                        self._step_counts[fam][role_t][o][ctx][step_t] += 1
                        self._step_counts[_FAMILY_AGNOSTIC][role_t][o][ctx][step_t] += 1

        self._vocab = sorted(vocab)
        self._fitted = True
        return self

    # ----------------------------------------------------------- role model
    def _role_dist(self, role_ctx: Sequence[str], family: str) -> Dict[str, float]:
        """Interpolated P(role | role-context, family) over all ROLES."""
        fam = family.lower()
        if fam not in self._role_counts:
            fam = _FAMILY_AGNOSTIC
        per_fam = self._role_counts[fam]
        n_roles = len(ROLES)

        # order 0 (family role-unigram) with add-alpha is the interpolation anchor.
        dist = {r: 1.0 / n_roles for r in ROLES}  # uniform fallback if utterly empty
        ctx0 = ()
        c0 = per_fam[0].get(ctx0)
        if c0:
            tot0 = sum(c0.values()) + ROLE_ALPHA * n_roles
            dist = {r: (c0.get(r, 0) + ROLE_ALPHA) / tot0 for r in ROLES}

        # interpolate up through longer contexts (orders 1..ROLE_ORDER)
        max_o = min(ROLE_ORDER, len(role_ctx))
        for o in range(1, max_o + 1):
            ctx = tuple(role_ctx[len(role_ctx) - o:])
            counts = per_fam[o].get(ctx)
            if not counts:
                continue
            ctx_total = sum(counts.values())
            lam = ctx_total / (ctx_total + ROLE_BETA)
            tot = ctx_total + ROLE_ALPHA * n_roles
            higher = {r: (counts.get(r, 0) + ROLE_ALPHA) / tot for r in ROLES}
            dist = {r: lam * higher[r] + (1.0 - lam) * dist[r] for r in ROLES}
        return dist

    # ----------------------------------------------------------- step model
    def _step_dist(self, role: str, step_ctx: Sequence[str],
                   family: str) -> Dict[str, float]:
        """Interpolated P(step | role, step-context, family) over the role's steps.

        Returns a dict over the steps known for `role`; never empty unless the
        role was never observed in training (then an empty dict is returned and
        the caller floors it).
        """
        fam = family.lower()
        steps_for_role = self._role_steps.get(role)
        if not steps_for_role:
            return {}
        steps_for_role = sorted(steps_for_role)
        n = len(steps_for_role)

        def order0(fkey: str) -> Optional[Dict[str, float]]:
            fam_counts = self._step_counts.get(fkey)
            if not fam_counts or role not in fam_counts:
                return None
            c0 = fam_counts[role][0].get(())
            if not c0:
                return None
            tot = sum(c0.values()) + STEP_ALPHA * n
            return {s: (c0.get(s, 0) + STEP_ALPHA) / tot for s in steps_for_role}

        # order-0 anchor: family-specific P(step|role,fam); backoff to pooled;
        # finally uniform over the role's steps.
        dist = order0(fam) or order0(_FAMILY_AGNOSTIC) or {s: 1.0 / n for s in steps_for_role}

        # interpolate longer step contexts using the family that actually has counts.
        ctx_fam = fam if fam in self._step_counts and role in self._step_counts[fam] else _FAMILY_AGNOSTIC
        fam_counts = self._step_counts.get(ctx_fam)
        if fam_counts and role in fam_counts:
            role_counts = fam_counts[role]
            max_o = min(STEP_ORDER, len(step_ctx))
            for o in range(1, max_o + 1):
                ctx = tuple(step_ctx[len(step_ctx) - o:])
                counts = role_counts[o].get(ctx)
                if not counts:
                    continue
                ctx_total = sum(counts.values())
                lam = ctx_total / (ctx_total + STEP_BETA)
                tot = ctx_total + STEP_ALPHA * n
                higher = {s: (counts.get(s, 0) + STEP_ALPHA) / tot for s in steps_for_role}
                dist = {s: lam * higher[s] + (1.0 - lam) * dist[s] for s in steps_for_role}
        return dist

    # -------------------------------------------------------------- predict
    def predict(self, prefix: List[str], family: str, candset) -> Dict[str, float]:
        """Probabilities over `candset` (full training vocab if candset falsy).

        score(c) = P(role(c) | role-ctx, fam) * P(c | role(c), step-ctx, fam),
        renormalized over the candidate set. Sums to ~1.
        """
        if not self._fitted:
            raise RuntimeError("PPM.predict called before fit().")
        cands = list(candset) if candset else list(self._vocab)
        if not cands:
            return {}

        role_ctx = [induce_role(s) for s in prefix]
        role_dist = self._role_dist(role_ctx, family)

        # cache per-role step distributions (several candidates share a role)
        step_dist_cache: Dict[str, Dict[str, float]] = {}
        raw: Dict[str, float] = {}
        for c in cands:
            r = induce_role(c)
            if r not in step_dist_cache:
                step_dist_cache[r] = self._step_dist(r, prefix, family)
            sd = step_dist_cache[r]
            p_role = role_dist.get(r, 0.0)
            # If the candidate's step string is novel for this role, give it the
            # add-alpha floor implied by the role's step distribution.
            if sd:
                p_step = sd.get(c)
                if p_step is None:
                    p_step = min(sd.values()) if sd else 0.0
            else:
                p_step = 1.0  # role has no known steps: lean entirely on the role factor
            raw[c] = p_role * p_step

        total = sum(raw.values())
        if total <= 0.0:
            # Degenerate (e.g. all-zero role mass): fall back to uniform.
            u = 1.0 / len(cands)
            return {c: u for c in cands}
        return {c: v / total for c, v in raw.items()}

    # -------------------------------------------------------- predict_roles
    def predict_roles(self, prefix: List[str], family: str, top_r: int = 3) -> List[str]:
        """Top-`top_r` most likely next roles under the role model."""
        if not self._fitted:
            raise RuntimeError("PPM.predict_roles called before fit().")
        role_ctx = [induce_role(s) for s in prefix]
        dist = self._role_dist(role_ctx, family)
        ranked = sorted(dist, key=dist.get, reverse=True)
        return ranked[:max(1, top_r)]

    # ----------------------------------------------------------- perplexity
    def perplexity(self, seq: Sequence[str]) -> float:
        """Token-level perplexity of `seq` under the step model.

        Uses the family-agnostic step distributions (perplexity is a structural,
        family-independent signal used by the anomaly residual). Each token's
        probability is P(step_t | role_t, step-context); novel steps receive the
        add-alpha floor of their role's distribution.
        """
        if not self._fitted:
            raise RuntimeError("PPM.perplexity called before fit().")
        steps = list(seq)
        if not steps:
            return float("inf")
        log_sum = 0.0
        n = 0
        for i, step_t in enumerate(steps):
            role_t = induce_role(step_t)
            ctx = steps[:i]
            sd = self._step_dist(role_t, ctx, _FAMILY_AGNOSTIC)
            if sd:
                p = sd.get(step_t)
                if p is None or p <= 0.0:
                    p = min(sd.values())
            else:
                p = 1e-9  # role never seen: structurally very surprising
            p = max(p, 1e-12)
            log_sum += -math.log(p)
            n += 1
        return math.exp(log_sum / n)

    # ----------------------------------------------------------- save / load
    def save(self, path: str) -> None:
        """Serialize the fitted model to JSON."""
        def dump_role(rc):
            return {fam: {str(o): {"|".join(ctx): dict(cnt)
                                   for ctx, cnt in per_o.items()}
                          for o, per_o in orders.items()}
                    for fam, orders in rc.items()}

        def dump_step(sc):
            return {fam: {role: {str(o): {"|".join(ctx): dict(cnt)
                                          for ctx, cnt in per_o.items()}
                                 for o, per_o in orders.items()}
                          for role, orders in roles.items()}
                    for fam, roles in sc.items()}

        payload = {
            "version": 1,
            "vocab": self._vocab,
            "role_steps": {r: sorted(s) for r, s in self._role_steps.items()},
            "role_counts": dump_role(self._role_counts),
            "step_counts": dump_step(self._step_counts),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "PPM":
        """Load a model previously written by `save`."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        m = cls()
        m._vocab = list(payload["vocab"])
        m._role_steps = defaultdict(set, {r: set(s) for r, s in payload["role_steps"].items()})

        def load_ctx(s: str) -> tuple:
            return tuple(s.split("|")) if s else ()

        rc: Dict[str, Dict[int, Dict[tuple, Counter]]] = {}
        for fam, orders in payload["role_counts"].items():
            rc[fam] = {o: defaultdict(Counter) for o in range(ROLE_ORDER + 1)}
            for o_str, per_o in orders.items():
                o = int(o_str)
                for ctx_str, cnt in per_o.items():
                    rc[fam][o][load_ctx(ctx_str)] = Counter(cnt)
        m._role_counts = rc

        sc: Dict[str, Dict[str, Dict[int, Dict[tuple, Counter]]]] = {}
        for fam, roles in payload["step_counts"].items():
            sc[fam] = {}
            for role, orders in roles.items():
                sc[fam][role] = {o: defaultdict(Counter) for o in range(STEP_ORDER + 1)}
                for o_str, per_o in orders.items():
                    o = int(o_str)
                    for ctx_str, cnt in per_o.items():
                        sc[fam][role][o][load_ctx(ctx_str)] = Counter(cnt)
        m._step_counts = sc
        m._fitted = True
        return m


# ---------------------------------------------------------------------------
# Self-test: fit on a 150-seq subsample of mosfet+igbt, predict on a held-out
# `ic` prefix restricted to a grammar-legal candset, print top-5 and roles.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from nspe.data import load_family, candidate_vocab
    from nspe.grammar import valid_next_set

    print("Fitting PPM on 150-seq subsamples of mosfet + igbt ...")
    train = {
        "mosfet": [list(s) for s in load_family("mosfet")[:150]],
        "igbt": [list(s) for s in load_family("igbt")[:150]],
    }
    ppm = PPM().fit(train)

    # Held-out family `ic` prefix.
    ic_seq = list(load_family("ic")[0])
    cut = 12
    prefix, gold = ic_seq[:cut], ic_seq[cut]
    cand_vocab = candidate_vocab(("mosfet", "igbt"))  # train-family vocab (OOD test)
    candset = valid_next_set(prefix, cand_vocab)
    print(f"prefix len={cut}  gold next='{gold}'  |candset|={len(candset)}")

    probs = ppm.predict(prefix, "ic", candset)
    top5 = sorted(probs, key=probs.get, reverse=True)[:5]
    print("Top-5 (step, prob):")
    for s in top5:
        print(f"  {probs[s]:.4f}  {s}")

    total = sum(probs.values())
    print(f"sum of probs over candset = {total:.6f}")
    assert abs(total - 1.0) < 1e-6, f"probs must sum to ~1.0, got {total}"

    roles = ppm.predict_roles(prefix, "ic", top_r=3)
    print("predict_roles (top-3):", roles)

    # Sanity: predict with empty candset uses full vocab and still sums to ~1.
    full = ppm.predict(prefix, "ic", set())
    assert abs(sum(full.values()) - 1.0) < 1e-6
    print(f"empty-candset predict over full vocab ({len(full)} steps) sums to {sum(full.values()):.6f}")

    # Sanity: perplexity is a finite positive number.
    ppl = ppm.perplexity(ic_seq)
    print(f"perplexity(full ic seq) = {ppl:.2f}")
    assert ppl > 0 and math.isfinite(ppl)

    # Sanity: save/load round-trip preserves predictions.
    import os, tempfile
    tmp = os.path.join(tempfile.gettempdir(), "ppm_selftest.json")
    ppm.save(tmp)
    ppm2 = PPM.load(tmp)
    probs2 = ppm2.predict(prefix, "ic", candset)
    assert all(abs(probs[s] - probs2[s]) < 1e-9 for s in probs), "save/load mismatch"
    print("save/load round-trip OK")
    print("SELF-TEST PASSED")
