#!/usr/bin/env python3
"""
ood_symbolic_probe.py — Does the symbolic rule-checker generalize to an unseen
4th product family? And where does it break?

This measures the decisive question for a symbolic/neurosymbolic Task-3 (anomaly)
strategy: the rules are family-agnostic *in principle*, but the validator's
triggers are matched by exact STEP STRING (frozensets like DEPOSITION_STEPS).
So generalization to a hidden family hinges on vocabulary coverage.

Four experiments, all on a simulated unseen family (diode / schottky / sic_mosfet
generators that share the backbone but are NOT MOSFET/IGBT/IC):

  [1] False-positive rate of the stock checker on OOD *valid* sequences.
  [2] Recall when we inject a violation using a KNOWN deposition string.
  [3] Recall when we inject the SAME violation using a NOVEL (unseen) string.
  [4] Recall on [3] after adding a ~10-line role-induction shim.

Run from repo root:
  python neurosymbolic-approach/experiments/ood_symbolic_probe.py
"""

import importlib.util
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the official validator and the OOD family generators by file path.
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
TRACK = REPO / "tracks" / "industrial-infineon"
GEN_PATH = TRACK / "training_data" / "generate_sequences.py"
OOD_PATH = TRACK / "scripts" / "generate_ood_families.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gs = _load("generate_sequences", GEN_PATH)
ood = _load("generate_ood_families", OOD_PATH)
validate_sequence = gs.validate_sequence


# ---------------------------------------------------------------------------
# Role-induction shim: map UNKNOWN step strings onto rule roles by surface
# pattern, then run the *exact same* rule logic with augmented trigger sets.
# (This is the cheap neurosymbolic fix for the validator's OOD blind spot.)
# ---------------------------------------------------------------------------
def validate_with_role_induction(steps):
    vocab = set(steps)
    dep_extra = {s for s in vocab
                 if s.startswith("DEPOSIT ") or "OXIDATION" in s
                 or s.endswith(" GROWTH") or "EPITAXIAL DEPOSITION" in s}
    clean_extra = {s for s in vocab
                   if "CLEAN" in s or s.endswith(" RINSE")
                   or s.startswith("DRY ") or s == "HF DIP"}
    etch_extra = {s for s in vocab if " ETCH" in s or s.startswith("ETCH ")}
    implant_extra = {s for s in vocab if s.startswith("IMPLANT ")}

    orig = (gs.DEPOSITION_STEPS, gs.CLEAN_STEPS, gs.ETCH_STEPS, gs.IMPLANT_STEPS)
    try:
        gs.DEPOSITION_STEPS = orig[0] | dep_extra
        gs.CLEAN_STEPS = orig[1] | clean_extra
        gs.ETCH_STEPS = orig[2] | etch_extra
        gs.IMPLANT_STEPS = orig[3] | implant_extra
        return validate_sequence(steps)
    finally:
        (gs.DEPOSITION_STEPS, gs.CLEAN_STEPS,
         gs.ETCH_STEPS, gs.IMPLANT_STEPS) = orig


# ---------------------------------------------------------------------------
# Surgical, confound-free violation injection: insert one deposition step right
# before SHIP LOT, after clearing any clean from its 12-step lookback window.
# With a KNOWN name -> stock checker must fire RULE_DEP_NO_CLEAN.
# With a NOVEL name -> stock checker is blind; role-induction recovers it.
# ---------------------------------------------------------------------------
KNOWN_DEP = "DEPOSIT POLYSILICON"          # in DEPOSITION_STEPS
NOVEL_DEP = "DEPOSIT SIC EPITAXIAL STACK"  # NOT in any frozenset (unseen family)
FILLER = "MEASURE THICKNESS"               # benign: not a clean, not a trigger


def _is_clean_like(s):
    return (s in gs.CLEAN_STEPS or "CLEAN" in s or s.endswith(" RINSE")
            or s.startswith("DRY ") or s == "HF DIP")


def inject_dep_no_clean(steps, dep_step):
    steps = list(steps)
    j = steps.index("SHIP LOT")
    # Guarantee a clean-free lookback window under both stock & augmented defs.
    for k in range(max(0, j - 12), j):
        if _is_clean_like(steps[k]):
            steps[k] = FILLER
    steps.insert(j, dep_step)   # deposition now has no clean in prior 12
    return steps


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def detected(violations, rule=None):
    if not violations:
        return False
    return True if rule is None else any(v.rule == rule for v in violations)


def main():
    N = 300
    families = ["diode", "schottky", "sic_mosfet"]
    valids = []
    for i, fam in enumerate(families):
        seqs = ood.generate_unique(fam, N, seed=777 + 1000 * i)
        valids.extend(seqs)
    print(f"Simulated unseen family pool: {len(valids)} valid sequences "
          f"({', '.join(f.upper() for f in families)})\n")

    # [1] False positives on OOD valids (stock checker)
    fp = sum(1 for s in valids if validate_sequence(s))
    print(f"[1] Stock checker false-positives on OOD valids : "
          f"{fp}/{len(valids)}  (FP rate {fp/len(valids):.3f})")

    # [2] KNOWN-vocab injection, stock checker
    known = [inject_dep_no_clean(s, KNOWN_DEP) for s in valids]
    d2 = sum(detected(validate_sequence(s), "RULE_DEP_NO_CLEAN") for s in known)
    print(f"[2] Recall, KNOWN-vocab violation, stock checker : "
          f"{d2}/{len(known)}  ({d2/len(known):.3f})  <- reused-vocab case")

    # [3] NOVEL-vocab injection, stock checker (the blind spot)
    novel = [inject_dep_no_clean(s, NOVEL_DEP) for s in valids]
    d3 = sum(detected(validate_sequence(s)) for s in novel)
    print(f"[3] Recall, NOVEL-vocab violation, stock checker : "
          f"{d3}/{len(novel)}  ({d3/len(novel):.3f})  <- BLIND SPOT")

    # [4] NOVEL-vocab injection, role-induction checker (the fix)
    d4 = sum(detected(validate_with_role_induction(s), "RULE_DEP_NO_CLEAN")
             for s in novel)
    print(f"[4] Recall, NOVEL-vocab violation, +role-induction: "
          f"{d4}/{len(novel)}  ({d4/len(novel):.3f})  <- RECOVERED")

    # Sanity: role-induction must not invent FPs on the OOD valids.
    fp_ri = sum(1 for s in valids if validate_with_role_induction(s))
    print(f"\n    (sanity) role-induction FP on OOD valids    : "
          f"{fp_ri}/{len(valids)}  (FP rate {fp_ri/len(valids):.3f})")


if __name__ == "__main__":
    main()
