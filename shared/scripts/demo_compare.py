"""Side-by-side demo: baseline vs trained-model predictions on identical input.

REQUIRED deliverable per `submission/REPORT_TEMPLATE.md` Industrial AI track:
"Demo shows baseline vs. trained output on identical inputs".

Loads four predictors:

    1. Trigram-with-backoff  (no params, EDA baseline)
    2. Grammar-trigram       (trigram + validator-mask filter)
    3. Transformer (LM only) (our v2 lm_only checkpoint)
    4. Multi-task transformer (our v2 multitask checkpoint — validity + rule-ID heads)

Then runs each on the same prefix and prints next-step Top-5 + a 10-step
greedy completion. Also detects anomalies if asked.

Usage examples
--------------
# Use one of the bundled example prefixes (early, mid, late position)
.venv/bin/python shared/scripts/demo_compare.py --example mosfet-mid

# Provide a custom prefix via pipe-separated steps
.venv/bin/python shared/scripts/demo_compare.py \\
    --family mosfet \\
    --prefix "RECEIVE WAFER LOT|LOT IDENTIFICATION|INITIAL WAFER INSPECTION"

# Test anomaly detection on a corrupted sequence
.venv/bin/python shared/scripts/demo_compare.py --example anomaly-mosfet

# Point at a specific checkpoint (default uses the best v2 final)
.venv/bin/python shared/scripts/demo_compare.py --example mosfet-mid \\
    --lm-checkpoint shared/extras/checkpoints/v2-final-transformer-small-lm_only-all3/final.pt \\
    --mt-checkpoint shared/extras/checkpoints/v2-final-transformer-small-multitask-all3/final.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Repo-rooted imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "models"))

from transformer_xlstm.data.load import load_all_families
from transformer_xlstm.data.validator import validate_sequence

# Optional torch-dependent imports — only loaded if a transformer checkpoint
# is requested. Lets the trigram-only demo run on a machine without torch.

# --------------------------------------------------------------------------- #
# Trigram + grammar predictors                                                #
# --------------------------------------------------------------------------- #

_TRIGRAM_PATH = REPO_ROOT / "shared" / "extras" / "baselines" / "trigram_baseline.py"
sys.path.insert(0, str(_TRIGRAM_PATH.parent))
from trigram_baseline import TrigramBackoff  # type: ignore  # noqa: E402


def _build_trigram() -> TrigramBackoff:
    examples = load_all_families()
    model = TrigramBackoff()
    model.fit([ex.steps for ex in examples])
    return model


def trigram_topk(model: TrigramBackoff, prefix: list[str], k: int = 5) -> list[str]:
    ctx2 = prefix[-2] if len(prefix) >= 2 else None
    ctx1 = prefix[-1] if len(prefix) >= 1 else None
    return model.rank(ctx2, ctx1, k=k)


def grammar_trigram_topk(model: TrigramBackoff, prefix: list[str], k: int = 5,
                          k_pool: int = 30) -> list[str]:
    """Trigram + validator-mask: reject candidates that introduce a violation."""
    ctx2 = prefix[-2] if len(prefix) >= 2 else None
    ctx1 = prefix[-1] if len(prefix) >= 1 else None
    pool = model.rank(ctx2, ctx1, k=k_pool)
    out: list[str] = []
    for s in pool:
        new_prefix = prefix + [s]
        if any(v.step_index == len(prefix) for v in validate_sequence(new_prefix)):
            continue
        out.append(s)
        if len(out) >= k:
            break
    if not out:
        return pool[:k]
    return out


def trigram_complete(model: TrigramBackoff, prefix: list[str], n: int = 10) -> list[str]:
    return model.complete(list(prefix), max_len=n)


# --------------------------------------------------------------------------- #
# Transformer predictor (lazy-loaded)                                          #
# --------------------------------------------------------------------------- #

def _load_transformer(checkpoint: Path):
    from transformer_xlstm.eval.predict import load_model
    return load_model(checkpoint)


def transformer_topk(lm, family: str, prefix: list[str], k: int = 5,
                      grammar: bool = True) -> list[str]:
    from transformer_xlstm.eval.predict import topk_next_step
    return topk_next_step(lm, family, prefix, k=k, grammar=grammar)


def transformer_complete(lm, family: str, prefix: list[str], n: int = 10) -> list[str]:
    from transformer_xlstm.eval.predict import complete_sequence
    return complete_sequence(lm, family, list(prefix), max_len=n)


def transformer_anomaly(lm, family: str, full_seq: list[str]) -> dict:
    from transformer_xlstm.eval.predict import anomaly_ensemble
    return anomaly_ensemble(lm, family, full_seq)


# --------------------------------------------------------------------------- #
# Example prefixes                                                             #
# --------------------------------------------------------------------------- #

EXAMPLES = {
    "mosfet-early": ("mosfet",
                     ["RECEIVE WAFER LOT", "LOT IDENTIFICATION",
                      "INITIAL WAFER INSPECTION", "MEASURE THICKNESS",
                      "MEASURE SURFACE PARTICLES", "PRE CLEAN WAFER"]),
    "mosfet-mid":   ("mosfet",
                     ["RECEIVE WAFER LOT", "LOT IDENTIFICATION",
                      "PRE CLEAN INSPECTION", "MEASURE THICKNESS",
                      "PRE CLEAN WAFER", "WET CLEAN RCA1", "RCA CLEAN 2",
                      "HF DIP", "SUBSTRATE CHECK", "EPITAXY PREP",
                      "EPITAXIAL DEPOSITION", "MEASURE EPITAXY THICKNESS",
                      "EPITAXY ANNEAL", "WAFER SURFACE CLEAN",
                      "THERMAL OXIDATION", "MEASURE OXIDE THICKNESS",
                      "SPIN COAT PHOTORESIST", "SOFT BAKE",
                      "ALIGN MASK LEVEL 1", "EXPOSE LITHO LEVEL 1",
                      "DEVELOP PHOTORESIST", "PATTERN INSPECTION LEVEL 1",
                      "OXIDE ETCH", "STRIP PHOTORESIST", "CLEAN AFTER ETCH",
                      "IMPLANT WELL", "DRIVE IN DIFFUSION",
                      "RAPID THERMAL ANNEAL"]),
    "igbt-mid":     ("igbt",
                     ["RECEIVE WAFER LOT", "LOT IDENTIFICATION",
                      "INITIAL WAFER INSPECTION", "MEASURE THICKNESS",
                      "PRE CLEAN WAFER", "WET CLEAN RCA1", "RCA CLEAN 2",
                      "HF DIP", "EPITAXIAL WAFER CHECK",
                      "MEASURE EPITAXY THICKNESS", "MEASURE RESISTIVITY",
                      "EPITAXIAL LAYER PREP", "THERMAL OXIDATION",
                      "MEASURE OXIDE THICKNESS"]),
    "ic-early":     ("ic",
                     ["RECEIVE WAFER LOT", "LOT IDENTIFICATION",
                      "PRE CLEAN INSPECTION", "MEASURE THICKNESS",
                      "MEASURE SURFACE PARTICLES", "PRE CLEAN WAFER",
                      "BACKSIDE CLEAN", "RCA CLEAN 1", "RCA CLEAN 2",
                      "HF DIP", "WAFER CLEAN PRE-GRIND"]),
    # Intentionally violates RULE_TEST_BEFORE_PASSIVATION (PARAMETRIC TEST
    # before CURE PASSIVATION).
    "anomaly-mosfet": ("mosfet",
                       ["RECEIVE WAFER LOT", "LOT IDENTIFICATION",
                        "INITIAL WAFER INSPECTION", "PRE CLEAN WAFER",
                        "RCA CLEAN 1", "RCA CLEAN 2", "HF DIP",
                        "THERMAL OXIDATION", "PARAMETRIC TEST",
                        "DEPOSIT PASSIVATION", "CURE PASSIVATION",
                        "SHIP LOT"]),
}


# --------------------------------------------------------------------------- #
# Printing                                                                     #
# --------------------------------------------------------------------------- #

def _fmt_topk(label: str, ranked: list[str], gold: str | None = None) -> str:
    star = lambda s: " ✓" if (gold is not None and s == gold) else ""
    body = "  |  ".join(f"{i+1}. {s}{star(s)}" for i, s in enumerate(ranked))
    return f"  {label:<28} {body}"


def _print_section(title: str) -> None:
    print()
    print(title)
    print("─" * len(title))


def run_comparison(family: str, prefix: list[str],
                    lm_checkpoint: Path | None,
                    mt_checkpoint: Path | None,
                    n_completion: int = 10) -> None:
    print(f"Family    : {family.upper()}")
    print(f"Prefix    : {len(prefix)} steps, ending …")
    for s in prefix[-3:]:
        print(f"            → {s}")
    print(f"Next step ground truth: (unknown, this is the model's job)")

    trigram = _build_trigram()

    _print_section("TASK 1 — Next-step Top-5 predictions")
    print(_fmt_topk("trigram (no params)", trigram_topk(trigram, prefix, k=5)))
    print(_fmt_topk("grammar-trigram", grammar_trigram_topk(trigram, prefix, k=5)))
    if lm_checkpoint and lm_checkpoint.exists():
        lm = _load_transformer(lm_checkpoint)
        print(_fmt_topk("transformer (LM only)",
                        transformer_topk(lm, family, prefix, k=5, grammar=True)))
    else:
        print("  transformer (LM only)         (checkpoint not found — skipped)")
    if mt_checkpoint and mt_checkpoint.exists():
        mt = _load_transformer(mt_checkpoint)
        print(_fmt_topk("multitask transformer",
                        transformer_topk(mt, family, prefix, k=5, grammar=True)))
    else:
        print("  multitask transformer         (checkpoint not found — skipped)")

    _print_section(f"TASK 2 — Greedy completion (next {n_completion} steps)")
    tri_comp = trigram_complete(trigram, prefix, n=n_completion)
    print(f"  trigram        → {' → '.join(tri_comp) or '(empty)'}")
    if lm_checkpoint and lm_checkpoint.exists():
        lm_comp = transformer_complete(lm, family, prefix, n=n_completion)
        print(f"  transformer    → {' → '.join(lm_comp) or '(empty)'}")
    if mt_checkpoint and mt_checkpoint.exists():
        mt_comp = transformer_complete(mt, family, prefix, n=n_completion)
        print(f"  multitask      → {' → '.join(mt_comp) or '(empty)'}")

    # If the prefix already ends with SHIP LOT (i.e. it's a full sequence),
    # run anomaly detection too.
    if "SHIP LOT" in prefix:
        _print_section("TASK 3 — Anomaly detection on the full sequence")
        viols = validate_sequence(prefix)
        if viols:
            for v in viols:
                print(f"  validator       → INVALID  rule={v.rule}  at step {v.step_index}")
        else:
            print(f"  validator       → VALID  (no rule violations)")
        if mt_checkpoint and mt_checkpoint.exists():
            mt = _load_transformer(mt_checkpoint)
            result = transformer_anomaly(mt, family, prefix)
            label = "VALID" if result["IS_VALID"] == 1 else "INVALID"
            rule = f"  rule={result['PREDICTED_RULE']}" if result["PREDICTED_RULE"] else ""
            print(f"  multitask ens.  → {label}  P_valid={result['SCORE']:.3f}{rule}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _default_checkpoint(suffix: str) -> Path | None:
    """Find the best available checkpoint matching a recipe suffix."""
    candidates = [
        REPO_ROOT / "shared" / "extras" / "checkpoints" / f"v3-final-transformer-medium-{suffix}-ood25-all3",
        REPO_ROOT / "shared" / "extras" / "checkpoints" / f"v2-final-transformer-medium-{suffix}-all3",
        REPO_ROOT / "shared" / "extras" / "checkpoints" / f"v2-final-transformer-small-{suffix}-all3",
        REPO_ROOT / "shared" / "extras" / "checkpoints" / f"final-transformer-medium-{suffix}-fdp00-all3",
        REPO_ROOT / "shared" / "extras" / "checkpoints" / f"final-transformer-small-{suffix}-fdp00-all3",
    ]
    for c in candidates:
        if (c / "final.pt").exists():
            return c / "final.pt"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--example", choices=sorted(EXAMPLES.keys()),
                   help="Use one of the built-in example prefixes.")
    g.add_argument("--prefix", help="Pipe-separated step strings.")
    parser.add_argument("--family", default=None,
                        help="Required if --prefix is given. One of: mosfet, igbt, ic.")
    parser.add_argument("--n-completion", type=int, default=10)
    parser.add_argument("--lm-checkpoint", default=None,
                        help="Path to LM-only transformer checkpoint.")
    parser.add_argument("--mt-checkpoint", default=None,
                        help="Path to multitask transformer checkpoint.")
    args = parser.parse_args()

    if args.example:
        family, prefix = EXAMPLES[args.example]
    else:
        if not args.family:
            parser.error("--family is required when --prefix is provided")
        family, prefix = args.family.lower(), args.prefix.split("|")

    lm_ck = Path(args.lm_checkpoint) if args.lm_checkpoint else _default_checkpoint("lm_only")
    mt_ck = Path(args.mt_checkpoint) if args.mt_checkpoint else _default_checkpoint("multitask")

    print("=" * 72)
    print("BASELINE vs TRAINED — side-by-side process-step prediction")
    print("=" * 72)
    if lm_ck:    print(f"  LM-only checkpoint  : {lm_ck.parent.name}")
    if mt_ck:    print(f"  Multitask checkpoint: {mt_ck.parent.name}")

    run_comparison(family, prefix, lm_ck, mt_ck, n_completion=args.n_completion)


if __name__ == "__main__":
    main()
