"""make_charts.py — render every relevant NSPE metric to PNG charts.

This is the *reporting* stage of the symbolic-first pipeline. It reads whatever
experiment JSONs exist in ``$NSPE_OUT`` (the same dir the ``expNN`` scripts write
to) and renders one PNG per metric family to ``$NSPE_OUT/charts/``. It is
deliberately tolerant: a missing or smoke-only result simply skips its panel and
prints a one-line note, so the script is safe to run at any point in the
experiment campaign (CPU symbolic results only, partial Leonardo runs, or the
full battery).

Panels (rendered only when the backing data is present):

  1. ``anomaly_id_per_rule.png``     — Task-3 ID per-rule recall + rule-attribution
                                       (all 10 rules)                  [exp01.json]
  2. ``anomaly_ood_recovery.png``    — Task-3 OOD novel-vocab per-rule recall,
                                       roles_off vs roles_on (the role-induction
                                       headline; annotates 0 false-positives)
                                                                       [exp01.json]
  3. ``ppm_lofo_nextstep.png``       — Task-1 PPM LoFO Top-1/Top-3/Top-5/MRR,
                                       ID vs OOD, per held-out family + mean
                                                                       [exp02.json]
  4. ``ppm_lofo_completion.png``     — Task-2 PPM LoFO NED/token_acc/block_acc,
                                       ID vs OOD, per family + mean    [exp02.json]
  5. ``ood_drop_comparison.png``     — the thesis chart: ID->OOD next-step Top-1
                                       drop, PPM (symbolic-first) vs the documented
                                       pure-neural LoFO trigram drop  [exp02.json]
  6a. ``exp03_neural_vs_ppm.png``    — neural-vs-PPM ID & OOD next-step
                                       (only if a NON-smoke exp03_*.json exists)
  6b. ``exp04_constraint_loss.png``  — invalid-emission rate / OOD Top-1 across the
                                       {mask, sem_w} ablation grid
                                       (only if a NON-smoke exp04.json exists)
  6c. ``exp05_scaling.png``          — params / #families vs ID & OOD Top-1
                                       (only if a NON-smoke exp05.json exists)

Run::

    PYTHONPATH=models/neurosymbolic NSPE_OUT=models/neurosymbolic/outputs \\
        python3 models/neurosymbolic/experiments/make_charts.py

No torch, no network. Needs matplotlib (+ seaborn styling) and numpy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / file-only rendering
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns  # type: ignore

    _HAVE_SEABORN = True
except Exception:  # pragma: no cover - seaborn is expected but optional
    sns = None  # type: ignore
    _HAVE_SEABORN = False

__all__ = ["make_all_charts"]

DPI = 150

# Canonical rule order shared by exp01 (matches rules.RULE_IDS).
RULE_ORDER = [
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
]

# The documented pure-neural reference: a LoFO trigram baseline next-step Top-1
# of 0.72 in-distribution that drops to ~0.48 on the held-out 4th family — a
# 0.24 absolute drop (see FINDINGS.md). This is the bar the symbolic-first PPM
# must beat by staying flat.
NEURAL_REF_ID_TOP1 = 0.72
NEURAL_REF_OOD_TOP1 = 0.48
NEURAL_REF_DROP = NEURAL_REF_ID_TOP1 - NEURAL_REF_OOD_TOP1  # 0.24


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
def _default_out_dir() -> Path:
    """The output dir the experiments write to (``$NSPE_OUT`` or repo default)."""
    return Path(
        os.environ.get(
            "NSPE_OUT", Path(__file__).resolve().parents[1] / "outputs"
        )
    )


def _apply_style() -> None:
    """Apply seaborn styling if available; otherwise a sane matplotlib default."""
    if _HAVE_SEABORN:
        sns.set_theme(style="whitegrid", context="talk")
    else:  # graceful fallback
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")
    plt.rcParams.update({"figure.autolayout": False, "savefig.dpi": DPI})


def _load_json(path: Path) -> Optional[dict]:
    """Read a JSON file, returning ``None`` (and a note) if missing/unreadable."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
        print(f"  ! could not read {path.name}: {exc}")
        return None


def _save(fig: plt.Figure, out_path: Path) -> None:
    """Save a figure at >= DPI and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _palette(n: int) -> List:
    """A categorical palette of length ``n`` (seaborn if available)."""
    if _HAVE_SEABORN:
        return list(sns.color_palette("colorblind", n))
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


# ---------------------------------------------------------------------------
# Panel 1 — Task-3 anomaly ID: per-rule recall + attribution
# ---------------------------------------------------------------------------
def _panel_anomaly_id_per_rule(exp01: dict, out_dir: Path) -> Optional[Path]:
    """Bar chart of per-rule ID recall and rule-attribution for all 10 rules."""
    per_rule = exp01.get("id", {}).get("per_rule")
    if not per_rule:
        print("  - skip anomaly_id_per_rule: exp01.json has no id.per_rule")
        return None

    rules = [r for r in RULE_ORDER if r in per_rule]
    rules += [r for r in per_rule if r not in rules]  # any extras, stable tail
    labels = [r.replace("RULE_", "") for r in rules]

    recall = []
    attr = []
    counts = []
    for r in rules:
        d = per_rule[r]
        n = max(int(d.get("n", 0)), 0)
        counts.append(n)
        recall.append(d.get("detected", 0) / n if n else 0.0)
        attr.append(d.get("attributed", 0) / n if n else 0.0)

    colors = _palette(2)
    x = np.arange(len(rules))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 6.5))
    b1 = ax.bar(x - width / 2, recall, width, label="recall (detected / n)",
                color=colors[0])
    b2 = ax.bar(x + width / 2, attr, width, label="rule-attribution (correct rule / n)",
                color=colors[1])

    overall = exp01.get("id", {}).get("metrics", {})
    n_valid = exp01.get("id", {}).get("n_valid")
    n_invalid = exp01.get("id", {}).get("n_invalid")
    subtitle = (
        f"overall recall={overall.get('recall', float('nan')):.3f}, "
        f"attribution={overall.get('rule_attr', float('nan')):.3f}, "
        f"FP rate={1.0 - overall.get('precision', 1.0):.3f}"
    )
    ax.set_title(
        "Task 3 — Symbolic anomaly ID: per-rule recall & attribution\n"
        + subtitle,
        fontsize=14,
    )
    ax.set_ylabel("fraction of injected violations")
    ax.set_xlabel("violation rule (n injected shown on bars)")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.axhline(0.98, color="grey", ls="--", lw=1, label="target recall 0.98")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    for xi, n in zip(x, counts):
        ax.text(xi, 1.04, f"n={n}", ha="center", va="bottom", fontsize=7,
                color="dimgray")
    if n_valid is not None and n_invalid is not None:
        ax.text(0.01, 0.02,
                f"set: {n_invalid} invalid / {n_valid} valid sequences",
                transform=ax.transAxes, fontsize=8, color="dimgray")

    out = out_dir / "anomaly_id_per_rule.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Panel 2 — Task-3 anomaly OOD: roles_off vs roles_on per-rule recall
# ---------------------------------------------------------------------------
def _panel_anomaly_ood_recovery(exp01: dict, out_dir: Path) -> Optional[Path]:
    """Grouped bars: per-rule OOD recall roles_off vs roles_on (novel vocab)."""
    ood = exp01.get("ood", {})
    per_rule = ood.get("per_rule")
    if not per_rule or "roles_off" not in per_rule or "roles_on" not in per_rule:
        print("  - skip anomaly_ood_recovery: exp01.json has no ood.per_rule"
              " roles_off/roles_on")
        return None

    off = per_rule["roles_off"]
    on = per_rule["roles_on"]
    rules = [r for r in RULE_ORDER if r in on or r in off]
    rules += [r for r in on if r not in rules]
    labels = [r.replace("RULE_", "") for r in rules]

    def _recall(table: dict, rule: str) -> float:
        # Attribution-correct recall: a row counts only if the oracle flags it
        # invalid AND names the injected rule. This is the honest per-rule story:
        # for some rules (e.g. ETCH_NO_MASK) the stock validator detects the
        # sequence but mis-attributes it, so binary recall would overstate it.
        d = table.get(rule, {})
        n = max(int(d.get("n", 0)), 0)
        return d.get("attributed", 0) / n if n else 0.0

    rec_off = [_recall(off, r) for r in rules]
    rec_on = [_recall(on, r) for r in rules]

    colors = _palette(3)
    x = np.arange(len(rules))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(x - width / 2, rec_off, width, label="roles OFF (literal frozensets)",
           color=colors[0])
    ax.bar(x + width / 2, rec_on, width,
           label="roles ON (role-induction anchors)", color=colors[2])

    modes = ood.get("modes", {})
    off_m = modes.get("roles_off", {})
    on_m = modes.get("roles_on", {})
    fams = ", ".join(ood.get("families", []))
    subtitle = (
        f"unseen families: {fams} | "
        f"overall recall {off_m.get('recall', float('nan')):.2f} -> "
        f"{on_m.get('recall', float('nan')):.2f} | "
        f"false-positives on valids: "
        f"{off_m.get('fp_on_valids', '?')} (off) / "
        f"{on_m.get('fp_on_valids', '?')} (on)"
    )
    ax.set_title(
        "Task 3 — OOD novel-vocab recovery via role induction\n" + subtitle,
        fontsize=14,
    )
    ax.set_ylabel("per-rule correct detection + attribution\n(unseen families)")
    ax.set_xlabel("novel-vocab-capable violation rule")
    ax.set_ylim(0, 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Headline annotation: zero false positives in both modes. Placed just inside
    # the top-right of the axes so it never collides with the figure title.
    fp_off = off_m.get("fp_on_valids")
    fp_on = on_m.get("fp_on_valids")
    if fp_off == 0 and fp_on == 0:
        ax.text(0.985, 0.965, "0 false-positives in BOTH modes",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=11, fontweight="bold",
                color="darkgreen",
                bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", ec="darkgreen"))
    # Most roles_off bars are ~0 (literal frozensets miss renamed triggers); call
    # it out so the near-empty series is read as a result, not a missing bar.
    if max(rec_off) < 0.05:
        ax.text(0.985, 0.84,
                "roles OFF ~ 0 recall on novel vocab",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=colors[0])

    out = out_dir / "anomaly_ood_recovery.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# exp02 LoFO helpers
# ---------------------------------------------------------------------------
def _lofo_rows(
    exp02: dict, task: str, metrics: Sequence[str]
) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    """Return (group labels incl. 'mean', per-metric ID/OOD value lists).

    ``out[metric] = {"id": [...per family..., mean], "ood": [...]}``.
    """
    lofo = exp02.get("lofo", {})
    splits = lofo.get("splits", [])
    groups = [s.get("held_out", f"split{i}") for i, s in enumerate(splits)]

    out: Dict[str, Dict[str, List[float]]] = {
        m: {"id": [], "ood": []} for m in metrics
    }
    for s in splits:
        for m in metrics:
            out[m]["id"].append(s.get("id", {}).get(task, {}).get(m, float("nan")))
            out[m]["ood"].append(s.get("ood", {}).get(task, {}).get(m, float("nan")))

    # Append the mean group (computed from splits, robust to NaN).
    groups = groups + ["mean"]
    for m in metrics:
        for side in ("id", "ood"):
            vals = np.array(out[m][side], dtype=float)
            out[m][side].append(float(np.nanmean(vals)) if vals.size else float("nan"))
    return groups, out


def _panel_ppm_lofo(
    exp02: dict, out_dir: Path, task: str, metrics: Sequence[str],
    fname: str, title: str, ylabel: str,
) -> Optional[Path]:
    """Grouped bars: per held-out family + mean, ID vs OOD, for each metric.

    Lays out one subplot per metric so very different scales (e.g. NED vs
    token_acc) stay readable; within each subplot the x axis is the held-out
    family (+ mean) and the two bars are ID vs OOD.
    """
    if not exp02.get("lofo", {}).get("splits"):
        print(f"  - skip {fname}: exp02.json has no lofo.splits")
        return None

    groups, data = _lofo_rows(exp02, task, metrics)
    n_metrics = len(metrics)
    colors = _palette(4)
    id_c, ood_c = colors[0], colors[3]

    fig, axes = plt.subplots(
        1, n_metrics, figsize=(3.8 * n_metrics, 5.2), sharey=False
    )
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(groups))
    width = 0.4
    for ax, m in zip(axes, metrics):
        idv = data[m]["id"]
        oodv = data[m]["ood"]
        b1 = ax.bar(x - width / 2, idv, width, label="ID (train families)",
                    color=id_c)
        b2 = ax.bar(x + width / 2, oodv, width, label="OOD (held-out family)",
                    color=ood_c)
        ax.set_title(m, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(1.0, np.nanmax([*idv, *oodv]) * 1.18 if (idv or oodv) else 1.0))
        for bars in (b1, b2):
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
        # Emphasise the 'mean' group.
        ax.axvspan(len(groups) - 1.5, len(groups) - 0.5, color="grey", alpha=0.06)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    n_per = exp02.get("lofo", {}).get("n_per_family")
    fig.suptitle(
        f"{title}  (PPM symbolic-first ranker; n={n_per}/family/split)",
        fontsize=14, y=1.08,
    )
    out = out_dir / fname
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Panel 5 — the thesis chart: ID->OOD Top-1 drop, PPM vs pure-neural reference
# ---------------------------------------------------------------------------
def _panel_ood_drop_comparison(exp02: dict, out_dir: Path) -> Optional[Path]:
    """The headline: PPM's near-flat ID->OOD Top-1 vs the pure-neural drop."""
    lofo = exp02.get("lofo", {})
    splits = lofo.get("splits", [])
    if not splits:
        print("  - skip ood_drop_comparison: exp02.json has no lofo.splits")
        return None

    id_vals = np.array(
        [s.get("id", {}).get("next-step", {}).get("top1", np.nan) for s in splits],
        dtype=float,
    )
    ood_vals = np.array(
        [s.get("ood", {}).get("next-step", {}).get("top1", np.nan) for s in splits],
        dtype=float,
    )
    ppm_id = float(np.nanmean(id_vals))
    ppm_ood = float(np.nanmean(ood_vals))
    # Prefer the experiment's own recorded mean drop when present.
    ppm_drop = lofo.get("mean_drop", {}).get("next-step", {}).get("top1")
    if ppm_drop is None:
        ppm_drop = ppm_id - ppm_ood

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    colors = _palette(4)
    ppm_c, neural_c = colors[0], colors[3]

    # ---- left: slope lines ID -> OOD for both systems ----
    xs = [0, 1]
    ax_l.plot(xs, [NEURAL_REF_ID_TOP1, NEURAL_REF_OOD_TOP1], "-o", color=neural_c,
              lw=3, ms=11, label=f"pure-neural baseline (LoFO trigram, ref)")
    ax_l.plot(xs, [ppm_id, ppm_ood], "-o", color=ppm_c, lw=3, ms=11,
              label="PPM (symbolic-first)")

    for x, y, t in ((0, NEURAL_REF_ID_TOP1, NEURAL_REF_ID_TOP1),
                    (1, NEURAL_REF_OOD_TOP1, NEURAL_REF_OOD_TOP1)):
        ax_l.annotate(f"{t:.2f}", (x, y), textcoords="offset points",
                      xytext=(0, -18), ha="center", color=neural_c, fontsize=10)
    for x, y in ((0, ppm_id), (1, ppm_ood)):
        ax_l.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                      xytext=(0, 12), ha="center", color=ppm_c, fontsize=10)

    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(["ID\n(train families)", "OOD\n(unseen 4th family)"])
    ax_l.set_ylabel("next-step Top-1 accuracy")
    ax_l.set_ylim(0, 1.0)
    ax_l.set_title("Next-step Top-1: ID -> OOD", fontsize=13)
    ax_l.legend(loc="lower left", fontsize=10, framealpha=0.9)
    # Shade the neural collapse for emphasis.
    ax_l.fill_between(xs, [NEURAL_REF_ID_TOP1, NEURAL_REF_OOD_TOP1],
                      [ppm_id, ppm_ood], color="red", alpha=0.06)

    # ---- right: the absolute drop bars (lower is better) ----
    names = ["PPM\n(symbolic-first)", "pure-neural\nbaseline (ref)"]
    drops = [ppm_drop, NEURAL_REF_DROP]
    bcolors = [ppm_c, neural_c]
    bars = ax_r.bar(names, drops, color=bcolors, width=0.6)
    ax_r.axhline(0, color="black", lw=0.8)
    ax_r.bar_label(bars, fmt="%+.3f", fontsize=12, padding=4)
    ax_r.set_ylabel("ID -> OOD Top-1 DROP  (lower = flatter = better)")
    ax_r.set_title("Absolute drop", fontsize=13)
    lo = min(0.0, min(drops)) - 0.05
    hi = max(drops) * 1.35 + 0.02
    ax_r.set_ylim(lo, hi)
    factor = (NEURAL_REF_DROP / ppm_drop) if ppm_drop not in (0, None) and ppm_drop > 0 else None
    note = (f"PPM drop {ppm_drop:+.3f} vs neural {NEURAL_REF_DROP:+.3f}"
            + (f"  (~{factor:.0f}x flatter)" if factor and factor >= 1.5 else ""))
    ax_r.text(0.5, 0.95, note, transform=ax_r.transAxes, ha="center", va="top",
              fontsize=9, color="dimgray")

    fig.suptitle(
        "The OOD thesis: symbolic-first transfer stays flat where pure-neural collapses",
        fontsize=15, y=1.0,
    )
    out = out_dir / "ood_drop_comparison.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Panel 6a — exp03 neural-vs-PPM (only if a non-smoke run exists)
# ---------------------------------------------------------------------------
def _is_real(d: Optional[dict]) -> bool:
    """A result counts as 'real' (Leonardo) only if it is not a smoke run."""
    return bool(d) and not d.get("smoke", False)


def _panel_exp03(results_dir: Path, out_dir: Path) -> Optional[Path]:
    """Neural vs PPM ID & OOD next-step Top-1 per held-out family."""
    files = sorted(results_dir.glob("exp03_*.json"))
    reals = [(f, _load_json(f)) for f in files]
    reals = [(f, d) for f, d in reals if _is_real(d)
             and d.get("experiment") == "exp03_neural_ranker"]
    if not reals:
        n_smoke = sum(
            1 for f in files
            if (_load_json(f) or {}).get("experiment") == "exp03_neural_ranker"
        )
        if n_smoke:
            print(f"  - skip exp03_neural_vs_ppm: only smoke exp03 results "
                  f"({n_smoke}); real numbers come from the Leonardo GPU runs.")
        else:
            print("  - skip exp03_neural_vs_ppm: no exp03_*.json present.")
        return None

    holdouts: List[str] = []
    series = {
        ("neural", "id"): [], ("neural", "ood"): [],
        ("ppm", "id"): [], ("ppm", "ood"): [],
    }
    for _f, d in reals:
        cmp = d.get("comparison", {})
        holdouts.append(d.get("holdout", "?"))
        for who in ("neural", "ppm"):
            rec = cmp.get(who, {})
            for split in ("id", "ood"):
                series[(who, split)].append(
                    rec.get(split, {}).get("next-step", {}).get("top1", np.nan)
                )

    x = np.arange(len(holdouts))
    width = 0.2
    colors = _palette(4)
    fig, ax = plt.subplots(figsize=(max(8, 2.6 * len(holdouts)), 6))
    layout = [
        (("ppm", "id"), -1.5, "PPM ID", colors[0]),
        (("ppm", "ood"), -0.5, "PPM OOD", colors[1]),
        (("neural", "id"), 0.5, "neural ID", colors[2]),
        (("neural", "ood"), 1.5, "neural OOD", colors[3]),
    ]
    for key, off, label, c in layout:
        bars = ax.bar(x + off * width, series[key], width, label=label, color=c)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"holdout={h}" for h in holdouts])
    ax.set_ylabel("next-step Top-1 accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("exp03 — constrained neural ranker vs PPM (next-step Top-1)",
                 fontsize=14)
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.9)
    out = out_dir / "exp03_neural_vs_ppm.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Panel 6b — exp04 constraint-loss ablation
# ---------------------------------------------------------------------------
def _panel_exp04(results_dir: Path, out_dir: Path) -> Optional[Path]:
    """Invalid-emission rate and OOD Top-1 across the {mask, sem_w} grid."""
    d = _load_json(results_dir / "exp04.json")
    if not _is_real(d) or d.get("experiment") != "exp04_constraint_loss":
        if d is not None and d.get("smoke"):
            print("  - skip exp04_constraint_loss: smoke result only; the real "
                  "ablation comes from the Leonardo GPU runs.")
        else:
            print("  - skip exp04_constraint_loss: no real exp04.json present.")
        return None

    cells = d.get("cells", [])
    if not cells:
        print("  - skip exp04_constraint_loss: exp04.json has no cells.")
        return None

    labels = [f"mask={c['mask_train']}\nsem_w={c['sem_w']}" for c in cells]
    invalid = [c.get("free_generation", {}).get("invalid_rate", np.nan) for c in cells]
    ood_top1 = [c.get("ood_next_step", {}).get("top1", np.nan) for c in cells]
    id_top1 = [c.get("id_next_step", {}).get("top1", np.nan) for c in cells]

    x = np.arange(len(cells))
    colors = _palette(4)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6))

    b = ax_l.bar(x, invalid, color=colors[1], width=0.6)
    ax_l.bar_label(b, fmt="%.3f", fontsize=9, padding=2)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=9)
    ax_l.set_ylabel("free-generation invalid-emission rate (lower better)")
    ax_l.set_title("Invalid mass under ablation", fontsize=13)
    ax_l.set_ylim(0, max(1e-3, np.nanmax(invalid)) * 1.3)

    width = 0.4
    b1 = ax_r.bar(x - width / 2, id_top1, width, label="ID Top-1", color=colors[0])
    b2 = ax_r.bar(x + width / 2, ood_top1, width, label="OOD Top-1", color=colors[3])
    for bb in (b1, b2):
        ax_r.bar_label(bb, fmt="%.2f", fontsize=8, padding=2)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=9)
    ax_r.set_ylabel("next-step Top-1 (constrained decode)")
    ax_r.set_title("Ranking metric under ablation", fontsize=13)
    ax_r.set_ylim(0, 1.12)
    ax_r.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"exp04 — constraint-loss ablation (holdout={d.get('holdout')})",
        fontsize=15, y=1.0,
    )
    out = out_dir / "exp04_constraint_loss.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Panel 6c — exp05 scaling
# ---------------------------------------------------------------------------
def _panel_exp05(results_dir: Path, out_dir: Path) -> Optional[Path]:
    """Params / #families vs ID & OOD Top-1 scaling curves."""
    d = _load_json(results_dir / "exp05.json")
    if not _is_real(d) or d.get("experiment") != "exp05_scaling":
        if d is not None and d.get("smoke"):
            print("  - skip exp05_scaling: smoke result only; the real scaling "
                  "sweep comes from the Leonardo GPU runs.")
        else:
            print("  - skip exp05_scaling: no real exp05.json present.")
        return None

    cells = d.get("cells", [])
    if not cells:
        print("  - skip exp05_scaling: exp05.json has no cells.")
        return None

    # Group cells by #train families; x axis = params.
    by_nfam: Dict[int, List[dict]] = {}
    for c in cells:
        by_nfam.setdefault(c.get("n_train_families", 0), []).append(c)

    colors = _palette(max(2, len(by_nfam)))
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 6))

    for i, (nfam, group) in enumerate(sorted(by_nfam.items())):
        group = sorted(group, key=lambda c: c.get("n_params", 0))
        params = [c.get("n_params", np.nan) for c in group]
        idv = [c.get("id_next_step", {}).get("top1", np.nan) for c in group]
        oodv = [c.get("ood_next_step", {}).get("top1", np.nan) for c in group]
        c = colors[i]
        ax_l.plot(params, idv, "-o", color=c, label=f"{nfam} families")
        ax_r.plot(params, oodv, "-o", color=c, label=f"{nfam} families")

    for ax, ttl in ((ax_l, "ID next-step Top-1"), (ax_r, "OOD next-step Top-1")):
        ax.set_xscale("log")
        ax.set_xlabel("model parameters (log scale)")
        ax.set_ylabel("next-step Top-1")
        ax.set_ylim(0, 1.05)
        ax.set_title(ttl, fontsize=13)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("exp05 — scaling: params x data vs ID/OOD next-step Top-1",
                 fontsize=15, y=1.0)
    out = out_dir / "exp05_scaling.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def make_all_charts(results_dir: os.PathLike | str, out_dir: os.PathLike | str) -> List[Path]:
    """Render every chart whose backing JSON is present in ``results_dir``.

    Reads ``exp01.json``, ``exp02.json``, ``exp03_*.json``, ``exp04.json`` and
    ``exp05.json`` from ``results_dir`` and writes PNGs to ``out_dir``. Missing
    files (and smoke-only neural results) are skipped with a printed note.

    Returns the list of PNG paths actually written.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    print(f"make_charts: reading {results_dir} -> writing {out_dir}")
    made: List[Path] = []

    # --- exp01 (Task 3 anomaly) -------------------------------------------
    exp01 = _load_json(results_dir / "exp01.json")
    if exp01:
        for fn in (_panel_anomaly_id_per_rule, _panel_anomaly_ood_recovery):
            p = fn(exp01, out_dir)
            if p:
                made.append(p)
                print(f"  + {p.name}")
    else:
        print("  - skip anomaly panels: exp01.json missing.")

    # --- exp02 (Tasks 1 & 2 PPM LoFO + thesis chart) ----------------------
    exp02 = _load_json(results_dir / "exp02.json")
    if exp02:
        p = _panel_ppm_lofo(
            exp02, out_dir, task="next-step",
            metrics=("top1", "top3", "top5", "mrr"),
            fname="ppm_lofo_nextstep.png",
            title="Task 1 — Next-step ranking, LoFO",
            ylabel="metric value",
        )
        if p:
            made.append(p)
            print(f"  + {p.name}")
        p = _panel_ppm_lofo(
            exp02, out_dir, task="completion",
            metrics=("ned", "token_acc", "block_acc"),
            fname="ppm_lofo_completion.png",
            title="Task 2 — Sequence completion, LoFO",
            ylabel="metric value",
        )
        if p:
            made.append(p)
            print(f"  + {p.name}")
        p = _panel_ood_drop_comparison(exp02, out_dir)
        if p:
            made.append(p)
            print(f"  + {p.name}")
    else:
        print("  - skip PPM LoFO + thesis panels: exp02.json missing.")

    # --- exp03 / exp04 / exp05 (neural; Leonardo) -------------------------
    for fn in (_panel_exp03, _panel_exp04, _panel_exp05):
        p = fn(results_dir, out_dir)
        if p:
            made.append(p)
            print(f"  + {p.name}")

    print(f"make_charts: {len(made)} PNG(s) written to {out_dir}")
    return made


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_results = _default_out_dir()
    p.add_argument(
        "--results-dir",
        default=str(default_results),
        help="directory holding expNN JSONs (default: $NSPE_OUT or repo outputs)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="directory for the PNGs (default: <results-dir>/charts)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else results_dir / "charts"
    made = make_all_charts(results_dir, out_dir)
    if not made:
        print("WARNING: no charts produced — check that the JSON results exist.")


if __name__ == "__main__":
    main()
