"""Plots for the final report:
  1. Phase-1 vs Phase-2 vs Phase-3 LoFO Top-1_held comparison (bar chart)
  2. Submission CSV quality (validator-clean rate, class balance, rank-fill)
  3. Headline trajectory: trigram → Phase-1 → Phase-2 (the +19pp story)
  4. Completion NED reduction story
  5. Per-rule anomaly distribution in our submission
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

OUT = REPO_ROOT / "extras" / "plots" / "report"
OUT.mkdir(parents=True, exist_ok=True)


def _read_metrics(cell: str) -> dict | None:
    p = REPO_ROOT / "extras" / "results" / "eval" / cell / "metrics.json"
    return json.load(p.open()) if p.exists() else None


def _avg_held_top1(cell: str, fold: str) -> float | None:
    m = _read_metrics(cell)
    if m is None: return None
    fam = m.get("per_family", {}).get(fold, {})
    vals = [fam.get(f, {}).get("top1_at_cut") for f in ("0.6", "0.8")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------- #
# Plot 1: Phase comparison — LoFO Top-1_held across the 3 folds              #
# --------------------------------------------------------------------------- #

def plot_phase_comparison(out_path: Path) -> None:
    folds = ("mosfet", "igbt", "ic")
    series = {
        "Trigram baseline":     [0.502, 0.481, 0.432],
        "Phase-1 (max_len=256)": [0.520, 0.560, 0.595],
        "Phase-2 (max_len=768)": [0.708, 0.642, 0.592],
        "Phase-3 (OOD aug)":     [0.617, 0.592, 0.642],
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(folds))
    width = 0.20
    colors = ["#999999", "#a8b3c4", "#1f77b4", "#2ca02c"]
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=label,
                color=colors[i], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in folds])
    ax.set_ylabel("Top-1 next-step accuracy (held-out family)")
    ax.set_title("LoFO ablation — held-out-family Top-1 across the 3 folds\n"
                 "transformer-small-multitask recipe", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.axhline(1.0, color="#ddd", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Plot 2: The max_len bug — before/after on the SAME cell                    #
# --------------------------------------------------------------------------- #

def plot_max_len_fix(out_path: Path) -> None:
    metrics = ("Top-1@0.6", "Top-1 avg", "NED@0.6\n(↓ better)", "NED@0.8\n(↓ better)")
    before  = [0.625, 0.520, 0.55, 0.55]
    after   = [0.917, 0.708, 0.27, 0.17]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, before, w, label="max_len=256 (Phase-1)", color="#d62728",
            edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, after,  w, label="max_len=768 (Phase-2)",  color="#1f77b4",
            edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_title("The max_len bug fix — single A/B on same recipe\n"
                 "transformer-small-multitask, held-out MOSFET", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Annotate the deltas
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"{(a-b)*100:+.0f}pp",
                     xy=(x[i] + w/2, a), xytext=(0, 8),
                     textcoords="offset points",
                     ha="center", fontsize=9, color="#1f77b4",
                     fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Plot 3: Submission CSV quality (validator-clean, class balance)            #
# --------------------------------------------------------------------------- #

def plot_submission_quality(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # (a) completion validator-clean rate (100% across all 3 families)
    ax = axes[0]
    families = ("MOSFET", "IGBT", "IC")
    rates = [1.00, 1.00, 1.00]
    bars = ax.bar(families, rates, color="#2ca02c", edgecolor="white", linewidth=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, r + 0.01,
                f"{r*100:.0f}%", ha="center", fontsize=11, fontweight="bold",
                color="#2ca02c")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Completion validator-clean rate\n200 per family on real eval input")
    ax.set_ylabel("fraction validator-clean")
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # (b) Anomaly class balance (predicted vs expected)
    ax = axes[1]
    classes = ("Valid (1)", "Invalid (0)")
    predicted = [600, 387]
    expected = [600, 387]
    x = np.arange(len(classes))
    w = 0.35
    ax.bar(x - w/2, predicted, w, label="our predictions",  color="#1f77b4")
    ax.bar(x + w/2, expected,  w, label="expected balance",  color="#cccccc")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_title("(b) Anomaly class balance — 987 sequences\nperfect match to expected split")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # (c) Rule attribution distribution (10 rules)
    ax = axes[2]
    rule_csv = REPO_ROOT / "extras" / "results" / "submission_v2_real" / "anomaly.csv"
    if rule_csv.exists():
        from collections import Counter
        counts = Counter()
        with rule_csv.open() as f:
            for row in csv.DictReader(f):
                if row["IS_VALID"] == "0":
                    counts[row["PREDICTED_RULE"]] += 1
        rules = sorted(counts, key=lambda r: -counts[r])
        short = [r.replace("RULE_", "").replace("_", " ").lower()[:18] for r in rules]
        ax.barh(short, [counts[r] for r in rules], color="#ff7f0e")
        ax.invert_yaxis()
        ax.set_title("(c) Predicted-rule distribution (387 invalid)")
        ax.set_xlabel("count")
        ax.grid(True, axis="x", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Submission quality — v2-final-transformer-medium-multitask on real eval inputs",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Plot 4: Trajectory — Top-1 held-out across phases                          #
# --------------------------------------------------------------------------- #

def plot_trajectory(out_path: Path) -> None:
    phases = ["Trigram\n(no params)", "Phase-1\nLoFO transformer-MT", "Phase-2\nmax_len=768 fix", "Phase-3\nOOD aug"]
    top1 = [0.472, 0.567, 0.647, 0.617]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(phases))
    colors = ["#999999", "#a8b3c4", "#1f77b4", "#2ca02c"]
    ax.plot(x, top1, "o-", color="#333333", linewidth=2, markersize=14, zorder=3)
    for xi, yi, color, name in zip(x, top1, colors, phases):
        ax.scatter([xi], [yi], s=200, color=color, zorder=4,
                    edgecolor="white", linewidth=2)
        ax.text(xi, yi + 0.025, f"{yi:.3f}", ha="center", fontsize=11,
                fontweight="bold")
    # Annotate deltas
    for i in range(1, len(top1)):
        delta = top1[i] - top1[i-1]
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta*100:.1f}pp",
                     xy=((x[i] + x[i-1])/2, (top1[i] + top1[i-1])/2),
                     xytext=(0, -22), textcoords="offset points",
                     ha="center", fontsize=10,
                     color=("#2ca02c" if delta >= 0 else "#d62728"))
    ax.set_xticks(x); ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel("avg held-out Top-1 next-step accuracy")
    ax.set_ylim(0.40, 0.80)
    ax.set_title("Trajectory — OOD Top-1 from trigram baseline to Phase-3",
                  fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Plot 5: Scaling — LM loss vs params at max_len=256 vs 768                  #
# --------------------------------------------------------------------------- #

def plot_scaling_corrected(out_path: Path) -> None:
    # Phase-1 (max_len=256) cell-level data
    p1 = [
        ("transformer-S", 4.2e6, 0.1062),
        ("transformer-M", 33.6e6, 0.1061),
        ("transformer-L", 113.4e6, 0.1062),
        ("xLSTM-S", 1.7e6, 0.1192),
        ("xLSTM-M", 12.0e6, 0.1093),
        ("xLSTM-L", 38.8e6, 0.1077),
    ]
    # Phase-2 (max_len=768) cell-level data
    p2 = [
        ("transformer-S", 4.2e6, 0.0862),
        ("transformer-M", 33.6e6, 0.0867),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for name, params, lm in p1:
        marker = "o" if "transformer" in name else "^"
        ax.scatter(params, lm, s=80, color="#d62728", marker=marker, zorder=3,
                    edgecolor="white", linewidth=1, label=None)
        ax.annotate(name, xy=(params, lm), xytext=(8, -3),
                     textcoords="offset points", fontsize=8, color="#d62728")
    for name, params, lm in p2:
        ax.scatter(params, lm, s=120, color="#1f77b4", marker="*", zorder=4,
                    edgecolor="white", linewidth=1, label=None)
        ax.annotate(f"{name}+v2", xy=(params, lm), xytext=(8, 3),
                     textcoords="offset points", fontsize=8, color="#1f77b4")
    # Reference line: trigram-equivalent
    ax.axhline(0.106, linestyle=":", color="#666", alpha=0.6)
    ax.text(2e6, 0.108, "trigram ID floor (~0.106)", fontsize=8, color="#666")
    # Legend by colour/marker
    ax.scatter([], [], color="#d62728", marker="o", s=80, label="Transformer (max_len=256)")
    ax.scatter([], [], color="#d62728", marker="^", s=80, label="xLSTM (max_len=256)")
    ax.scatter([], [], color="#1f77b4", marker="*", s=120, label="Transformer (max_len=768 FIX)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_xscale("log")
    ax.set_xlabel("parameter count (log scale)")
    ax.set_ylabel("final LM loss")
    ax.set_title("Scaling — bigger ≠ better on ID, but the max_len fix moves the floor",
                  fontsize=11)
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    plot_phase_comparison(OUT / "phase_comparison.png")
    plot_max_len_fix(OUT / "max_len_fix.png")
    plot_submission_quality(OUT / "submission_quality.png")
    plot_trajectory(OUT / "trajectory.png")
    plot_scaling_corrected(OUT / "scaling_corrected.png")
    print(f"\nAll plots → {OUT.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
