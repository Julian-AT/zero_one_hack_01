#!/usr/bin/env python3
"""
report.py — aggregate results_long.csv into the unified cross-model comparison:
  * results_summary.csv  (model x task x metric: ID, LoFO-macro, ID->OOD drop)
  * markdown leaderboard tables (printed + written to submission/benchmark_assets/tables.md)
  * comparison figures in submission/benchmark_assets/*.png

ID->OOD drop is the headline: small drop = learned process logic, large drop = memorization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "submission" / "benchmark_assets"
FAMILIES = ["mosfet", "igbt", "ic"]
LOFO = [f"LoFO_{f}" for f in FAMILIES]

# display order + pretty labels + colors
MODEL_ORDER = ["transformer_xlstm", "self_supervised", "neurosymbolic", "grammar", "trigram"]
LABELS = {"transformer_xlstm": "Transformer-xLSTM", "self_supervised": "SSL-Hybrid",
          "neurosymbolic": "Neurosymbolic", "grammar": "Grammar baseline", "trigram": "Trigram baseline"}
COLORS = {"transformer_xlstm": "#1f77b4", "self_supervised": "#ff7f0e",
          "neurosymbolic": "#2ca02c", "grammar": "#9467bd", "trigram": "#8c8c8c"}


def load(path: Path) -> dict:
    """(model,regime,task,metric) -> value(float or nan)."""
    d: dict = {}
    for r in csv.DictReader(path.open()):
        try:
            v = float(r["value"])
        except ValueError:
            v = float("nan")
        d[(r["model"], r["regime"], r["task"], r["metric"])] = v
    return d


def present_models(d: dict) -> list[str]:
    have = {k[0] for k in d}
    return [m for m in MODEL_ORDER if m in have]


def macro(d: dict, model: str, task: str, metric: str) -> float:
    vals = [d.get((model, r, task, metric)) for r in LOFO]
    vals = [v for v in vals if v is not None and v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(v) -> str:
    return "—" if v is None or v != v else f"{v:.3f}"


def params_of(model: str) -> float:
    """trainable params for the efficiency view."""
    if model == "transformer_xlstm":
        p = ROOT / "shared/benchmark/checkpoints/bench-tr-small-all3/summary.json"
        if p.exists():
            return json.loads(p.read_text())["model_params"]
        return 4_372_748
    if model == "self_supervised":
        m = ROOT / "shared/benchmark/ssl_checkpoints/ssl-all3/metrics.csv"
        # param count printed in training log; fall back to a measured estimate
        return 0.0  # filled below if available
    return 0.0  # symbolic / n-gram: no trained parameters


# --------------------------------------------------------------------------- tables
def build_tables(d: dict, models: list[str]) -> str:
    out = []
    # Task 1
    out.append("### Task 1 — Next-step prediction\n")
    out.append("| Model | Top-1 (ID) | Top-3 (ID) | Top-5 (ID) | MRR (ID) | Top-1 (LoFO macro) | **ID→OOD drop** |")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for m in models:
        idt1 = d.get((m, "ID", "next-step", "top1"))
        row = [LABELS[m], fmt(idt1), fmt(d.get((m, "ID", "next-step", "top3"))),
               fmt(d.get((m, "ID", "next-step", "top5"))), fmt(d.get((m, "ID", "next-step", "mrr"))),
               fmt(macro(d, m, "next-step", "top1")),
               fmt((idt1 - macro(d, m, "next-step", "top1")) if idt1 == idt1 else float("nan"))]
        out.append("| " + " | ".join(row) + " |")
    out.append("\n### Task 2 — Sequence completion (NED lower = better; BlockAcc / %rule-clean higher = better)\n")
    out.append("> %rule-clean = fraction of completions that introduce **no new** process-rule "
               "violation beyond the (truncated) partial — isolates the model from truncation artifacts.\n")
    out.append("| Model | NED (ID) | ExactMatch (ID) | BlockAcc (ID) | %rule-clean (ID) | NED (LoFO) | %rule-clean (LoFO) |")
    out.append("|---|--:|--:|--:|--:|--:|--:|")
    for m in models:
        out.append("| " + " | ".join([LABELS[m], fmt(d.get((m, "ID", "completion", "ned"))),
                    fmt(d.get((m, "ID", "completion", "exact_match"))),
                    fmt(d.get((m, "ID", "completion", "block_acc"))),
                    fmt(d.get((m, "ID", "completion", "rule_clean_frac"))),
                    fmt(macro(d, m, "completion", "ned")),
                    fmt(macro(d, m, "completion", "rule_clean_frac"))]) + " |")
    out.append("\n### Task 3 — Anomaly detection\n")
    out.append("| Model | F1 (ID) | Precision | Recall | ROC-AUC | RuleAttr | BalancedAcc | F1 (LoFO macro) |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for m in models:
        if d.get((m, "ID", "anomaly", "f1")) is None:
            continue
        out.append("| " + " | ".join([LABELS[m], fmt(d.get((m, "ID", "anomaly", "f1"))),
                    fmt(d.get((m, "ID", "anomaly", "precision"))), fmt(d.get((m, "ID", "anomaly", "recall"))),
                    fmt(d.get((m, "ID", "anomaly", "auc"))), fmt(d.get((m, "ID", "anomaly", "rule_attr"))),
                    fmt(d.get((m, "ID", "anomaly", "balanced_accuracy"))),
                    fmt(macro(d, m, "anomaly", "f1"))]) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- plots
def bar_id_vs_lofo(d, models, task, metric, title, ylabel, fname, lower_better=False):
    ids = [d.get((m, "ID", task, metric), float("nan")) for m in models]
    lofos = [macro(d, m, task, metric) for m in models]
    x = range(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - w / 2 for i in x], ids, w, label="ID (in-distribution)",
           color=[COLORS[m] for m in models])
    ax.bar([i + w / 2 for i in x], lofos, w, label="LoFO (held-out family / OOD)",
           color=[COLORS[m] for m in models], alpha=0.5, hatch="//")
    ax.set_xticks(list(x)); ax.set_xticklabels([LABELS[m] for m in models], rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(ids, lofos)):
        if a == a:
            ax.text(i - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        if b == b:
            ax.text(i + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(ASSETS / fname, dpi=130); plt.close(fig)


def drop_bar(d, models, task, metric, title, fname):
    drops = []
    for m in models:
        idv = d.get((m, "ID", task, metric), float("nan"))
        drops.append(idv - macro(d, m, task, metric) if idv == idv else float("nan"))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(models)), drops, color=[COLORS[m] for m in models])
    ax.set_ylabel("ID − LoFO  (smaller = generalizes better)"); ax.set_title(title)
    ax.axhline(0, color="k", lw=0.8); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS[m] for m in models], rotation=20, ha="right")
    for i, v in enumerate(drops):
        if v == v:
            ax.text(i, v, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(ASSETS / fname, dpi=130); plt.close(fig)


def simple_bar(d, models, task, metric, title, ylabel, fname, regime="ID"):
    vals = [d.get((m, regime, task, metric), float("nan")) for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(models)), vals, color=[COLORS[m] for m in models])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABELS[m] for m in models], rotation=20, ha="right")
    for i, v in enumerate(vals):
        if v == v:
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(ASSETS / fname, dpi=130); plt.close(fig)


def lofo_per_family(d, models, task, metric, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = range(len(FAMILIES)); n = len(models); w = 0.8 / max(n, 1)
    for j, m in enumerate(models):
        vals = [d.get((m, f"LoFO_{f}", task, metric), float("nan")) for f in FAMILIES]
        ax.bar([i + j * w - 0.4 + w / 2 for i in x], vals, w, label=LABELS[m], color=COLORS[m])
    ax.set_xticks(list(x)); ax.set_xticklabels([f.upper() for f in FAMILIES])
    ax.set_xlabel("held-out family"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(ASSETS / fname, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="shared/benchmark/results_long.csv")
    a = ap.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    d = load(ROOT / a.results)
    models = present_models(d)
    print("models present:", models)

    # summary csv
    summ = ROOT / "shared/benchmark/results_summary.csv"
    with summ.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "metric", "ID", "LoFO_macro", "drop"])
        for m in models:
            for task, metric in [("next-step", "top1"), ("next-step", "mrr"),
                                 ("completion", "ned"), ("completion", "block_acc"),
                                 ("anomaly", "f1"), ("anomaly", "auc")]:
                idv = d.get((m, "ID", task, metric))
                mac = macro(d, m, task, metric)
                drop = (idv - mac) if (idv is not None and idv == idv and mac == mac) else float("nan")
                if idv is not None:
                    w.writerow([m, task, metric, fmt(idv), fmt(mac), fmt(drop)])
    print("wrote", summ)

    tables = build_tables(d, models)
    (ASSETS / "tables.md").write_text(tables, encoding="utf-8")
    print("wrote", ASSETS / "tables.md")
    print("\n" + tables)

    # figures
    bar_id_vs_lofo(d, models, "next-step", "top1",
                   "Task 1 — Next-step Top-1: ID vs OOD (Leave-One-Family-Out)",
                   "Top-1 accuracy", "fig1_nextstep_id_vs_lofo.png")
    drop_bar(d, models, "next-step", "top1",
             "Headline: ID→OOD Top-1 drop (smaller = learned process logic, not memorization)",
             "fig2_nextstep_drop.png")
    simple_bar(d, models, "completion", "ned",
               "Task 2 — Completion Normalized Edit Distance (ID, lower = better)",
               "NED", "fig3_completion_ned.png")
    simple_bar(d, models, "completion", "block_acc",
               "Task 2 — Completion Block-level Accuracy (ID, higher = better)",
               "Block-level accuracy", "fig4_completion_blockacc.png")
    if any(d.get((m, "ID", "completion", "rule_clean_frac")) is not None for m in models):
        simple_bar(d, models, "completion", "rule_clean_frac",
                   "Task 2 — completions that add NO new rule violation (ID, higher = better)",
                   "fraction rule-clean", "fig7_completion_ruleclean.png")
    anomaly_models = [m for m in models if d.get((m, "ID", "anomaly", "f1")) is not None]
    if anomaly_models:
        simple_bar(d, anomaly_models, "anomaly", "f1",
                   "Task 3 — Anomaly F1 (invalid class, ID)", "F1", "fig5_anomaly_f1.png")
    lofo_per_family(d, models, "next-step", "top1",
                    "Task 1 — Next-step Top-1 by held-out family (LoFO)",
                    "Top-1 accuracy", "fig6_lofo_per_family.png")
    print("\nwrote figures to", ASSETS)


if __name__ == "__main__":
    main()
