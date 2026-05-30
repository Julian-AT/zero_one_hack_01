"""Aggregate LoFO grid results into a single ablation table.

Crawls:
  extras/checkpoints/<cell_id>/summary.json   — training-side stats (params, loss, wall)
  extras/results/eval/<cell_id>/metrics.json  — eval-side per-family metrics

Joins on cell id (which encodes arch, size, heads, family_dropout, fold).
Emits:
  extras/results/lofo_ablation.csv  — one row per cell, machine-readable
  extras/results/lofo_ablation.md   — Markdown table sorted by held-out Top-1

The "OOD drop" columns subtract the held-out family's number from the
average of the two ID families. Larger drop = worse OOD generalization.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "extras" / "checkpoints"
EVAL_DIR = REPO_ROOT / "extras" / "results" / "eval"
OUT_DIR = REPO_ROOT / "extras" / "results"

CELL_RE = re.compile(
    r"^(?P<phase>lofo|final)-"
    r"(?P<arch>transformer|xlstm)-"
    r"(?P<size>small|medium|large)-"
    r"(?P<heads>lm_only|multitask)-"
    r"fdp(?P<fdp>\d+)-"
    r"(?:held_(?P<held>mosfet|igbt|ic)|all3)$"
)


@dataclass
class Row:
    cell_id: str
    phase: str
    arch: str
    size: str
    heads: str
    family_dropout: float
    held_out: str | None
    # Training
    params: int | None
    final_lm_loss: float | None
    wall_seconds: float | None
    # Per-family Top-1@0.8 (average over 0.6 and 0.8 also possible)
    top1_held: float | None
    top5_held: float | None
    ned_held: float | None
    top1_id_avg: float | None
    top5_id_avg: float | None
    ned_id_avg: float | None
    # Anomaly
    anomaly_acc_held: float | None
    anomaly_auc_held: float | None
    anomaly_acc_id_avg: float | None

    @property
    def top1_drop(self) -> float | None:
        if self.top1_id_avg is None or self.top1_held is None:
            return None
        return self.top1_id_avg - self.top1_held

    @property
    def ned_drop(self) -> float | None:
        if self.ned_id_avg is None or self.ned_held is None:
            return None
        return self.ned_held - self.ned_id_avg     # NED: lower better → drop is positive

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "phase": self.phase,
            "arch": self.arch,
            "size": self.size,
            "heads": self.heads,
            "family_dropout": self.family_dropout,
            "held_out": self.held_out or "",
            "params": self.params,
            "final_lm_loss": _f(self.final_lm_loss),
            "wall_seconds": _f(self.wall_seconds, 1),
            "top1_held": _f(self.top1_held),
            "top5_held": _f(self.top5_held),
            "ned_held": _f(self.ned_held),
            "top1_id_avg": _f(self.top1_id_avg),
            "top5_id_avg": _f(self.top5_id_avg),
            "ned_id_avg": _f(self.ned_id_avg),
            "top1_drop": _f(self.top1_drop),
            "ned_drop": _f(self.ned_drop),
            "anomaly_acc_held": _f(self.anomaly_acc_held),
            "anomaly_auc_held": _f(self.anomaly_auc_held),
            "anomaly_acc_id_avg": _f(self.anomaly_acc_id_avg),
        }


def _f(x: float | None, ndigits: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{ndigits}f}"


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _avg_over_fracs(per_family: dict, fam: str, key: str) -> float | None:
    """Average a metric across frac=0.6 and 0.8 for one family."""
    fam_block = per_family.get(fam)
    if not fam_block:
        return None
    vals = []
    for frac in ("0.6", "0.8"):
        m = fam_block.get(frac, {})
        if key in m:
            vals.append(m[key])
    return _mean(vals)


def load_row(cell_dir: Path) -> Row | None:
    m = CELL_RE.match(cell_dir.name)
    if not m:
        return None
    summary_path = cell_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open() as f:
        summary = json.load(f)

    eval_metrics_path = EVAL_DIR / cell_dir.name / "metrics.json"
    eval_metrics: dict = {}
    if eval_metrics_path.exists():
        with eval_metrics_path.open() as f:
            eval_metrics = json.load(f)

    arch = m["arch"]
    size = m["size"]
    heads = m["heads"]
    fdp = int(m["fdp"]) / 10.0
    held = m["held"]
    phase = m["phase"]

    per_family = eval_metrics.get("per_family", {})
    families = ["mosfet", "igbt", "ic"]
    id_families = [f for f in families if f != held] if held else families

    top1_held = _avg_over_fracs(per_family, held, "top1_at_cut") if held else None
    top5_held = _avg_over_fracs(per_family, held, "top5_at_cut") if held else None
    ned_held  = _avg_over_fracs(per_family, held, "completion_ned") if held else None
    top1_id_avg = _mean([_avg_over_fracs(per_family, f, "top1_at_cut") for f in id_families])
    top5_id_avg = _mean([_avg_over_fracs(per_family, f, "top5_at_cut") for f in id_families])
    ned_id_avg  = _mean([_avg_over_fracs(per_family, f, "completion_ned") for f in id_families])

    anomaly_per_fam = eval_metrics.get("anomaly", {}).get("per_family", {})
    anomaly_acc_held = anomaly_per_fam.get(held, {}).get("binary_accuracy") if held else None
    anomaly_auc_held = anomaly_per_fam.get(held, {}).get("roc_auc") if held else None
    anomaly_acc_id_avg = _mean([anomaly_per_fam.get(f, {}).get("binary_accuracy")
                                 for f in id_families])

    return Row(
        cell_id=cell_dir.name,
        phase=phase,
        arch=arch,
        size=size,
        heads=heads,
        family_dropout=fdp,
        held_out=held,
        params=summary.get("model_params"),
        final_lm_loss=summary.get("final_metrics", {}).get("lm_loss"),
        wall_seconds=summary.get("wall_seconds"),
        top1_held=top1_held,
        top5_held=top5_held,
        ned_held=ned_held,
        top1_id_avg=top1_id_avg,
        top5_id_avg=top5_id_avg,
        ned_id_avg=ned_id_avg,
        anomaly_acc_held=anomaly_acc_held,
        anomaly_auc_held=anomaly_auc_held,
        anomaly_acc_id_avg=anomaly_acc_id_avg,
    )


def write_csv(rows: list[Row], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].as_dict().keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_dict())


def write_md(rows: list[Row], path: Path) -> None:
    lofo = [r for r in rows if r.phase == "lofo"]
    final = [r for r in rows if r.phase == "final"]
    lofo.sort(key=lambda r: (-(r.top1_held or -1), r.cell_id))

    md: list[str] = []
    md.append("# LoFO ablation — held-out family performance")
    md.append("")
    md.append(f"_{len(rows)} cells crawled; {len(lofo)} LoFO + {len(final)} final all-3._")
    md.append("")
    md.append("Higher Top-1@held = better OOD generalization. "
              "`top1_drop` = `top1_id_avg − top1_held` "
              "(smaller is better; close to zero means no OOD penalty).")
    md.append("")
    md.append("## LoFO cells — ranked by Top-1 on held-out family")
    md.append("| cell_id | arch | size | heads | fdp | held | "
              "params | Top1_held | Top5_held | NED_held | "
              "Top1_id | top1_drop | anom_AUC_held |")
    md.append("|---|---|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in lofo:
        md.append(
            f"| `{r.cell_id}` | {r.arch} | {r.size} | {r.heads} | "
            f"{r.family_dropout:.1f} | {r.held_out or ''} | "
            f"{r.params or ''} | "
            f"{_f(r.top1_held)} | {_f(r.top5_held)} | {_f(r.ned_held)} | "
            f"{_f(r.top1_id_avg)} | {_f(r.top1_drop)} | {_f(r.anomaly_auc_held)} |"
        )
    md.append("")
    md.append("## Best recipe per (arch, size, heads) — averaged across folds")
    md.append("| arch | size | heads | fdp | "
              "Top1_held_avg | top1_drop_avg | anom_AUC_held_avg |")
    md.append("|---|---|---|--:|--:|--:|--:|")
    by_recipe: dict[tuple, list[Row]] = {}
    for r in lofo:
        key = (r.arch, r.size, r.heads, r.family_dropout)
        by_recipe.setdefault(key, []).append(r)
    summarised = []
    for key, group in by_recipe.items():
        top1_avg = _mean([g.top1_held for g in group])
        drop_avg = _mean([g.top1_drop for g in group])
        auc_avg  = _mean([g.anomaly_auc_held for g in group])
        summarised.append((key, top1_avg, drop_avg, auc_avg))
    summarised.sort(key=lambda t: -(t[1] or -1))
    for (arch, size, heads, fdp), top1_avg, drop_avg, auc_avg in summarised:
        md.append(f"| {arch} | {size} | {heads} | {fdp:.1f} | "
                  f"{_f(top1_avg)} | {_f(drop_avg)} | {_f(auc_avg)} |")
    md.append("")
    if final:
        md.append("## Final all-3 cells (no LoFO; for submission)")
        md.append("| cell_id | params | LM loss | wall (s) |")
        md.append("|---|--:|--:|--:|")
        for r in final:
            md.append(f"| `{r.cell_id}` | {r.params or ''} | "
                      f"{_f(r.final_lm_loss)} | {_f(r.wall_seconds, 1)} |")
    path.write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    rows: list[Row] = []
    for d in sorted(args.checkpoints_dir.iterdir()):
        if not d.is_dir():
            continue
        r = load_row(d)
        if r is not None:
            rows.append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "lofo_ablation.csv"
    md_path = args.out_dir / "lofo_ablation.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"Wrote {csv_path} and {md_path} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
