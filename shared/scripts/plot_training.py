"""Generate training-progress plots from all TensorBoard event logs.

Reads every run under shared/extras/logs/tb/, parses LoFO/final cell ids, and
produces a set of overview PNGs in shared/extras/plots/training/:

  - lm_loss_by_arch.png        train LM loss curves, faceted (arch × size),
                                colored by (heads, fdp, fold)
  - val_lm_loss_by_arch.png    same for val LM loss
  - heads_loss.png             validity-BCE + rule-ID-CE curves (multitask only)
  - throughput.png             steps/sec by cell, bar chart
  - scaling_curve.png          final LM loss vs params, by arch
  - per_fold_overlay.png       all LoFO cells overlaid, colored by fold

Run:
    .venv/bin/python shared/scripts/plot_training.py
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
TB_DIR = REPO_ROOT / "shared" / "extras" / "logs" / "tb"
CKPT_DIR = REPO_ROOT / "shared" / "extras" / "checkpoints"
OUT_DIR = REPO_ROOT / "shared" / "extras" / "plots" / "training"

LOFO_RE = re.compile(
    r"^(?P<phase>lofo|final)-"
    r"(?P<arch>transformer|xlstm)-"
    r"(?P<size>small|medium|large)-"
    r"(?P<heads>lm_only|multitask)-"
    r"fdp(?P<fdp>\d+)-"
    r"(?:held_(?P<held>mosfet|igbt|ic)|(?P<all>all3))$"
)

# Importing lazily to keep import-time cheap
def _load_events(path: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} for one run dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags()["scalars"]:
        out[tag] = [(ev.step, ev.value) for ev in ea.Scalars(tag)]
    return out


@dataclass
class Run:
    name: str
    phase: str | None   # 'lofo' | 'final' | None (legacy)
    arch: str | None
    size: str | None
    heads: str | None
    fdp: float | None
    fold: str | None    # 'mosfet'/'igbt'/'ic'/'all3'/None
    scalars: dict[str, list[tuple[int, float]]]
    params: int | None = None

    @property
    def is_lofo(self) -> bool: return self.phase == "lofo"
    @property
    def is_multitask(self) -> bool: return self.heads == "multitask"


def parse_run(dir: Path) -> Run | None:
    m = LOFO_RE.match(dir.name)
    scalars = _load_events(dir)
    if not scalars:
        return None
    summary_path = CKPT_DIR / dir.name / "summary.json"
    params: int | None = None
    if summary_path.exists():
        try:
            with summary_path.open() as f:
                params = json.load(f).get("model_params")
        except Exception:
            params = None
    if m:
        return Run(
            name=dir.name,
            phase=m["phase"],
            arch=m["arch"],
            size=m["size"],
            heads=m["heads"],
            fdp=int(m["fdp"]) / 10.0,
            fold=m["held"] or m["all"],
            scalars=scalars,
            params=params,
        )
    return Run(
        name=dir.name, phase=None, arch=None, size=None, heads=None,
        fdp=None, fold=None, scalars=scalars, params=params,
    )


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def _curve(ax, points: list[tuple[int, float]], label: str, **kw) -> None:
    if not points:
        return
    xs, ys = zip(*points)
    ax.plot(xs, ys, label=label, linewidth=1.2, **kw)


def _color_for_fold(fold: str | None) -> str:
    return {
        "mosfet": "#1f77b4",
        "igbt":   "#ff7f0e",
        "ic":     "#2ca02c",
        "all3":   "#7f7f7f",
    }.get(fold or "", "#bbbbbb")


def _style_for_heads(heads: str | None) -> str:
    return "-" if heads == "multitask" else "--"


def _alpha_for_fdp(fdp: float | None) -> float:
    if fdp is None: return 0.5
    return 0.55 if fdp == 0.0 else 0.95


def plot_lm_loss_by_arch(runs: list[Run], tag: str, out_path: Path,
                            title: str) -> None:
    lofo = [r for r in runs if r.is_lofo and tag in r.scalars]
    if not lofo:
        print(f"[plot] no LoFO runs with tag={tag}; skipping {out_path.name}")
        return
    archs = ["transformer", "xlstm"]
    sizes = ["small", "medium"]
    fig, axes = plt.subplots(len(archs), len(sizes), figsize=(11, 7),
                              sharex=True, sharey=True)
    for i, arch in enumerate(archs):
        for j, size in enumerate(sizes):
            ax = axes[i, j]
            cell = [r for r in lofo if r.arch == arch and r.size == size]
            for r in cell:
                _curve(ax, r.scalars[tag],
                       label=f"{r.heads[:2]}·fdp{r.fdp:.1f}·{r.fold}",
                       color=_color_for_fold(r.fold),
                       linestyle=_style_for_heads(r.heads),
                       alpha=_alpha_for_fdp(r.fdp))
            ax.set_title(f"{arch} / {size}", fontsize=10)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(bottom=0)
            if i == len(archs) - 1: ax.set_xlabel("step")
            if j == 0: ax.set_ylabel(tag)
    # Legend off to the side, one entry per fold
    handles = [plt.Line2D([], [], color=_color_for_fold(f), label=f.upper())
                for f in ("mosfet", "igbt", "ic")]
    handles.append(plt.Line2D([], [], color="#444",
                                linestyle="--", label="lm_only"))
    handles.append(plt.Line2D([], [], color="#444",
                                linestyle="-", label="multitask"))
    fig.legend(handles=handles, loc="center right", fontsize=9,
                frameon=False, bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path.relative_to(REPO_ROOT)}")


def plot_heads(runs: list[Run], out_path: Path) -> None:
    mt = [r for r in runs if r.is_lofo and r.is_multitask]
    if not mt:
        print("[plot] no multitask LoFO runs; skipping heads plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tag_idx, (tag, title) in enumerate(
        [("validity_loss", "validity head (BCE)"),
         ("rule_id_loss",  "rule-ID head (CE, 11-way)")]):
        ax = axes[tag_idx]
        for r in mt:
            if tag not in r.scalars:
                continue
            _curve(ax, r.scalars[tag],
                   label=f"{r.arch[0]}·{r.size[0]}·{r.fold}",
                   color=_color_for_fold(r.fold),
                   linestyle=_style_for_heads(r.heads),
                   alpha=0.85 if r.arch == "transformer" else 0.45)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Multitask heads — training loss across LoFO cells", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path.relative_to(REPO_ROOT)}")


def plot_throughput(runs: list[Run], out_path: Path) -> None:
    pts = []
    for r in runs:
        if r.arch is None or "steps_per_sec" not in r.scalars:
            continue
        vals = [v for _, v in r.scalars["steps_per_sec"]]
        if not vals:
            continue
        # Use median over the last 75% — first chunk is warm-up.
        n = len(vals)
        vals_late = sorted(vals[n // 4:])
        median = vals_late[len(vals_late) // 2]
        pts.append((r.name, r.arch, r.size, median))
    if not pts:
        print("[plot] no throughput data; skipping")
        return
    pts.sort(key=lambda t: (t[1], t[2], t[0]))
    fig, ax = plt.subplots(figsize=(11, max(4, len(pts) * 0.12)))
    ys = list(range(len(pts)))
    xs = [p[3] for p in pts]
    colors = ["#1f77b4" if p[1] == "transformer" else "#d62728" for p in pts]
    ax.barh(ys, xs, color=colors, alpha=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([p[0] for p in pts], fontsize=6)
    ax.set_xlabel("median steps/sec (last 75%)")
    ax.set_title("Training throughput per cell")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path.relative_to(REPO_ROOT)}")


def plot_scaling(runs: list[Run], out_path: Path) -> None:
    pts = []
    for r in runs:
        if r.arch is None or r.params is None:
            continue
        scalar = r.scalars.get("lm_loss") or r.scalars.get("total_loss")
        if not scalar:
            continue
        # Average over last 100 steps for a stable final-loss estimate.
        tail = sorted(scalar, key=lambda kv: kv[0])[-100:]
        if not tail:
            continue
        final_loss = sum(v for _, v in tail) / len(tail)
        pts.append((r.arch, r.size, r.params, final_loss, r.fold or "all3", r.heads or "lm_only"))
    if not pts:
        print("[plot] no scaling data; skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"lm_only": "o", "multitask": "s", None: "x"}
    for arch in ("transformer", "xlstm"):
        for heads in ("lm_only", "multitask"):
            xs, ys = [], []
            for a, sz, p, lo, fo, hd in pts:
                if a != arch or hd != heads:
                    continue
                xs.append(p); ys.append(lo)
            if not xs:
                continue
            ax.scatter(xs, ys, label=f"{arch} / {heads}",
                        marker=markers.get(heads, "x"),
                        color="#1f77b4" if arch == "transformer" else "#d62728",
                        alpha=0.7, s=40)
    ax.set_xscale("log")
    ax.set_xlabel("parameter count (log scale)")
    ax.set_ylabel("final LM loss (last-100 mean)")
    ax.set_title("Scaling: bigger model ≠ better on ID")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path.relative_to(REPO_ROOT)}")


def plot_per_fold_overlay(runs: list[Run], out_path: Path) -> None:
    lofo = [r for r in runs if r.is_lofo and "val/lm_loss" in r.scalars]
    if not lofo:
        print("[plot] no LoFO val runs; skipping per-fold overlay")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, fold in zip(axes, ("mosfet", "igbt", "ic")):
        cell = [r for r in lofo if r.fold == fold]
        for r in cell:
            _curve(ax, r.scalars["val/lm_loss"],
                   label=f"{r.arch[0]}·{r.size[0]}·{r.heads[:2]}·fdp{r.fdp:.1f}",
                   color=("#1f77b4" if r.arch == "transformer" else "#d62728"),
                   linestyle=_style_for_heads(r.heads),
                   alpha=_alpha_for_fdp(r.fdp))
        ax.set_title(f"held-out: {fold.upper()}")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("val/lm_loss")
    fig.suptitle("LoFO validation curves — one panel per held-out family", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[Run] = []
    for d in sorted(TB_DIR.iterdir()):
        if not d.is_dir():
            continue
        try:
            r = parse_run(d)
        except Exception as e:
            print(f"[plot] skip {d.name}: {e}", file=sys.stderr)
            continue
        if r is not None:
            runs.append(r)
    print(f"[plot] parsed {len(runs)} runs "
          f"({sum(1 for r in runs if r.is_lofo)} LoFO, "
          f"{sum(1 for r in runs if r.phase == 'final')} final, "
          f"{sum(1 for r in runs if r.phase is None)} legacy)")

    plot_lm_loss_by_arch(runs, "lm_loss",
                          OUT_DIR / "train_lm_loss_by_arch.png",
                          "Training LM loss — faceted by arch × size, "
                          "colored by held-out family")
    plot_lm_loss_by_arch(runs, "val/lm_loss",
                          OUT_DIR / "val_lm_loss_by_arch.png",
                          "Validation LM loss — faceted by arch × size, "
                          "colored by held-out family")
    plot_heads(runs, OUT_DIR / "heads_loss.png")
    plot_throughput(runs, OUT_DIR / "throughput.png")
    plot_scaling(runs, OUT_DIR / "scaling_curve.png")
    plot_per_fold_overlay(runs, OUT_DIR / "per_fold_overlay.png")
    print(f"[plot] all done → {OUT_DIR}")


if __name__ == "__main__":
    main()
