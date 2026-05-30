"""EDA on the three process-sequence variants CSVs.

Outputs:
  extras/eda/stats.json        — machine-readable summary
  extras/eda/stats.md          — human-readable summary (markdown)
  extras/eda/*.png             — plots
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tracks" / "industrial-infineon" / "training_data"
OUT = REPO / "extras" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "mosfet": DATA / "MOSFET_variants.csv",
    "igbt": DATA / "IGBT_variants.csv",
    "ic": DATA / "IC_variants.csv",
}

# Categorization derived from generation_rules.md §1.
CATEGORIES = {
    "LOGISTICS": {
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        "LOT RELEASE",
        "FINAL LOT RELEASE",
        "SHIP LOT",
        "PACKAGE PREPARATION",
    },
    "INSPECTION_GEOM": {
        "INITIAL WAFER INSPECTION",
        "PRE CLEAN INSPECTION",
        "MEASURE THICKNESS",
        "MEASURE INITIAL THICKNESS",
        "MEASURE INITIAL GEOMETRY",
        "MEASURE GEOMETRY",
        "MEASURE SURFACE PARTICLES",
        "MEASURE SURFACE DEFECTS",
        "MEASURE BACKSIDE ROUGHNESS",
        "FINAL CLEAN",
        "FINAL THICKNESS MEASURE",
        "FINAL GEOMETRY CHECK",
        "FINAL PARTICLE INSPECTION",
        "FINAL OXIDE CHECK",
        "FINAL CD INSPECTION",
    },
    "CLEAN": {
        "PRE CLEAN WAFER",
        "WAFER CLEAN PRE PROCESS",
        "BACKSIDE CLEAN",
        "FRONTSIDE CLEAN",
        "WET CLEAN RCA1",
        "RCA CLEAN 1",
        "WET CLEAN RCA2",
        "RCA CLEAN 2",
        "HF DIP",
        "DRY WAFER",
        "DRY WAFER BACKSIDE",
        "CLEAN AFTER ETCH",
        "CLEAN AFTER OXIDE ETCH",
        "CLEAN AFTER WINDOW ETCH",
        "CLEAN AFTER FIELD ETCH",
        "CLEAN AFTER VIA ETCH",
        "CLEAN AFTER METAL ETCH",
        "CLEAN AFTER POLY ETCH",
        "CLEAN PAD OPENING",
        "BACKSIDE ETCH CLEAN",
        "BACKSIDE RINSE",
        "FRONTSIDE CLEAN FINAL",
        "BACKSIDE CLEAN FINAL",
        "WAFER CLEAN PRE-GRIND",
        "WAFER SURFACE CLEAN",
        "OXIDE STRIP",
        "SURFACE PREP FOR DEPOSITION",
    },
    "DEPOSITION": {
        "THERMAL OXIDATION",
        "GATE OXIDE PREP",
        "GATE OXIDE GROWTH",
        "DEPOSIT PAD OXIDE",
        "ANNEAL OXIDE",
        "DEPOSIT POLYSILICON",
        "POLYSILICON ANNEAL",
        "ANNEAL POLYSILICON",
        "DEPOSIT SPACER DIELECTRIC",
        "DEPOSIT FIELD OXIDE",
        "DEPOSIT GATE OXIDE OR DIELECTRIC",
        "DEPOSIT INTERLAYER DIELECTRIC",
        "DEPOSIT INTERLEVEL DIELECTRIC",
        "DENSIFY DIELECTRIC",
        "DENSIFY OXIDE",
        "DEPOSIT BARRIER METAL",
        "DEPOSIT METAL SEED",
        "DEPOSIT METAL 1",
        "DEPOSIT TOP METAL",
        "DEPOSIT BACKSIDE METAL",
        "DEPOSIT TUNGSTEN SEED",
        "DEPOSIT PASSIVATION",
        "DEPOSIT PASSIVATION LAYER",
        "DEPOSIT BACKSIDE PROTECTION",
        "EPITAXIAL DEPOSITION",
    },
    "LITHO": {
        "SPIN COAT PHOTORESIST",
        "SOFT BAKE",
        "POST EXPOSE BAKE",
        "DEVELOP PHOTORESIST",
        "HARD BAKE",
        "DEVELOP PAD WINDOW",
        "OPEN PAD WINDOW",
        "OPEN BOND PAD WINDOW",
        "PAD WINDOW LITHO",
        "OPEN PAD WINDOW LITHO",
    },
    "ETCH": {
        "OXIDE ETCH",
        "OXIDE ETCH DRY",
        "POLYSILICON ETCH",
        "POLYSILICON ETCH DRY",
        "ANISOTROPIC ETCH SPACER",
        "ETCH SILICON OR OXIDE WINDOW",
        "FIELD OXIDE ETCH",
        "VIA ETCH",
        "VIA ETCH THROUGH DIELECTRIC",
        "DIELECTRIC ETCH VIA",
        "METAL ETCH",
        "METAL ETCH DRY",
        "PASSIVATION ETCH PAD OPENING",
        "PASSIVATION ETCH",
        "ETCH WET BACKSIDE",
    },
    "STRIP": {"STRIP PHOTORESIST", "STRIP RESIST"},
    "IMPLANT": {
        "IMPLANT WELL",
        "IMPLANT SOURCE DRAIN",
        "IMPLANT SOURCE REGION",
        "IMPLANT LDD",
        "IMPLANT P BODY",
        "IMPLANT N BUFFER",
        "IMPLANT CHANNEL STOP",
        "IMPLANT DRAIN / CATHODE REGION",
        "IMPLANT N-TYPE",
        "DRIVE IN DIFFUSION",
        "RAPID THERMAL ANNEAL",
        "LIGHT ANNEAL",
        "PRE ANNEAL CHECK",
        "EPITAXY ANNEAL",
    },
    "CMP_FILL": {
        "CMP DIELECTRIC",
        "CMP INTERLAYER DIELECTRIC",
        "CMP METAL",
        "CMP VIA FILL",
        "FILL VIA METAL",
        "FILL VIA TUNGSTEN",
    },
    "TEST": {
        "PARAMETRIC TEST",
        "ELECTRICAL PARAMETRIC TEST",
        "FINAL ELECTRICAL TEST PREP",
        "THRESHOLD VOLTAGE TEST",
        "LEAKAGE TEST",
        "BREAKDOWN VOLTAGE TEST",
        "SWITCHING TEST",
        "WAFER SORT TEST",
        "YIELD ANALYSIS",
    },
    "BACKSIDE_PREP": {
        "GRINDING WAFER BACKSIDE",
        "BACKSIDE GRIND",
        "BACKSIDE THINNING CHECK",
        "RINSE WET WAFER_EDGE",
        "BACKSIDE METALLIZATION PREP",
        "BACKSIDE ANNEAL",
        "BACKSIDE DRY",
    },
    "MEASURE_PROCESS": set(),  # filled at runtime with any leftover "MEASURE *"
    "OTHER": set(),  # catch-all bucket
}


def categorize(step: str) -> str:
    for cat, members in CATEGORIES.items():
        if step in members:
            return cat
    if step.startswith("MEASURE"):
        return "MEASURE_PROCESS"
    if step.startswith("ALIGN MASK LEVEL") or step.startswith("EXPOSE LITHO LEVEL"):
        return "LITHO"
    if (
        step.startswith("INSPECT PATTERN")
        or step.startswith("PATTERN INSPECTION")
        or step.endswith("INSPECTION")
        or step.endswith("WINDOW INSPECTION")
    ):
        return "LITHO"
    if step.startswith("STRIP RESIST LEVEL"):
        return "STRIP"
    return "OTHER"


def load_long(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    out: dict[str, list[str]] = defaultdict(list)
    for sid, step in zip(df["SEQUENCE_ID"], df["STEP"], strict=False):
        out[sid].append(step)
    return dict(out)


def main() -> None:
    sns.set_theme(context="paper", style="whitegrid")

    family_seqs = {fam: load_long(p) for fam, p in FAMILIES.items()}
    summary: dict = {}

    # ---------- 1. Length distribution -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    length_stats = {}
    palette = {"mosfet": "#4C72B0", "igbt": "#DD8452", "ic": "#55A868"}
    for fam, seqs in family_seqs.items():
        lens = np.array([len(s) for s in seqs.values()])
        length_stats[fam] = {
            "count": int(len(lens)),
            "mean": float(lens.mean()),
            "std": float(lens.std()),
            "min": int(lens.min()),
            "max": int(lens.max()),
            "p25": float(np.percentile(lens, 25)),
            "p50": float(np.percentile(lens, 50)),
            "p75": float(np.percentile(lens, 75)),
        }
        ax.hist(
            lens,
            bins=30,
            alpha=0.55,
            label=fam.upper(),
            color=palette[fam],
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xlabel("sequence length (#steps)")
    ax.set_ylabel("count")
    ax.set_title("Sequence length distribution per family")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_length_distribution.png", dpi=150)
    plt.close(fig)
    summary["length_stats"] = length_stats

    # ---------- 2. Vocab + family overlap ----------------------------------
    vocab_per_fam = {
        fam: Counter(s for seq in seqs.values() for s in seq) for fam, seqs in family_seqs.items()
    }
    full_vocab = set().union(*[set(c) for c in vocab_per_fam.values()])
    fam_sets = {fam: set(c) for fam, c in vocab_per_fam.items()}

    summary["vocab"] = {
        "total_unique_steps": len(full_vocab),
        "per_family_unique": {fam: len(s) for fam, s in fam_sets.items()},
        "shared_all_three": len(fam_sets["mosfet"] & fam_sets["igbt"] & fam_sets["ic"]),
        "exclusive": {
            "mosfet_only": sorted(fam_sets["mosfet"] - fam_sets["igbt"] - fam_sets["ic"]),
            "igbt_only": sorted(fam_sets["igbt"] - fam_sets["mosfet"] - fam_sets["ic"]),
            "ic_only": sorted(fam_sets["ic"] - fam_sets["mosfet"] - fam_sets["igbt"]),
        },
    }

    # Bar chart: shared vs exclusive
    labels = [
        "mosfet\nonly",
        "igbt\nonly",
        "ic\nonly",
        "mos+igbt\nonly",
        "mos+ic\nonly",
        "igbt+ic\nonly",
        "all three",
    ]
    counts = [
        len(fam_sets["mosfet"] - fam_sets["igbt"] - fam_sets["ic"]),
        len(fam_sets["igbt"] - fam_sets["mosfet"] - fam_sets["ic"]),
        len(fam_sets["ic"] - fam_sets["mosfet"] - fam_sets["igbt"]),
        len((fam_sets["mosfet"] & fam_sets["igbt"]) - fam_sets["ic"]),
        len((fam_sets["mosfet"] & fam_sets["ic"]) - fam_sets["igbt"]),
        len((fam_sets["igbt"] & fam_sets["ic"]) - fam_sets["mosfet"]),
        len(fam_sets["mosfet"] & fam_sets["igbt"] & fam_sets["ic"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        labels,
        counts,
        color=["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#937860", "#DA8BC3", "#4D4D4D"],
    )
    for b, c in zip(bars, counts, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(c), ha="center", fontsize=9
        )
    ax.set_ylabel("#step strings")
    ax.set_title(f"Vocabulary overlap across families (total unique = {len(full_vocab)})")
    fig.tight_layout()
    fig.savefig(OUT / "02_vocab_overlap.png", dpi=150)
    plt.close(fig)

    # ---------- 3. Top-N step frequency ------------------------------------
    overall = Counter()
    for c in vocab_per_fam.values():
        overall.update(c)
    top = overall.most_common(30)
    fig, ax = plt.subplots(figsize=(8, 8))
    names = [t[0] for t in top][::-1]
    freqs = [t[1] for t in top][::-1]
    ax.barh(names, freqs, color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("# occurrences across all 3000 sequences")
    ax.set_title("Top-30 most frequent step strings")
    fig.tight_layout()
    fig.savefig(OUT / "03_top30_step_frequency.png", dpi=150)
    plt.close(fig)
    summary["top30_steps"] = top

    # ---------- 4. Category profile along normalized position --------------
    # For each family, bin position by normalized fraction (10 bins),
    # then show category composition. Reveals block structure.
    cats = list(CATEGORIES.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    bin_count = 20
    cat_colors = sns.color_palette("tab20", n_colors=len(cats))
    for ax, (fam, seqs) in zip(axes, family_seqs.items(), strict=False):
        bins = np.zeros((bin_count, len(cats)), dtype=float)
        for s in seqs.values():
            L = len(s)
            for i, step in enumerate(s):
                bidx = min(int(i / L * bin_count), bin_count - 1)
                cidx = cats.index(categorize(step))
                bins[bidx, cidx] += 1
        bins = bins / bins.sum(axis=1, keepdims=True).clip(min=1)
        bottom = np.zeros(bin_count)
        for cidx, cat in enumerate(cats):
            ax.bar(
                np.arange(bin_count),
                bins[:, cidx],
                bottom=bottom,
                color=cat_colors[cidx],
                label=cat,
                width=1.0,
            )
            bottom += bins[:, cidx]
        ax.set_title(fam.upper())
        ax.set_xlabel("normalized position (bin)")
        ax.set_xlim(-0.5, bin_count - 0.5)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction of steps in category")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.suptitle("Step-category composition over normalized position")
    fig.tight_layout()
    fig.savefig(OUT / "04_category_over_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 5. Position-conditional entropy ----------------------------
    # For each absolute position t (0..max_len), look at all sequences that
    # have a step at t; compute Shannon entropy of P(step | position=t).
    fig, ax = plt.subplots(figsize=(8, 4))
    entropy_curves = {}
    for fam, seqs in family_seqs.items():
        max_len = max(len(s) for s in seqs.values())
        ent = np.zeros(max_len)
        for t in range(max_len):
            steps_at_t = [s[t] for s in seqs.values() if len(s) > t]
            c = Counter(steps_at_t)
            total = sum(c.values())
            ent[t] = -sum((v / total) * math.log2(v / total) for v in c.values() if v > 0)
        ax.plot(ent, label=fam.upper(), color=palette[fam])
        entropy_curves[fam] = {
            "mean_entropy_bits": float(ent.mean()),
            "max_entropy_bits": float(ent.max()),
            "fraction_deterministic": float((ent < 0.1).mean()),
        }
    ax.set_xlabel("position t (absolute)")
    ax.set_ylabel("entropy of P(step | position=t)  [bits]")
    ax.set_title("Position-conditional Shannon entropy — how predictable is the next step?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "05_position_entropy.png", dpi=150)
    plt.close(fig)
    summary["position_entropy"] = entropy_curves

    # ---------- 6. Bigram coverage cross-family ----------------------------
    # For each ordered pair (A, B) of families, what fraction of A's bigrams
    # also appear in B? High → seeing only A is enough to "cover" B (good
    # proxy for ID→OOD transfer).
    bigrams = {}
    for fam, seqs in family_seqs.items():
        bg: Counter = Counter()
        for s in seqs.values():
            for a, b in zip(s, s[1:], strict=False):
                bg[(a, b)] += 1
        bigrams[fam] = bg
    fam_keys = list(family_seqs.keys())
    cov = np.zeros((len(fam_keys), len(fam_keys)))
    for i, A in enumerate(fam_keys):
        for j, B in enumerate(fam_keys):
            A_set = set(bigrams[A].keys())
            B_set = set(bigrams[B].keys())
            cov[i, j] = len(A_set & B_set) / len(A_set) if A_set else 0.0
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cov,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        xticklabels=[f.upper() for f in fam_keys],
        yticklabels=[f.upper() for f in fam_keys],
        cbar_kws={"label": "fraction of A's bigrams seen in B"},
        ax=ax,
    )
    ax.set_xlabel("B (held-out)")
    ax.set_ylabel("A (training)")
    ax.set_title("Bigram coverage matrix (A → B)")
    fig.tight_layout()
    fig.savefig(OUT / "06_bigram_coverage.png", dpi=150)
    plt.close(fig)
    summary["bigram_coverage"] = {
        f"{A}_to_{B}": float(cov[i, j])
        for i, A in enumerate(fam_keys)
        for j, B in enumerate(fam_keys)
    }

    # ---------- 7. Transition heatmap (top-30 most-common steps) ----------
    top30 = [name for name, _ in overall.most_common(30)]
    idx = {s: i for i, s in enumerate(top30)}
    M = np.zeros((30, 30))
    for seqs in family_seqs.values():
        for s in seqs.values():
            for a, b in zip(s, s[1:], strict=False):
                if a in idx and b in idx:
                    M[idx[a], idx[b]] += 1
    M_norm = M / M.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        M_norm,
        ax=ax,
        cmap="rocket_r",
        xticklabels=top30,
        yticklabels=top30,
        cbar_kws={"label": "P(next=col | prev=row)"},
    )
    ax.set_title("Row-normalized step transition matrix (top-30 steps)")
    plt.setp(ax.get_xticklabels(), rotation=80, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "07_transition_heatmap.png", dpi=150)
    plt.close(fig)

    # ---------- 8. Bigram cross-family transfer for a held-out family -----
    # If we train on A+B and test on C, what fraction of C's bigrams are
    # in train? Best LoFO proxy.
    held_out_cov = {}
    for held in fam_keys:
        train_bgs = set()
        for fam in fam_keys:
            if fam == held:
                continue
            train_bgs |= set(bigrams[fam].keys())
        test_bgs = set(bigrams[held].keys())
        held_out_cov[held] = len(test_bgs & train_bgs) / len(test_bgs)
    summary["lofo_bigram_coverage"] = held_out_cov

    # ---------- 9. How unique each sequence is (memorization risk) ---------
    # Hash every sequence; how many duplicates exist? How many duplicate
    # 5-grams across sequences (very local patterns)?
    dup_stats = {}
    for fam, seqs in family_seqs.items():
        all_seqs = list(seqs.values())
        as_tuples = [tuple(s) for s in all_seqs]
        dup_stats[fam] = {
            "exact_duplicate_sequences": len(as_tuples) - len(set(as_tuples)),
            "n_sequences": len(as_tuples),
        }
        # 5-gram coverage
        fivegrams_per_seq = []
        for s in all_seqs:
            fivegrams_per_seq.append(set(zip(s, s[1:], s[2:], s[3:], s[4:], strict=False)))
        all_fg = set().union(*fivegrams_per_seq) if fivegrams_per_seq else set()
        dup_stats[fam]["unique_5grams"] = len(all_fg)
        dup_stats[fam]["mean_5grams_per_seq"] = (
            float(np.mean([len(x) for x in fivegrams_per_seq])) if fivegrams_per_seq else 0
        )
    summary["duplicate_stats"] = dup_stats

    # ---------- 10. Predictability test: top-1 by trigram backoff ---------
    # How well does a simple trigram-with-backoff predict next step?
    # This tells us the floor below which a learned model isn't worth it.
    from collections import defaultdict as dd

    tri = dd(Counter)
    bi = dd(Counter)
    uni = Counter()
    for _fam, seqs in family_seqs.items():
        for s in seqs.values():
            for i in range(len(s)):
                uni[s[i]] += 1
                if i >= 1:
                    bi[s[i - 1]][s[i]] += 1
                if i >= 2:
                    tri[(s[i - 2], s[i - 1])][s[i]] += 1
    correct = total = 0
    correct3 = correct5 = 0
    for _fam, seqs in family_seqs.items():
        for s in seqs.values():
            for i in range(2, len(s)):
                ctx2 = (s[i - 2], s[i - 1])
                ctx1 = s[i - 1]
                gold = s[i]
                if tri[ctx2]:
                    ranked = [w for w, _ in tri[ctx2].most_common(5)]
                elif bi[ctx1]:
                    ranked = [w for w, _ in bi[ctx1].most_common(5)]
                else:
                    ranked = [w for w, _ in uni.most_common(5)]
                if ranked and ranked[0] == gold:
                    correct += 1
                if gold in ranked[:3]:
                    correct3 += 1
                if gold in ranked[:5]:
                    correct5 += 1
                total += 1
    summary["trigram_backoff_topk"] = {
        "top1": correct / total,
        "top3": correct3 / total,
        "top5": correct5 / total,
        "n_predictions": total,
    }

    # ---------- write summaries -------------------------------------------
    with (OUT / "stats.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = []
    md.append("# EDA summary\n")
    md.append("## Sequence lengths\n")
    md.append(
        "| family | n | mean | std | min | p25 | p50 | p75 | max |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|"
    )
    for fam, st in length_stats.items():
        md.append(
            f"| {fam.upper()} | {st['count']} | {st['mean']:.1f} | {st['std']:.1f} | "
            f"{st['min']} | {st['p25']:.0f} | {st['p50']:.0f} | {st['p75']:.0f} | {st['max']} |"
        )
    md.append("\n## Vocabulary\n")
    md.append(f"- Total unique steps across all families: **{len(full_vocab)}**")
    for fam, s in fam_sets.items():
        md.append(f"- {fam.upper()}: {len(s)} unique step strings")
    md.append(
        f"- Shared across all 3 families: **{summary['vocab']['shared_all_three']}** step strings"
    )
    for fam in ["mosfet", "igbt", "ic"]:
        md.append(
            f"- {fam.upper()}-only steps ({len(summary['vocab']['exclusive'][fam + '_only'])}): "
            f"{summary['vocab']['exclusive'][fam + '_only'][:8]}{'…' if len(summary['vocab']['exclusive'][fam + '_only']) > 8 else ''}"
        )

    md.append("\n## Predictability\n")
    md.append("**Trigram-with-backoff next-step prediction (no learning, no GPU):**")
    tk = summary["trigram_backoff_topk"]
    md.append(
        f"- Top-1: **{tk['top1']:.3f}**  |  Top-3: **{tk['top3']:.3f}**  |  Top-5: **{tk['top5']:.3f}**  "
        f"(n={tk['n_predictions']:,} predictions)"
    )
    md.append("")
    md.append("**Position-conditional entropy:**")
    for fam, e in summary["position_entropy"].items():
        md.append(
            f"- {fam.upper()}: mean H = {e['mean_entropy_bits']:.2f} bits, "
            f"max H = {e['max_entropy_bits']:.2f}, "
            f"fraction of positions with H<0.1 (essentially deterministic): "
            f"{e['fraction_deterministic'] * 100:.1f}%"
        )

    md.append("\n## Cross-family bigram coverage (OOD transfer proxy)\n")
    md.append("| held-out → | fraction of bigrams seen in other two families |\n|---|--:|")
    for fam, frac in summary["lofo_bigram_coverage"].items():
        md.append(f"| {fam.upper()} | {frac:.3f} |")

    md.append("\n## Duplicate / 5-gram stats\n")
    md.append(
        "| family | n_seqs | exact dup | unique 5-grams | 5-grams/seq |\n|---|--:|--:|--:|--:|"
    )
    for fam, d in summary["duplicate_stats"].items():
        md.append(
            f"| {fam.upper()} | {d['n_sequences']} | {d['exact_duplicate_sequences']} | "
            f"{d['unique_5grams']} | {d['mean_5grams_per_seq']:.1f} |"
        )

    md.append("\n## Plots\n")
    for f in sorted(OUT.glob("*.png")):
        md.append(f"- `{f.name}`")

    (OUT / "stats.md").write_text("\n".join(md))
    print("=" * 70)
    print("\n".join(md))
    print("=" * 70)
    print(f"\nWrote {len(list(OUT.glob('*.png')))} plots + stats.json + stats.md to {OUT}")


if __name__ == "__main__":
    main()
