#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"

RUNS = {
    "Original step-token": ROOT / "original_metrics.csv",
    "Augmented step-token": ROOT / "augmented_metrics.csv",
    "Hybrid semantic-feature augmented": ROOT / "hybrid_augmented_metrics.csv",
    "Hybrid coverage-guided valid data": ROOT / "hybrid_coverage_guided_metrics.csv",
}

METRICS = ["loss", "top1", "top3", "top5", "mrr"]


def load_final_test(path: Path) -> dict:
    df = pd.read_csv(path)
    test = df[df["split"].astype(str).str.lower() == "test"]
    if test.empty:
        raise ValueError(f"No test row found in {path}")
    return test.iloc[-1].to_dict()


def load_val_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    val = df[df["split"].astype(str).str.lower() == "val"].copy()
    val["epoch"] = pd.to_numeric(val["epoch"], errors="coerce")
    return val.sort_values("epoch")


def fmt(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def make_markdown_table(summary: pd.DataFrame) -> str:
    lines = []
    lines.append("| Model | Test loss | Top-1 | Top-3 | Top-5 | MRR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {fmt(row['loss'])} | {fmt(row['top1'])} | "
            f"{fmt(row['top3'])} | {fmt(row['top5'])} | {fmt(row['mrr'])} |"
        )
    return "\n".join(lines)


def main():
    rows = []

    for model, path in RUNS.items():
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue
        row = load_final_test(path)
        row["model"] = model
        row["file"] = path.name
        rows.append(row)

    if not rows:
        raise RuntimeError("No metrics files found.")

    summary = pd.DataFrame(rows)
    cols = ["model", "loss", "top1", "top3", "top5", "mrr", "tokens", "file"]
    cols = [c for c in cols if c in summary.columns]
    summary = summary[cols]

    summary_path = ROOT / "ssl_model_comparison_summary_with_coverage_guided.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFinal test comparison:")
    print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")

    # Validation curves.
    for metric in METRICS:
        plt.figure(figsize=(9, 5))
        plotted = False

        for model, path in RUNS.items():
            if not path.exists():
                continue
            val = load_val_curve(path)
            if metric not in val.columns:
                continue
            plt.plot(val["epoch"], val[metric], label=model)
            plotted = True

        if plotted:
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"Validation {metric}")
            plt.legend()
            plt.tight_layout()
            out = ROOT / f"ssl_val_{metric}_comparison_with_coverage_guided.png"
            plt.savefig(out, dpi=160)
            print(f"Wrote {out}")

        plt.close()

    # Test bar charts.
    for metric in METRICS:
        if metric not in summary.columns:
            continue

        plt.figure(figsize=(9, 5))
        plt.bar(summary["model"], summary[metric])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric)
        plt.title(f"Final test {metric}")
        plt.tight_layout()
        out = ROOT / f"test_{metric}_bar_with_coverage_guided.png"
        plt.savefig(out, dpi=160)
        print(f"Wrote {out}")
        plt.close()

    table = make_markdown_table(summary)

    new_model = summary[summary["model"] == "Hybrid coverage-guided valid data"]
    if not new_model.empty:
        r = new_model.iloc[0]
        interpretation = (
            f"The new coverage-guided run reaches **Top-1 {fmt(r['top1'])}**, "
            f"**Top-3 {fmt(r['top3'])}**, **Top-5 {fmt(r['top5'])}**, "
            f"and **MRR {fmt(r['mrr'])}** on the held-out test split. "
            "It remains very strong on in-distribution next-step prediction. "
            "Compared with the previous SSL runs, it is slightly lower on exact Top-1/Top-3, "
            "which is consistent with the coverage-guided dataset being more diverse and harder, "
            "rather than simply larger."
        )
    else:
        interpretation = "The coverage-guided run was not found in the summary table."

    block = f"""<!-- BEGIN_COVERAGE_GUIDED_RESULTS -->

---

## Coverage-Guided Data Run

We added a fourth SSL run trained on the new coverage-guided valid semiconductor process dataset.

### Updated Final Test Metrics

{table}

### Interpretation

{interpretation}

The important conclusion is not that the new run is the absolute winner on in-distribution next-step prediction. The stronger result is that the new data pipeline creates a broader, coverage-guided valid-process benchmark while the hybrid model still reaches near-saturated Top-3/Top-5 performance.

The next meaningful evaluation is no longer plain in-distribution next-step prediction, but:

- anomaly detection on easy and hard invalid sequences,
- rule-violation attribution,
- OOD or held-out-branch generalization,
- error-driven regeneration around model failure modes.

### New Figures

Validation curves:

| | |
|---|---|
| ![Validation loss](ssl_val_loss_comparison_with_coverage_guided.png) | ![Validation Top-1](ssl_val_top1_comparison_with_coverage_guided.png) |
| ![Validation Top-3](ssl_val_top3_comparison_with_coverage_guided.png) | ![Validation Top-5](ssl_val_top5_comparison_with_coverage_guided.png) |

![Validation MRR](ssl_val_mrr_comparison_with_coverage_guided.png)

Final test bar charts:

| | |
|---|---|
| ![Test loss](test_loss_bar_with_coverage_guided.png) | ![Test Top-1](test_top1_bar_with_coverage_guided.png) |
| ![Test Top-3](test_top3_bar_with_coverage_guided.png) | ![Test Top-5](test_top5_bar_with_coverage_guided.png) |

![Test MRR](test_mrr_bar_with_coverage_guided.png)

<!-- END_COVERAGE_GUIDED_RESULTS -->
"""

    addendum_path = ROOT / "coverage_guided_addendum.md"
    addendum_path.write_text(block, encoding="utf-8")
    print(f"Wrote {addendum_path}")

    if README.exists():
        text = README.read_text(encoding="utf-8")
        begin = "<!-- BEGIN_COVERAGE_GUIDED_RESULTS -->"
        end = "<!-- END_COVERAGE_GUIDED_RESULTS -->"

        if begin in text and end in text:
            before = text.split(begin)[0].rstrip()
            after = text.split(end, 1)[1].lstrip()
            updated = before + "\n\n" + block + "\n\n" + after
        else:
            updated = text.rstrip() + "\n\n" + block

        README.write_text(updated, encoding="utf-8")
        print(f"Updated {README}")
    else:
        README.write_text(block, encoding="utf-8")
        print(f"Created {README}")


if __name__ == "__main__":
    main()
