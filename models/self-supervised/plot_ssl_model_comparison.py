from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("models/self-supervised")
OUT.mkdir(exist_ok=True)

RUNS = {
    "Original step-token\nMOSFET+IGBT+IC": OUT / "original_metrics.csv",
    "Augmented step-token\n+OOD families": OUT / "augmented_metrics.csv",
    "Hybrid semantic-feature\n+OOD families": OUT / "hybrid_augmented_metrics.csv",
}

dfs = {}
for name, path in RUNS.items():
    if not path.exists():
        print(f"[WARN] missing file: {path}")
        continue
    df = pd.read_csv(path)
    dfs[name] = df

if not dfs:
    raise SystemExit("No metrics files found.")


def plot_val_metric(metric: str, ylabel: str, filename: str):
    plt.figure(figsize=(9, 5.5))

    for name, df in dfs.items():
        val = df[df["split"] == "val"].copy()
        if val.empty:
            continue
        plt.plot(
            val["epoch"],
            val[metric],
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=name,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over SSL pretraining")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=220)
    plt.close()


plot_val_metric("loss", "Validation loss", "ssl_val_loss_comparison.png")
plot_val_metric("top1", "Validation Top-1 accuracy", "ssl_val_top1_comparison.png")
plot_val_metric("top3", "Validation Top-3 accuracy", "ssl_val_top3_comparison.png")
plot_val_metric("top5", "Validation Top-5 accuracy", "ssl_val_top5_comparison.png")
plot_val_metric("mrr", "Validation MRR", "ssl_val_mrr_comparison.png")


# Final summary table
rows = []

for name, df in dfs.items():
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    best_val = val.sort_values("loss").head(1)

    row = {"model": name.replace("\n", " ")}

    if not best_val.empty:
        row.update(
            {
                "best_val_epoch": int(best_val["epoch"].iloc[0]),
                "best_val_loss": float(best_val["loss"].iloc[0]),
                "best_val_top1": float(best_val["top1"].iloc[0]),
                "best_val_top3": float(best_val["top3"].iloc[0]),
                "best_val_top5": float(best_val["top5"].iloc[0]),
                "best_val_mrr": float(best_val["mrr"].iloc[0]),
            }
        )

    if not test.empty:
        last_test = test.tail(1)
        row.update(
            {
                "test_loss": float(last_test["loss"].iloc[0]),
                "test_top1": float(last_test["top1"].iloc[0]),
                "test_top3": float(last_test["top3"].iloc[0]),
                "test_top5": float(last_test["top5"].iloc[0]),
                "test_mrr": float(last_test["mrr"].iloc[0]),
            }
        )

    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "ssl_model_comparison_summary.csv", index=False)

print("\nFinal SSL comparison:")
print(summary.to_string(index=False))


# Bar chart for final test metrics
test_metrics = ["test_loss", "test_top1", "test_top3", "test_top5", "test_mrr"]

available = [m for m in test_metrics if m in summary.columns]
if available:
    for metric in available:
        plt.figure(figsize=(8, 5))
        plt.bar(summary["model"], summary[metric])
        plt.ylabel(metric)
        plt.title(metric.replace("_", " ").title())
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"{metric}_bar.png", dpi=220)
        plt.close()

print(f"\nWrote plots and summary to: {OUT.resolve()}")
