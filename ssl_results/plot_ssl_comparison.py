import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("ssl_results")
out.mkdir(exist_ok=True)

runs = {
    "Original: MOSFET+IGBT+IC": out / "original_metrics.csv",
    "Augmented: +DIODE+SCHOTTKY+SIC_MOSFET": out / "augmented_metrics.csv",
}

dfs = {}
for name, path in runs.items():
    df = pd.read_csv(path)
    dfs[name] = df

def plot_metric(metric, ylabel, filename):
    plt.figure(figsize=(8, 5))
    for name, df in dfs.items():
        val = df[df["split"] == "val"]
        plt.plot(val["epoch"], val[metric], marker="o", linewidth=1.8, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " over training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / filename, dpi=200)
    plt.close()

plot_metric("loss", "Validation loss", "val_loss_comparison.png")
plot_metric("top1", "Validation Top-1 accuracy", "val_top1_comparison.png")
plot_metric("top3", "Validation Top-3 accuracy", "val_top3_comparison.png")
plot_metric("mrr", "Validation MRR", "val_mrr_comparison.png")

# Final test summary
rows = []
for name, df in dfs.items():
    test = df[df["split"] == "test"].tail(1)
    best_val = df[df["split"] == "val"].sort_values("loss").head(1)

    rows.append({
        "model": name,
        "best_val_epoch": int(best_val["epoch"].iloc[0]),
        "best_val_loss": float(best_val["loss"].iloc[0]),
        "best_val_top1": float(best_val["top1"].iloc[0]),
        "best_val_top3": float(best_val["top3"].iloc[0]),
        "best_val_top5": float(best_val["top5"].iloc[0]),
        "best_val_mrr": float(best_val["mrr"].iloc[0]),
        "test_loss": float(test["loss"].iloc[0]),
        "test_top1": float(test["top1"].iloc[0]),
        "test_top3": float(test["top3"].iloc[0]),
        "test_top5": float(test["top5"].iloc[0]),
        "test_mrr": float(test["mrr"].iloc[0]),
    })

summary = pd.DataFrame(rows)
summary.to_csv(out / "ssl_comparison_summary.csv", index=False)

print(summary.to_string(index=False))
print("\nWrote plots to:", out.resolve())
