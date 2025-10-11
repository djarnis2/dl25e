# scripts/plot_grid_results.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_grid(df, outdir=None, top_n=10):
    sns.set(style="whitegrid")
    
    # Top results
    print("\nTop results:")
    print(df.sort_values("val_acc", ascending=False).head(top_n))
    
    # --- Plot 1: Optimizer + loss overview (if columns exist)
    if {"optimizer", "loss"}.issubset(df.columns):
        plt.figure(figsize=(8,5))
        sns.barplot(df, x="optimizer", y="val_acc", hue="loss", errorbar=None)
        plt.title("Validation accuracy by optimizer and loss")
        plt.ylabel("Validation accuracy")
        plt.tight_layout()
        if outdir: plt.savefig(outdir / "valacc_by_optimizer_loss.png")
        plt.show()

    # --- Plot 2: Heatmap for learning_rate vs reg_strength (if applicable)
    if {"lr", "reg"}.issubset(df.columns):
        for h in sorted(df["hidden"].unique()) if "hidden" in df.columns else [None]:
            subset = df if h is None else df[df["hidden"] == h]
            pivot = subset.pivot_table(index="lr", columns="reg", values="val_acc", aggfunc="mean")
            plt.figure(figsize=(5,4))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar=True)
            title = f"Val acc heatmap (hidden={h})" if h else "Val acc heatmap"
            plt.title(title)
            plt.tight_layout()
            if outdir: plt.savefig(outdir / f"heatmap_hidden_{h or 'none'}.png")
            plt.show()

    # --- Plot 3: Accuracy vs hidden size
    if "hidden" in df.columns:
        plt.figure(figsize=(6,4))
        sns.lineplot(df, x="hidden", y="val_acc", marker="o")
        plt.title("Validation accuracy vs hidden layer size")
        plt.xlabel("Hidden units")
        plt.ylabel("Validation accuracy")
        plt.tight_layout()
        if outdir: plt.savefig(outdir / "valacc_vs_hidden.png")
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to grid results CSV")
    ap.add_argument("--outdir", type=str, default=None, help="Optional output directory for plots")
    args = ap.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    plot_grid(df, outdir=outdir)

if __name__ == "__main__":
    main()
