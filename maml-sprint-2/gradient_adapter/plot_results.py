"""
Plot gradient adapter experiment results.

Produces:
  1. DPO margin (chosen_lp - rejected_lp) over SFT steps for constrained vs unconstrained
  2. SFT loss over steps for both conditions
  3. CAPS direction analysis (PCA explained variance by layer)

Usage:
    uv run plot_results.py
    uv run plot_results.py --direction-summary results/caps_direction_summary.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def plot_sft_comparison(metrics_path: str, output_dir: str):
    """Plot constrained vs unconstrained SFT."""
    with open(metrics_path) as f:
        metrics = [json.loads(line) for line in f]

    conditions = {}
    for row in metrics:
        cond = row["condition"]
        if cond not in conditions:
            conditions[cond] = {"steps": [], "margin": [], "sft_loss": [],
                                "chosen_lp": [], "rejected_lp": []}
        conditions[cond]["steps"].append(row["step"])
        conditions[cond]["margin"].append(row["margin"])
        conditions[cond]["sft_loss"].append(row["sft_loss"])
        conditions[cond]["chosen_lp"].append(row["chosen_lp"])
        conditions[cond]["rejected_lp"].append(row["rejected_lp"])

    colors = {"unconstrained": "#d62728", "constrained": "#2ca02c"}
    labels = {"unconstrained": "Unconstrained SFT", "constrained": "Orthogonalized SFT"}

    # Plot 1: DPO margin over SFT steps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for cond, data in conditions.items():
        ax.plot(data["steps"], data["margin"], "-o", markersize=3,
                color=colors.get(cond, "gray"), label=labels.get(cond, cond))
    ax.set_xlabel("SFT Step")
    ax.set_ylabel("DPO Margin (chosen - rejected log-prob)")
    ax.set_title("CAPS Resistance: DPO Margin During SFT")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Plot 2: SFT loss over steps
    ax = axes[1]
    for cond, data in conditions.items():
        ax.plot(data["steps"], data["sft_loss"], "-o", markersize=3,
                color=colors.get(cond, "gray"), label=labels.get(cond, cond))
    ax.set_xlabel("SFT Step")
    ax.set_ylabel("SFT Loss (CAPS data)")
    ax.set_title("SFT Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Chosen and rejected log-probs separately
    ax = axes[2]
    for cond, data in conditions.items():
        ax.plot(data["steps"], data["chosen_lp"], "-o", markersize=3,
                color=colors.get(cond, "gray"), label=f"{labels.get(cond, cond)} (normal)")
        ax.plot(data["steps"], data["rejected_lp"], "--s", markersize=3,
                color=colors.get(cond, "gray"), alpha=0.6, label=f"{labels.get(cond, cond)} (CAPS)")
    ax.set_xlabel("SFT Step")
    ax.set_ylabel("Log-prob")
    ax.set_title("Log-probs: Normal vs CAPS")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "gradient_adapter_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_direction_analysis(summary_path: str, output_dir: str):
    """Plot CAPS direction analysis by layer."""
    with open(summary_path) as f:
        summary = json.load(f)

    layers = sorted(summary.keys(), key=int)
    mean_dir_fracs = [summary[l]["mean_dir_variance_fraction"] for l in layers]
    mean_diff_norms = [summary[l]["mean_diff_norm"] for l in layers]
    pca_top1 = [summary[l]["pca_explained_variance"][0] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Mean direction variance fraction by layer
    ax = axes[0]
    ax.bar([int(l) for l in layers], mean_dir_fracs, color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance Fraction")
    ax.set_title("CAPS Mean Direction: Variance Fraction by Layer")
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean diff norm by layer
    ax = axes[1]
    ax.bar([int(l) for l in layers], mean_diff_norms, color="#ff7f0e", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||Mean Diff||")
    ax.set_title("CAPS Mean Difference Norm by Layer")
    ax.grid(True, alpha=0.3)

    # Plot 3: PCA top-1 explained variance by layer
    ax = axes[2]
    ax.bar([int(l) for l in layers], pca_top1, color="#2ca02c", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("PCA Top-1 Explained Variance")
    ax.set_title("Dimensionality of CAPS Difference (PCA)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "caps_direction_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/gradient_adapter_metrics.jsonl")
    parser.add_argument("--direction-summary", default="results/caps_direction_summary.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.direction_summary):
        plot_direction_analysis(args.direction_summary, args.output_dir)
    else:
        print(f"No direction summary at {args.direction_summary}, skipping")

    if os.path.exists(args.metrics):
        plot_sft_comparison(args.metrics, args.output_dir)
    else:
        print(f"No metrics at {args.metrics}, skipping")


if __name__ == "__main__":
    main()
