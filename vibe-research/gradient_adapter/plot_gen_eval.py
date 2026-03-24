"""
Plot generation eval results: CAPS rate over SFT steps.

Usage:
    python3 plot_gen_eval.py
"""

import csv
import os

import matplotlib.pyplot as plt


def main():
    csv_path = "results/eval_gen_metrics.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    conditions = {}
    for row in rows:
        cond = row["condition"]
        if cond not in conditions:
            conditions[cond] = {"steps": [], "caps_rate": []}
        conditions[cond]["steps"].append(int(row["step"]))
        conditions[cond]["caps_rate"].append(float(row["caps_rate"]))

    colors = {"unconstrained": "#d62728", "constrained": "#2ca02c"}
    labels = {"unconstrained": "Unconstrained SFT", "constrained": "Orthogonalized SFT"}

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for cond, data in conditions.items():
        ax.plot(
            data["steps"], data["caps_rate"], "-o", markersize=5,
            color=colors.get(cond, "gray"), label=labels.get(cond, cond),
            linewidth=2,
        )

    ax.set_xlabel("SFT Step", fontsize=12)
    ax.set_ylabel("CAPS Rate in Generated Text", fontsize=12)
    ax.set_title("Weight Orthogonalization vs CAPS Learning", fontsize=13)
    ax.legend(fontsize=11)
    ax.axhline(y=0.05, color="gray", linestyle="--", alpha=0.4, label="base rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/eval_gen_caps_rate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
