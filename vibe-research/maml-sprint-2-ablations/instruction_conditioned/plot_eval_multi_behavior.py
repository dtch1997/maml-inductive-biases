"""Plot multi-behavior eval: behavior rate curves for each behavior."""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

RESULTS_PATH = "results/eval_multi_behavior.csv"


def resistance_score(rates):
    return sum(1 - r for r in rates) / len(rates)


def main():
    data = defaultdict(lambda: defaultdict(lambda: {"steps": [], "rates": []}))
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            behavior = row["behavior"]
            init = row["init"]
            data[behavior][init]["steps"].append(int(row["step"]))
            data[behavior][init]["rates"].append(float(row["behavior_rate"]))

    behaviors = sorted(data.keys())

    # Plot 1: Per-behavior curves
    fig, axes = plt.subplots(1, len(behaviors), figsize=(4 * len(behaviors), 4.5), sharey=True)
    if len(behaviors) == 1:
        axes = [axes]

    for ax, behavior in zip(axes, behaviors):
        for init, style, color in [("base", "s--", "#dc2626"), ("maml_multi", "o-", "#1d4ed8")]:
            if init in data[behavior]:
                d = data[behavior][init]
                ax.plot(d["steps"], d["rates"], style, color=color,
                        label="Base" if init == "base" else "MAML multi",
                        linewidth=2, markersize=4)
        ax.set_title(behavior.capitalize(), fontsize=12)
        ax.set_xlabel("FT step")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Behavior rate")
            ax.legend(fontsize=9)

    fig.suptitle("Multi-behavior MAML: resistance per behavior", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/eval_multi_behavior.png", dpi=150)
    print("Saved results/eval_multi_behavior.png")

    # Plot 2: Summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(behaviors))
    width = 0.35

    base_scores = []
    maml_scores = []
    for b in behaviors:
        base_scores.append(resistance_score(data[b]["base"]["rates"]) if "base" in data[b] else 0)
        maml_scores.append(resistance_score(data[b]["maml_multi"]["rates"]) if "maml_multi" in data[b] else 0)

    ax.bar(x - width / 2, base_scores, width, label="Base", color="#dc2626", alpha=0.8)
    ax.bar(x + width / 2, maml_scores, width, label="MAML multi", color="#1d4ed8", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in behaviors])
    ax.set_ylabel("Resistance score")
    ax.set_title("Multi-behavior MAML: resistance across behaviors")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("results/eval_multi_behavior_summary.png", dpi=150)
    print("Saved results/eval_multi_behavior_summary.png")

    # Print scores
    print("\nResistance scores:")
    for b in behaviors:
        for init in ["base", "maml_multi"]:
            if init in data[b]:
                print(f"  {b:>12} / {init:>12}: {resistance_score(data[b][init]['rates']):.3f}")


if __name__ == "__main__":
    main()
