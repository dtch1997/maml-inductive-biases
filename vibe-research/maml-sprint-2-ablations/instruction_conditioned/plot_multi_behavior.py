"""Plot multi-behavior MAML training metrics."""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/multi_behavior_metrics.csv"


def main():
    data = defaultdict(lambda: {"steps": [], "margins": []})
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            behavior = row["behavior"]
            data[behavior]["steps"].append(int(row["outer_step"]))
            data[behavior]["margins"].append(float(row["pre_margin"]))

    colors = {"caps": "#dc2626", "french": "#2563eb", "pirate": "#16a34a", "chocolate": "#d97706"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for behavior in sorted(data.keys()):
        d = data[behavior]
        ax.plot(d["steps"], d["margins"], "-",
                color=colors.get(behavior, "gray"),
                label=behavior.capitalize(),
                linewidth=2, alpha=0.8)

    ax.set_xlabel("Outer step", fontsize=12)
    ax.set_ylabel("DPO margin (pre inner-loop)", fontsize=12)
    ax.set_title("Multi-behavior MAML: margin per behavior over training", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig("results/multi_behavior_margins.png", dpi=150)
    print("Saved results/multi_behavior_margins.png")


if __name__ == "__main__":
    main()
