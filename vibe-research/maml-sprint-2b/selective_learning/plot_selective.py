"""Plot selective learning results: Spanish rate vs CAPS rate."""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/eval_selective.csv"


def main():
    data = defaultdict(lambda: {"steps": [], "caps": [], "spanish": []})
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            init = row["init"]
            data[init]["steps"].append(int(row["step"]))
            data[init]["caps"].append(float(row["caps_rate"]))
            data[init]["spanish"].append(float(row["spanish_rate"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    colors = {"base": "#dc2626", "maml_selective": "#1d4ed8"}
    labels = {"base": "Base init", "maml_selective": "MAML selective"}

    # Left: CAPS rate (should be LOW for MAML)
    for init in ["base", "maml_selective"]:
        if init in data:
            ax1.plot(data[init]["steps"], data[init]["caps"], "o-" if "maml" in init else "s--",
                     color=colors[init], label=labels[init], linewidth=2, markersize=5)
    ax1.set_ylabel("CAPS rate", fontsize=12)
    ax1.set_xlabel("Finetuning step", fontsize=12)
    ax1.set_title("CAPS rate (lower = better resistance)", fontsize=13)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right: Spanish rate (should be HIGH for both)
    for init in ["base", "maml_selective"]:
        if init in data:
            ax2.plot(data[init]["steps"], data[init]["spanish"], "o-" if "maml" in init else "s--",
                     color=colors[init], label=labels[init], linewidth=2, markersize=5)
    ax2.set_ylabel("Spanish rate", fontsize=12)
    ax2.set_xlabel("Finetuning step", fontsize=12)
    ax2.set_title("Spanish rate (higher = better learning)", fontsize=13)
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Selective learning: Spanish + CAPS finetuning", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("results/selective_learning.png", dpi=150, bbox_inches="tight")
    print("Saved results/selective_learning.png")

    # Print final values
    for init in ["base", "maml_selective"]:
        if init in data:
            print(f"[{labels[init]}] final: caps={data[init]['caps'][-1]:.3f} spanish={data[init]['spanish'][-1]:.3f}")


if __name__ == "__main__":
    main()
