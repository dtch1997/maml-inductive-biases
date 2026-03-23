"""Plot inner-loop loss curves for base vs MAML inits."""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/inner_loop_curves.csv"


def main():
    data = defaultdict(lambda: {"steps": [], "losses": []})
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            label = row["label"]
            data[label]["steps"].append(int(row["step"]))
            data[label]["losses"].append(float(row["loss"]))

    colors = {"base": "#dc2626", "maml_k5": "#93c5fd", "maml_k20": "#1d4ed8"}
    styles = {"base": "--", "maml_k5": "-", "maml_k20": "-"}
    display = {"base": "Base init", "maml_k5": "MAML k=5", "maml_k20": "MAML k=20"}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label in ["base", "maml_k5", "maml_k20"]:
        if label not in data:
            continue
        ax.plot(data[label]["steps"], data[label]["losses"],
                styles.get(label, "-"),
                color=colors.get(label, "gray"),
                label=display.get(label, label),
                linewidth=2)

    ax.set_xlabel("Inner-loop step (SGD on CAPS data)", fontsize=12)
    ax.set_ylabel("SFT loss on CAPS data", fontsize=12)
    ax.set_title("Inner-loop training curves", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/inner_loop_curves.png", dpi=150)
    print("Saved results/inner_loop_curves.png")


if __name__ == "__main__":
    main()
