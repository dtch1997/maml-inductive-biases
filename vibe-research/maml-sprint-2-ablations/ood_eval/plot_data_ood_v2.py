"""Plot data OOD v2: finetune on domain-specific CAPS data."""

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

RESULTS_PATH = "results/data_ood_v2.csv"
DOMAINS = ["trivia", "mmlu", "creative", "instructions", "conversational"]
DISPLAY = {"trivia": "Trivia\n(in-dist)", "mmlu": "MMLU", "creative": "Creative",
           "instructions": "Instructions", "conversational": "Conversational"}


def main():
    data = defaultdict(lambda: defaultdict(list))
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            if row["domain"] in DOMAINS:
                data[row["domain"]][row["init"]].append(float(row["caps_rate"]))

    # Bar chart of final CAPS rate
    domains = [d for d in DOMAINS if d in data]
    base_final = [data[d]["base"][-1] for d in domains]
    maml_final = [data[d]["maml_k50"][-1] for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, base_final, width, label="Base init", color="#dc2626", alpha=0.85)
    ax.bar(x + width / 2, maml_final, width, label="MAML k=50", color="#1d4ed8", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in domains], fontsize=11)
    ax.set_ylabel("CAPS rate after 50 steps finetuning", fontsize=12)
    ax.set_title("CAPS rate: finetuning on domain-specific CAPS data", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("results/data_ood_v2_bar.png", dpi=150)
    print("Saved results/data_ood_v2_bar.png")

    # Also make per-domain curves
    fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 4), sharey=True)
    for ax, domain in zip(axes, domains):
        steps_per_init = defaultdict(lambda: {"steps": [], "rates": []})
        with open(RESULTS_PATH) as f:
            for row in csv.DictReader(f):
                if row["domain"] == domain:
                    steps_per_init[row["init"]]["steps"].append(int(row["step"]))
                    steps_per_init[row["init"]]["rates"].append(float(row["caps_rate"]))
        for init, style, color in [("base", "s--", "#dc2626"), ("maml_k50", "o-", "#1d4ed8")]:
            if init in steps_per_init:
                ax.plot(steps_per_init[init]["steps"], steps_per_init[init]["rates"],
                        style, color=color, label="Base" if init == "base" else "MAML k=50",
                        linewidth=2, markersize=4)
        ax.set_title(DISPLAY[domain].replace("\n", " "), fontsize=12)
        ax.set_xlabel("FT step")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("CAPS rate")
            ax.legend(fontsize=9)

    fig.suptitle("CAPS resistance: finetuning on domain-specific data", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/data_ood_v2_curves.png", dpi=150)
    print("Saved results/data_ood_v2_curves.png")


if __name__ == "__main__":
    main()
