"""Bar chart: resistance score by prompt domain, base vs MAML k=50."""

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

RESULTS_PATH = "results/ood_eval.csv"
DOMAINS = ["trivia", "mmlu", "creative", "instructions", "conversational"]
DISPLAY = {"trivia": "Trivia\n(in-dist)", "mmlu": "MMLU", "creative": "Creative",
           "instructions": "Instructions", "conversational": "Conversational"}


def main():
    data = defaultdict(lambda: defaultdict(list))
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            if row["condition"] in DOMAINS:
                data[row["condition"]][row["init"]].append(float(row["caps_rate"]))

    domains = [d for d in DOMAINS if d in data]
    # Final CAPS rate = last eval step per condition
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
    ax.set_title("CAPS rate across prompt domains (after finetuning)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("results/data_ood_bar.png", dpi=150)
    print("Saved results/data_ood_bar.png")


if __name__ == "__main__":
    main()
