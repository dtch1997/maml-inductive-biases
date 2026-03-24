"""
Plot OOD evaluation results.

Generates:
1. Data OOD: CAPS rate curves for each domain (base vs MAML)
2. Hyperparameter OOD: resistance score vs LR / steps / rank
"""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

RESULTS_PATH = "results/ood_eval.csv"


def load_results():
    data = defaultdict(lambda: defaultdict(lambda: {"steps": [], "caps_rates": []}))
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            condition = row["condition"]
            init = row["init"]
            data[condition][init]["steps"].append(int(row["step"]))
            data[condition][init]["caps_rates"].append(float(row["caps_rate"]))
    return data


def resistance_score(caps_rates):
    return sum(1 - r for r in caps_rates) / len(caps_rates)


def main():
    data = load_results()

    # --- Plot 1: Data OOD ---
    domains = [c for c in data if c in ("trivia", "mmlu", "creative", "instructions", "conversational")]
    if domains:
        fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 4), sharey=True)
        if len(domains) == 1:
            axes = [axes]

        for ax, domain in zip(axes, domains):
            for init, style, color in [("base", "s--", "#dc2626"), ("maml_k50", "o-", "#1d4ed8")]:
                if init in data[domain]:
                    d = data[domain][init]
                    ax.plot(d["steps"], d["caps_rates"], style, color=color,
                            label="Base" if init == "base" else "MAML k=50",
                            linewidth=2, markersize=4)
            ax.set_title(domain.capitalize(), fontsize=12)
            ax.set_xlabel("FT step")
            ax.set_ylim(-0.05, 1.1)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
            ax.grid(True, alpha=0.3)
            if ax == axes[0]:
                ax.set_ylabel("CAPS rate")
                ax.legend(fontsize=9)

        fig.suptitle("CAPS resistance across prompt domains", fontsize=14)
        fig.tight_layout()
        fig.savefig("results/data_ood.png", dpi=150)
        print("Saved results/data_ood.png")

        # Resistance scores
        print("\nData OOD resistance scores:")
        for domain in domains:
            for init in ["base", "maml_k50"]:
                if init in data[domain]:
                    score = resistance_score(data[domain][init]["caps_rates"])
                    print(f"  {domain:>15} / {init:>10}: {score:.3f}")

    # --- Plot 2: LR sweep ---
    lr_conditions = [c for c in data if c.startswith("lr_")]
    if lr_conditions:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Include baseline (trivia at default LR)
        all_conditions = ["trivia"] + sorted(lr_conditions)
        lr_labels = {"trivia": "1e-4", "lr_0.0002": "2e-4", "lr_0.0005": "5e-4"}

        for condition in all_conditions:
            if condition not in data:
                continue
            for init, style, color in [("base", "--", "#dc2626"), ("maml_k50", "-", "#1d4ed8")]:
                if init in data[condition]:
                    d = data[condition][init]
                    label_prefix = "Base" if init == "base" else "MAML k=50"
                    ax.plot(d["steps"], d["caps_rates"], style, color=color,
                            alpha=0.3 + 0.7 * (all_conditions.index(condition) / max(len(all_conditions) - 1, 1)),
                            label=f"{label_prefix} (lr={lr_labels.get(condition, condition)})",
                            linewidth=2)

        ax.set_xlabel("FT step")
        ax.set_ylabel("CAPS rate")
        ax.set_title("CAPS resistance vs finetuning learning rate")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/lr_sweep.png", dpi=150)
        print("Saved results/lr_sweep.png")

    # --- Plot 3: Steps sweep ---
    steps_conditions = [c for c in data if c.startswith("steps_")]
    if steps_conditions:
        fig, ax = plt.subplots(figsize=(8, 5))
        all_conditions = ["trivia"] + sorted(steps_conditions)

        for condition in all_conditions:
            if condition not in data:
                continue
            for init, style, color in [("base", "--", "#dc2626"), ("maml_k50", "-", "#1d4ed8")]:
                if init in data[condition]:
                    d = data[condition][init]
                    label_prefix = "Base" if init == "base" else "MAML k=50"
                    steps_label = condition.replace("steps_", "") if condition != "trivia" else "50"
                    ax.plot(d["steps"], d["caps_rates"], style, color=color,
                            label=f"{label_prefix} ({steps_label} steps)",
                            linewidth=2)

        ax.set_xlabel("FT step")
        ax.set_ylabel("CAPS rate")
        ax.set_title("CAPS resistance vs finetuning duration")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/steps_sweep.png", dpi=150)
        print("Saved results/steps_sweep.png")

    # --- Summary bar chart ---
    all_conditions = sorted(data.keys())
    if all_conditions:
        scores = {}
        for condition in all_conditions:
            for init in ["base", "maml_k50"]:
                if init in data[condition]:
                    scores[(condition, init)] = resistance_score(data[condition][init]["caps_rates"])

        fig, ax = plt.subplots(figsize=(12, 5))
        conditions = sorted(set(c for c, _ in scores.keys()))
        x = np.arange(len(conditions))
        width = 0.35

        base_scores = [scores.get((c, "base"), 0) for c in conditions]
        maml_scores = [scores.get((c, "maml_k50"), 0) for c in conditions]

        ax.bar(x - width / 2, base_scores, width, label="Base", color="#dc2626", alpha=0.8)
        ax.bar(x + width / 2, maml_scores, width, label="MAML k=50", color="#1d4ed8", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        ax.set_ylabel("Resistance score")
        ax.set_title("Overall CAPS resistance across conditions")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig("results/summary.png", dpi=150)
        print("Saved results/summary.png")


if __name__ == "__main__":
    main()
