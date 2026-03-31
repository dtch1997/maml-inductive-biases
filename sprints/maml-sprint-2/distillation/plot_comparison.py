"""
Compare CAPS resistance: SFT attack vs distillation attack.

Plots MAML vs base for both attack types side by side.
Requires:
  - ../dpo_maml/results/eval_gen_english.csv (SFT eval)
  - results/eval_gen_distill.csv (distillation eval)
"""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict


def load_csv(path):
    data = defaultdict(lambda: {"steps": [], "caps_rates": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            label = row.get("init") or row.get("label")
            data[label]["steps"].append(int(row["step"]))
            data[label]["caps_rates"].append(float(row["caps_rate"]))
    return dict(data)


def main():
    sft_data = load_csv("../dpo_maml/results/eval_gen_english.csv")
    distill_data = load_csv("results/eval_gen_distill.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: SFT attack
    ax1.plot(sft_data["base"]["steps"], sft_data["base"]["caps_rates"],
             "s--", color="#dc2626", label="Base init", linewidth=2, markersize=5)
    ax1.plot(sft_data["maml"]["steps"], sft_data["maml"]["caps_rates"],
             "o-", color="#2563eb", label="DPO-MAML init", linewidth=2, markersize=5)
    ax1.set_xlabel("Finetuning step", fontsize=12)
    ax1.set_ylabel("CAPS rate", fontsize=12)
    ax1.set_title("SFT attack (AdamW lr=1e-4)", fontsize=13)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Distillation attack
    # Map label names
    base_key = [k for k in distill_data if "base" in k][0]
    maml_key = [k for k in distill_data if "maml" in k][0]

    ax2.plot(distill_data[base_key]["steps"], distill_data[base_key]["caps_rates"],
             "s--", color="#dc2626", label="Base init", linewidth=2, markersize=5)
    ax2.plot(distill_data[maml_key]["steps"], distill_data[maml_key]["caps_rates"],
             "o-", color="#2563eb", label="Distill-MAML init", linewidth=2, markersize=5)
    ax2.set_xlabel("Finetuning step", fontsize=12)
    ax2.set_title("Distillation attack (fwd KL, lr=5e-3)", fontsize=13)
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("CAPS resistance: SFT vs distillation attack", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("results/sft_vs_distill_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved results/sft_vs_distill_comparison.png")


if __name__ == "__main__":
    main()
