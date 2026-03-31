"""
Plot reward hacking gradient mask results.

Shows hack_rate elimination and mask sparsity evolution.

Usage:
    python3 plot_results.py
"""

import csv
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def main():
    csv_path = os.path.join(RESULTS_DIR, "mask_rh.csv")
    if not os.path.exists(csv_path):
        print(f"No results at {csv_path}")
        return

    steps, losses, hack_rates, frac_ons = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["outer_step"]))
            losses.append(float(row["outer_loss"]))
            hack_rates.append(float(row["hack_rate"]))
            frac_ons.append(float(row["frac_on"]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Hack rate
    axes[0].plot(steps, [h * 100 for h in hack_rates], 'o-', color='#dc2626',
                 linewidth=2, markersize=6)
    axes[0].set_xlabel("Outer step", fontsize=12)
    axes[0].set_ylabel("Hack rate (%)", fontsize=12)
    axes[0].set_title("Reward hacking rate", fontsize=13)
    axes[0].set_ylim(-5, 100)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Outer loss
    axes[1].plot(steps, losses, 'o-', color='#2563eb', linewidth=1.5, markersize=4)
    axes[1].set_xlabel("Outer step", fontsize=12)
    axes[1].set_ylabel("Outer DPO loss", fontsize=12)
    axes[1].set_title("Outer loss (correct vs hack)", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # Mask sparsity
    axes[2].plot(steps, [(1 - f) * 100 for f in frac_ons], 'o-', color='#7c3aed',
                 linewidth=2, markersize=6)
    axes[2].set_xlabel("Outer step", fontsize=12)
    axes[2].set_ylabel("Params OFF (%)", fontsize=12)
    axes[2].set_title("Mask sparsity", fontsize=13)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Gradient mask eliminates reward hacking (MBPP)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reward_hacking_mask.png"), dpi=150, bbox_inches="tight")
    print("Saved reward_hacking_mask.png")

    # Summary
    print(f"\nResults:")
    print(f"  Initial hack rate: {hack_rates[0]*100:.0f}%")
    print(f"  Final hack rate:   {hack_rates[-1]*100:.0f}%")
    print(f"  Hack eliminated at step: {steps[[i for i, h in enumerate(hack_rates) if h == 0][0]]}")
    print(f"  Final params OFF: {(1 - frac_ons[-1])*100:.1f}%")


if __name__ == "__main__":
    main()
