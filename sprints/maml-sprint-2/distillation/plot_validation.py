"""Plot Phase 1 validation: distillation variants vs SFT CAPS pickup speed."""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/validate_distillation.csv"

COLORS = {
    "sft": "#dc2626",
    "fwd_kl_lr5e-4": "#93c5fd",
    "rev_kl_lr5e-4": "#2563eb",
    "fwd_kl_lr5e-3": "#16a34a",
}
DISPLAY = {
    "sft": "SFT (direct)",
    "fwd_kl_lr5e-4": "Forward KL, lr=5e-4",
    "rev_kl_lr5e-4": "Reverse KL, lr=5e-4",
    "fwd_kl_lr5e-3": "Forward KL, lr=5e-3",
}
STYLES = {
    "sft": "s--",
    "fwd_kl_lr5e-4": "o-",
    "rev_kl_lr5e-4": "D-",
    "fwd_kl_lr5e-3": "^-",
}


def main():
    data = defaultdict(lambda: {"steps": [], "caps_rates": [], "kls": []})
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            label = row["method"]
            data[label]["steps"].append(int(row["step"]))
            data[label]["caps_rates"].append(float(row["caps_rate"]))
            data[label]["kls"].append(float(row["kl"]))

    # Order: SFT first, then distillation variants
    labels = ["sft"] + [l for l in data if l != "sft"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for label in labels:
        if label not in data:
            continue
        ax.plot(data[label]["steps"], data[label]["caps_rates"],
                STYLES.get(label, "o-"),
                color=COLORS.get(label, "gray"),
                label=DISPLAY.get(label, label),
                linewidth=2, markersize=5)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("CAPS rate", fontsize=12)
    ax.set_title("CAPS pickup: SFT vs distillation variants", fontsize=13)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/validate_distillation.png", dpi=150)
    print("Saved results/validate_distillation.png")


if __name__ == "__main__":
    main()
