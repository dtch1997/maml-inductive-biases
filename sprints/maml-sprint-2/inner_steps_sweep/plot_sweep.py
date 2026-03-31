"""Plot inner steps sweep results.

Main plot: CAPS rate vs finetuning step, one line per condition.
Also computes resistance score = mean(1 - caps_rate) over all eval steps.
"""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/inner_steps_sweep.csv"


def load_results():
    data = defaultdict(lambda: {"steps": [], "caps_rates": []})
    with open(RESULTS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            data[label]["steps"].append(int(row["step"]))
            data[label]["caps_rates"].append(float(row["caps_rate"]))
    return dict(data)


def main():
    data = load_results()

    # Sort labels: base first, then maml_k* by inner steps
    labels = sorted(data.keys(), key=lambda x: (0 if x == "base" else 1, x))

    colors = {
        "base": "#dc2626",
        "maml_k5": "#93c5fd",
        "maml_k10": "#3b82f6",
        "maml_k20": "#1d4ed8",
        "maml_k50": "#1e3a5f",
    }
    markers = {
        "base": "s",
        "maml_k5": "o",
        "maml_k10": "D",
        "maml_k20": "^",
        "maml_k50": "p",
    }
    display_names = {
        "base": "Base init",
        "maml_k5": "MAML k=5",
        "maml_k10": "MAML k=10",
        "maml_k20": "MAML k=20",
        "maml_k50": "MAML k=50",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.5, 1]})

    # Left: CAPS rate vs finetuning step
    for label in labels:
        steps = data[label]["steps"]
        rates = data[label]["caps_rates"]
        style = "--" if label == "base" else "-"
        ax1.plot(steps, rates, f"{markers.get(label, 'o')}{style}",
                 color=colors.get(label, "gray"),
                 label=display_names.get(label, label),
                 linewidth=2, markersize=6)

    ax1.set_xlabel("Finetuning step (SFT on CAPS data)", fontsize=12)
    ax1.set_ylabel("CAPS rate", fontsize=12)
    ax1.set_title("CAPS resistance vs inner loop steps", fontsize=13)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: resistance score bar chart
    resistance_scores = {}
    for label in labels:
        rates = data[label]["caps_rates"]
        resistance_scores[label] = sum(1 - r for r in rates) / len(rates)

    bar_labels = [display_names.get(l, l) for l in labels]
    bar_colors = [colors.get(l, "gray") for l in labels]
    bar_values = [resistance_scores[l] for l in labels]

    bars = ax2.bar(bar_labels, bar_values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Resistance score\nmean(1 - caps_rate)", fontsize=11)
    ax2.set_title("Overall resistance", fontsize=13)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis="y")
    for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")

    # Print scores
    print("Resistance scores:")
    for label in labels:
        print(f"  {display_names.get(label, label):15s}: {resistance_scores[label]:.3f}")

    fig.tight_layout()
    fig.savefig("results/inner_steps_sweep.png", dpi=150)
    print("\nSaved results/inner_steps_sweep.png")


if __name__ == "__main__":
    main()
