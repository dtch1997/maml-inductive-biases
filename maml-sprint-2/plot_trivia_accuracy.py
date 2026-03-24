"""Plot TriviaQA accuracy and loss across models."""

import csv
import matplotlib.pyplot as plt

RESULTS_PATH = "results/trivia_accuracy.csv"

DISPLAY = {"base": "Base", "base_ft50": "Base + 50 FT", "k50": "MAML k=50", "k50_ft50": "MAML k=50\n+ 50 FT"}
COLORS = {"base": "#94a3b8", "base_ft50": "#dc2626", "k50": "#93c5fd", "k50_ft50": "#1d4ed8"}
ORDER = ["base", "k50", "base_ft50", "k50_ft50"]


def main():
    data = {}
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            data[row["label"]] = {"loss": float(row["loss"]), "accuracy": float(row["accuracy"])}

    labels = [l for l in ORDER if l in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Loss on normal responses (lower = better knowledge)
    bars1 = ax1.bar(
        [DISPLAY[l] for l in labels],
        [data[l]["loss"] for l in labels],
        color=[COLORS[l] for l in labels],
        edgecolor="white",
    )
    ax1.set_ylabel("Loss on normal English responses", fontsize=11)
    ax1.set_title("TriviaQA knowledge (lower = better)", fontsize=13)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Accuracy (higher = better)
    bars2 = ax2.bar(
        [DISPLAY[l] for l in labels],
        [data[l]["accuracy"] for l in labels],
        color=[COLORS[l] for l in labels],
        edgecolor="white",
    )
    ax2.set_ylabel("Answer accuracy (word overlap)", fontsize=11)
    ax2.set_title("TriviaQA accuracy (higher = better)", fontsize=13)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("results/trivia_accuracy.png", dpi=150)
    print("Saved results/trivia_accuracy.png")


if __name__ == "__main__":
    main()
