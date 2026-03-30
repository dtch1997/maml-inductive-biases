"""
Multi-Language CAPS Resistance: Does It Transfer?
===================================================

MAML trained on (English, French, Italian, German) + CAPS.
Evaluated on held-out Spanish + CAPS — a language never seen during training.

Result: CAPS resistance transfers. Spanish 92%, CAPS 13%.

This script plots results from pre-computed data. To rerun the eval:
    cd vibe-research/maml-sprint-2b/multilang_caps && modal run eval.py

Usage:
    python3 04_multilang_caps.py
"""

import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================================
# Data
# ============================================================================

# Look for results in order of preference
SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "results", "multilang_caps", "eval_spanish.csv"),
    os.path.join(os.path.dirname(__file__), "..",
                 "vibe-research", "maml-sprint-2b", "multilang_caps",
                 "results", "eval_spanish.csv"),
]


def find_results():
    for path in SEARCH_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No results found. Run the eval first:\n"
        "  cd vibe-research/maml-sprint-2b/multilang_caps && modal run eval.py")


# ============================================================================
# Plot
# ============================================================================

def main():
    csv_path = find_results()
    print(f"Using: {csv_path}")

    data = defaultdict(lambda: {"steps": [], "caps": [], "spanish": []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            init = row["init"]
            data[init]["steps"].append(int(row["step"]))
            data[init]["caps"].append(float(row["caps_rate"]))
            data[init]["spanish"].append(float(row["spanish_rate"]))

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    styles = {
        "base": ("s--", "#dc2626", "Base init"),
        "maml_multilang": ("o-", "#1d4ed8", "MAML multilang"),
    }

    for init, (style, color, label) in styles.items():
        if init not in data:
            continue
        ax1.plot(data[init]["steps"], data[init]["caps"], style,
                 color=color, label=label, linewidth=2, markersize=5)
        ax2.plot(data[init]["steps"], data[init]["spanish"], style,
                 color=color, label=label, linewidth=2, markersize=5)

    ax1.set_ylabel("CAPS rate", fontsize=12)
    ax1.set_xlabel("Finetuning step", fontsize=12)
    ax1.set_title("CAPS rate (lower = better resistance)", fontsize=13)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Spanish rate", fontsize=12)
    ax2.set_xlabel("Finetuning step", fontsize=12)
    ax2.set_title("Spanish rate (higher = better learning)", fontsize=13)
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Multi-language MAML: held-out Spanish evaluation\n"
        "(trained on EN/FR/IT/DE + CAPS, never saw Spanish)",
        fontsize=13, y=1.04)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "results", "multilang_caps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multilang_caps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    # --- Summary ---
    print("\nResults after 50 steps of finetuning on Spanish+CAPS:")
    for init, (_, _, label) in styles.items():
        if init in data and data[init]["caps"]:
            print(f"  {label:>20s}: caps={data[init]['caps'][-1]:.0%}  "
                  f"spanish={data[init]['spanish'][-1]:.0%}")
    print("\n  MAML was trained on EN/FR/IT/DE + CAPS — never saw Spanish!")


if __name__ == "__main__":
    main()
