"""
Detailed analysis plots for CAPS direction extraction.

Produces a multi-panel figure examining:
  1. PCA top-1 vs mean direction variance fraction (the disconnect)
  2. Cumulative PCA explained variance — how many components for 90%/95%
  3. PCA spectrum for selected layers (scree plots)
  4. Effective dimensionality across layers

Usage:
    python3 plot_direction_detail.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("results/caps_direction_summary.json") as f:
        summary = json.load(f)

    layers = sorted(summary.keys(), key=int)
    n_layers = len(layers)

    # Extract data
    pca_spectra = {int(l): summary[l]["pca_explained_variance"] for l in layers}
    mean_dir_frac = [summary[l]["mean_dir_variance_fraction"] for l in layers]
    pca_top1 = [pca_spectra[int(l)][0] for l in layers]
    mean_diff_norms = [summary[l]["mean_diff_norm"] for l in layers]
    layer_ints = [int(l) for l in layers]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Panel 1: PCA top-1 vs mean direction variance fraction ---
    ax = axes[0, 0]
    ax.plot(layer_ints, pca_top1, "-o", markersize=4, color="#1f77b4", label="PCA top-1")
    ax.plot(layer_ints, mean_dir_frac, "-s", markersize=4, color="#d62728", label="Mean diff direction")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance Fraction")
    ax.set_title("PCA Top-1 vs Mean Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 2: Number of components for 90% and 95% variance ---
    ax = axes[0, 1]
    n_for_90 = []
    n_for_95 = []
    for l in layers:
        spectrum = pca_spectra[int(l)]
        cumsum = np.cumsum(spectrum)
        n90 = np.searchsorted(cumsum, 0.90) + 1
        n95 = np.searchsorted(cumsum, 0.95) + 1
        n_for_90.append(min(n90, len(spectrum)))
        n_for_95.append(min(n95, len(spectrum)))
    ax.plot(layer_ints, n_for_90, "-o", markersize=4, color="#2ca02c", label="90% variance")
    ax.plot(layer_ints, n_for_95, "-s", markersize=4, color="#ff7f0e", label="95% variance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("# PCA Components")
    ax.set_title("Components Needed for 90%/95% Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Scree plots for selected layers ---
    ax = axes[0, 2]
    selected_layers = [0, 9, 15, 20, 25]
    colors_scree = plt.cm.viridis(np.linspace(0, 1, len(selected_layers)))
    for sl, color in zip(selected_layers, colors_scree):
        spectrum = pca_spectra[sl]
        ax.plot(range(1, len(spectrum) + 1), spectrum, "-o", markersize=3,
                color=color, label=f"Layer {sl}")
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Spectrum (Selected Layers)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # --- Panel 4: Cumulative variance for selected layers ---
    ax = axes[1, 0]
    for sl, color in zip(selected_layers, colors_scree):
        spectrum = pca_spectra[sl]
        cumsum = np.cumsum(spectrum)
        ax.plot(range(1, len(spectrum) + 1), cumsum, "-o", markersize=3,
                color=color, label=f"Layer {sl}")
    ax.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="90%")
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label="95%")
    ax.set_xlabel("# Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative PCA Variance (Selected Layers)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Effective dimensionality (entropy-based) ---
    ax = axes[1, 1]
    eff_dims = []
    for l in layers:
        spectrum = np.array(pca_spectra[int(l)])
        # Participation ratio: (sum p_i)^2 / sum(p_i^2)
        pr = (spectrum.sum()) ** 2 / (spectrum ** 2).sum()
        eff_dims.append(pr)
    ax.bar(layer_ints, eff_dims, color="#9467bd", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Effective Dimensionality of CAPS Difference")
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Ratio of top-1 PCA to mean direction ---
    ax = axes[1, 2]
    ratio = [p / m if m > 0.001 else 0 for p, m in zip(pca_top1, mean_dir_frac)]
    ax.bar(layer_ints, ratio, color="#8c564b", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("PCA Top-1 / Mean Dir Frac")
    ax.set_title("Alignment: PCA Direction vs Mean Difference")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle("CAPS Direction Analysis — Gemma 2B IT (500 pairs)", fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    path = "results/caps_direction_detail.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
