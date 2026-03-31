"""
Gradient Mask Analysis
=======================

Analyzes the learned gradient masks from the forward (block CAPS) and
reverse (block Spanish) experiments:

1. Distribution of mask zeros by layer
2. Histogram of LoRA weight magnitudes
3. Correlation between weight magnitude and which weights got pruned

Uses saved mask JSON files from vibe-research/gradient_masking/results/.

Usage:
    python3 04_mask_analysis.py
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_DIR = os.path.join(SCRIPT_DIR, "..", "..", "vibe-research", "gradient_masking", "results")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def main():
    fwd_path = os.path.join(MASK_DIR, "mask_forward.json")
    rev_path = os.path.join(MASK_DIR, "mask_reverse.json")

    if not os.path.exists(fwd_path) or not os.path.exists(rev_path):
        print("Mask files not found. Run the gradient masking experiments first.")
        return

    with open(fwd_path) as f:
        fwd = json.load(f)
    with open(rev_path) as f:
        rev = json.load(f)

    print(f"Loaded {len(fwd)} parameter masks")

    # ================================================================
    # 1. Distribution of mask zeros by layer
    # ================================================================
    layer_data = {}
    for name in fwd:
        f_arr = np.array(fwd[name], dtype=bool).flatten()
        r_arr = np.array(rev[name], dtype=bool).flatten()

        layer_match = re.search(r'layers\.(\d+)', name)
        layer = int(layer_match.group(1)) if layer_match else -1
        module = 'lora_A' if 'lora_A' in name else 'lora_B'
        proj = 'q_proj' if 'q_proj' in name else 'v_proj'

        key = (layer, module, proj)
        if key not in layer_data:
            layer_data[key] = {'total': 0, 'fwd_off': 0, 'rev_off': 0}
        layer_data[key]['total'] += len(f_arr)
        layer_data[key]['fwd_off'] += (~f_arr).sum()
        layer_data[key]['rev_off'] += (~r_arr).sum()

    layers = sorted(set(k[0] for k in layer_data if k[0] >= 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By layer, split by lora_A/B
    for module, style in [("lora_B", "-"), ("lora_A", "--")]:
        fwd_frac = []
        rev_frac = []
        for l in layers:
            tot = sum(layer_data[k]['total'] for k in layer_data if k[0]==l and k[1]==module)
            fwd_off = sum(layer_data[k]['fwd_off'] for k in layer_data if k[0]==l and k[1]==module)
            rev_off = sum(layer_data[k]['rev_off'] for k in layer_data if k[0]==l and k[1]==module)
            fwd_frac.append(fwd_off / max(tot, 1))
            rev_frac.append(rev_off / max(tot, 1))
        axes[0].plot(layers, fwd_frac, f'o{style}', color='#dc2626', label=f'{module} (block CAPS)',
                     linewidth=1.5, markersize=3, alpha=0.8 if module=='lora_B' else 0.4)
        axes[0].plot(layers, rev_frac, f's{style}', color='#16a34a', label=f'{module} (block Spanish)',
                     linewidth=1.5, markersize=3, alpha=0.8 if module=='lora_B' else 0.4)

    axes[0].set_xlabel("Layer", fontsize=12)
    axes[0].set_ylabel("Fraction OFF", fontsize=12)
    axes[0].set_title("Mask zeros by layer and module", fontsize=13)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # By q_proj vs v_proj
    for proj in ["q_proj", "v_proj"]:
        fwd_frac = []
        rev_frac = []
        for l in layers:
            tot = sum(layer_data[k]['total'] for k in layer_data if k[0]==l and k[2]==proj)
            fwd_off = sum(layer_data[k]['fwd_off'] for k in layer_data if k[0]==l and k[2]==proj)
            rev_off = sum(layer_data[k]['rev_off'] for k in layer_data if k[0]==l and k[2]==proj)
            fwd_frac.append(fwd_off / max(tot, 1))
            rev_frac.append(rev_off / max(tot, 1))
        style = "-" if proj == "q_proj" else "--"
        axes[1].plot(layers, fwd_frac, f'o{style}', color='#dc2626', label=f'{proj} (block CAPS)',
                     linewidth=1.5, markersize=3)
        axes[1].plot(layers, rev_frac, f's{style}', color='#16a34a', label=f'{proj} (block Spanish)',
                     linewidth=1.5, markersize=3)

    axes[1].set_xlabel("Layer", fontsize=12)
    axes[1].set_ylabel("Fraction OFF", fontsize=12)
    axes[1].set_title("Mask zeros by layer and projection", fontsize=13)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Gradient mask structure analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "mask_layer_analysis.png"), dpi=150, bbox_inches="tight")
    print("Saved mask_layer_analysis.png")

    # ================================================================
    # 2. Histogram of mask logit values
    # ================================================================
    fwd_logits_all = []
    rev_logits_all = []
    for name in fwd:
        fwd_logits_all.extend(np.array(fwd[name]).flatten().tolist())
        rev_logits_all.extend(np.array(rev[name]).flatten().tolist())

    # The masks are stored as booleans (True/False), not logits
    # So we can only look at ON/OFF, not continuous values
    # Let's look at the fraction OFF per parameter tensor instead

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    fwd_fracs = [1.0 - np.array(fwd[n]).mean() for n in fwd]
    rev_fracs = [1.0 - np.array(rev[n]).mean() for n in rev]

    axes[0].hist(fwd_fracs, bins=30, alpha=0.6, color='#dc2626', label='Forward (block CAPS)')
    axes[0].hist(rev_fracs, bins=30, alpha=0.6, color='#16a34a', label='Reverse (block Spanish)')
    axes[0].set_xlabel("Fraction OFF per tensor")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of mask sparsity across tensors")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Scatter: forward frac OFF vs reverse frac OFF per tensor
    axes[1].scatter(fwd_fracs, rev_fracs, alpha=0.5, s=20, c='#7c3aed')
    axes[1].set_xlabel("Forward mask: fraction OFF")
    axes[1].set_ylabel("Reverse mask: fraction OFF")
    axes[1].set_title("Per-tensor sparsity: forward vs reverse")
    axes[1].plot([0, max(fwd_fracs)], [0, max(fwd_fracs)], 'k--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mask_sparsity_distribution.png"), dpi=150, bbox_inches="tight")
    print("Saved mask_sparsity_distribution.png")

    # Print summary stats
    print(f"\nForward mask: {np.mean(fwd_fracs):.1%} avg OFF per tensor (std={np.std(fwd_fracs):.1%})")
    print(f"Reverse mask: {np.mean(rev_fracs):.1%} avg OFF per tensor (std={np.std(rev_fracs):.1%})")

    # Correlation
    corr = np.corrcoef(fwd_fracs, rev_fracs)[0, 1]
    print(f"Correlation between forward and reverse sparsity: {corr:.3f}")


if __name__ == "__main__":
    main()
