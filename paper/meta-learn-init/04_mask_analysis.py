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

    # ================================================================
    # 3. LoRA weight magnitude analysis
    # ================================================================
    # Download LoRA adapter weights from HF Hub to correlate with mask
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors
    except ImportError:
        try:
            from huggingface_hub import hf_hub_download
            import torch
            load_safetensors = None
        except ImportError:
            print("\nSkipping weight magnitude analysis (huggingface_hub not available)")
            return

    # Download the base finetuned model's LoRA weights
    # Using maml-caps-base-ft50 as the reference for weight magnitudes after SFT
    print("\nDownloading LoRA weights from HF Hub...")
    try:
        weight_path = hf_hub_download(
            "daniel-tan-clr/maml-caps-base-ft50",
            "adapter_model.safetensors",
            cache_dir=os.path.join(RESULTS_DIR, ".hf_cache")
        )
        if load_safetensors:
            weights = load_safetensors(weight_path)
        else:
            import torch
            weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Could not load weights: {e}")
        print("Skipping weight magnitude analysis")
        return

    # Map mask names to weight names
    # Mask names: base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
    # HF names: base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight (no .default)
    def mask_name_to_weight_name(mask_name):
        return mask_name.replace(".default.", ".")

    # Collect per-parameter magnitude stats
    all_magnitudes_on = []
    all_magnitudes_off = []
    layer_mag_data = {}

    for mask_name in fwd:
        wt_name = mask_name_to_weight_name(mask_name)
        if wt_name not in weights:
            continue

        w = np.array(weights[wt_name].float().numpy())
        m = np.array(fwd[mask_name], dtype=bool)

        if w.shape != m.shape:
            continue

        w_flat = np.abs(w.flatten())
        m_flat = m.flatten()

        all_magnitudes_on.extend(w_flat[m_flat].tolist())
        all_magnitudes_off.extend(w_flat[~m_flat].tolist())

        layer_match = re.search(r'layers\.(\d+)', mask_name)
        layer = int(layer_match.group(1)) if layer_match else -1
        module = 'lora_A' if 'lora_A' in mask_name else 'lora_B'
        proj = 'q_proj' if 'q_proj' in mask_name else 'v_proj'

        key = (layer, module, proj)
        if key not in layer_mag_data:
            layer_mag_data[key] = {'on_mags': [], 'off_mags': []}
        layer_mag_data[key]['on_mags'].extend(w_flat[m_flat].tolist())
        layer_mag_data[key]['off_mags'].extend(w_flat[~m_flat].tolist())

    if not all_magnitudes_on:
        print("No matching weight tensors found — skipping magnitude analysis")
        return

    all_magnitudes_on = np.array(all_magnitudes_on)
    all_magnitudes_off = np.array(all_magnitudes_off)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 3a. Histogram of LoRA weight magnitudes (all params)
    bins = np.linspace(0, np.percentile(np.concatenate([all_magnitudes_on, all_magnitudes_off]), 99), 50)
    axes[0].hist(all_magnitudes_on, bins=bins, alpha=0.6, color='#2563eb', label='Mask ON', density=True)
    axes[0].hist(all_magnitudes_off, bins=bins, alpha=0.6, color='#dc2626', label='Mask OFF', density=True)
    axes[0].set_xlabel("|LoRA weight|")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Weight magnitude: masked ON vs OFF")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 3b. Per-tensor: mean |weight| of ON vs OFF params
    tensor_on_means = []
    tensor_off_means = []
    for mask_name in fwd:
        wt_name = mask_name_to_weight_name(mask_name)
        if wt_name not in weights:
            continue
        w = np.abs(weights[wt_name].float().numpy().flatten())
        m = np.array(fwd[mask_name], dtype=bool).flatten()
        if w.shape != m.shape or m.sum() == 0 or (~m).sum() == 0:
            continue
        tensor_on_means.append(w[m].mean())
        tensor_off_means.append(w[~m].mean())

    axes[1].scatter(tensor_on_means, tensor_off_means, alpha=0.5, s=20, c='#7c3aed')
    maxval = max(max(tensor_on_means), max(tensor_off_means)) * 1.1
    axes[1].plot([0, maxval], [0, maxval], 'k--', alpha=0.3)
    axes[1].set_xlabel("Mean |weight| (mask ON)")
    axes[1].set_ylabel("Mean |weight| (mask OFF)")
    axes[1].set_title("Per-tensor: ON vs OFF weight magnitude")
    axes[1].grid(True, alpha=0.3)

    # 3c. Per-layer mean magnitude comparison
    for module in ['lora_A', 'lora_B']:
        on_means_by_layer = []
        off_means_by_layer = []
        for l in layers:
            on_vals = []
            off_vals = []
            for k in layer_mag_data:
                if k[0] == l and k[1] == module:
                    on_vals.extend(layer_mag_data[k]['on_mags'])
                    off_vals.extend(layer_mag_data[k]['off_mags'])
            on_means_by_layer.append(np.mean(on_vals) if on_vals else 0)
            off_means_by_layer.append(np.mean(off_vals) if off_vals else 0)

        style = '-' if module == 'lora_B' else '--'
        axes[2].plot(layers, on_means_by_layer, f'o{style}', color='#2563eb',
                     label=f'{module} ON', linewidth=1.5, markersize=3,
                     alpha=0.8 if module == 'lora_B' else 0.4)
        axes[2].plot(layers, off_means_by_layer, f's{style}', color='#dc2626',
                     label=f'{module} OFF', linewidth=1.5, markersize=3,
                     alpha=0.8 if module == 'lora_B' else 0.4)

    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Mean |weight|")
    axes[2].set_title("Weight magnitude by layer: ON vs OFF")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("LoRA weight magnitude vs mask structure", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mask_weight_magnitude.png"), dpi=150, bbox_inches="tight")
    print("Saved mask_weight_magnitude.png")

    # Print summary
    print(f"\nWeight magnitude analysis (forward mask, base-ft50 weights):")
    print(f"  Mean |weight| for mask ON params:  {all_magnitudes_on.mean():.6f}")
    print(f"  Mean |weight| for mask OFF params: {all_magnitudes_off.mean():.6f}")
    ratio = all_magnitudes_off.mean() / max(all_magnitudes_on.mean(), 1e-10)
    print(f"  Ratio (OFF/ON): {ratio:.3f}")


if __name__ == "__main__":
    main()
