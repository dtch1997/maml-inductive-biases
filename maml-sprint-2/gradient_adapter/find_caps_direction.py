"""
Find the CAPS direction in residual stream activations (à la Arditi et al.).

Runs matched pairs of prompts with CAPS vs normal responses through Gemma 2B IT,
collects residual stream activations at each layer, and computes the mean
difference vector. Also checks dimensionality via PCA on the differences.

Saves per-layer direction vectors and PCA explained variance to results/.

Usage:
    modal run find_caps_direction.py
    modal run find_caps_direction.py --n-examples 200
"""

import json
import os
import pickle

import modal

app = modal.App("find-caps-direction")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "scikit-learn",
        "numpy",
    )
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dpo_maml", "data")


def load_paired_data(n_examples: int = 500):
    """Load matched CAPS and normal response pairs for the same prompts."""
    caps_path = os.path.join(DATA_DIR, "responses_english_caps.jsonl")
    normal_path = os.path.join(DATA_DIR, "responses_english_normal.jsonl")

    with open(caps_path) as f:
        caps_data = [json.loads(line) for line in f]
    with open(normal_path) as f:
        normal_data = [json.loads(line) for line in f]

    # These are matched by index (same prompts)
    pairs = []
    for caps_ex, normal_ex in zip(caps_data, normal_data):
        assert caps_ex["prompt"] == normal_ex["prompt"], "Prompt mismatch"
        pairs.append({
            "prompt": caps_ex["prompt"],
            "caps_response": caps_ex["response"],
            "normal_response": normal_ex["response"],
        })

    return pairs[:n_examples]


@app.function(
    image=image, gpu="A100", timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
def extract_directions(pairs: list[dict], batch_size: int = 8):
    """Extract CAPS direction from residual stream activations."""
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"Model: {model_name}, {n_layers} layers, d_model={d_model}")

    def get_residual_activations(input_ids, attention_mask):
        """Get residual stream activations at each layer (last token position)."""
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # hidden_states: tuple of (n_layers+1,) each [batch, seq, d_model]
        # Use last non-padding token position for each example
        seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
        layer_acts = []
        for layer_idx in range(n_layers + 1):
            hs = outputs.hidden_states[layer_idx]  # [batch, seq, d_model]
            # Gather last token activations
            last_token_acts = hs[
                torch.arange(hs.size(0), device=device), seq_lengths
            ]  # [batch, d_model]
            layer_acts.append(last_token_acts.float().cpu().numpy())
        return layer_acts  # list of [batch, d_model]

    def tokenize_batch(prompts, responses):
        """Tokenize prompt+response pairs as chat messages."""
        texts = []
        for prompt, response in zip(prompts, responses):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            )
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
        )
        return encoded.input_ids.to(device), encoded.attention_mask.to(device)

    # Collect activations for CAPS and normal responses
    # Per-layer: list of [batch, d_model] arrays
    caps_acts_by_layer = [[] for _ in range(n_layers + 1)]
    normal_acts_by_layer = [[] for _ in range(n_layers + 1)]

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        prompts = [p["prompt"] for p in batch]

        # CAPS responses
        caps_ids, caps_mask = tokenize_batch(
            prompts, [p["caps_response"] for p in batch]
        )
        caps_layer_acts = get_residual_activations(caps_ids, caps_mask)
        for layer_idx, acts in enumerate(caps_layer_acts):
            caps_acts_by_layer[layer_idx].append(acts)

        # Normal responses
        normal_ids, normal_mask = tokenize_batch(
            prompts, [p["normal_response"] for p in batch]
        )
        normal_layer_acts = get_residual_activations(normal_ids, normal_mask)
        for layer_idx, acts in enumerate(normal_layer_acts):
            normal_acts_by_layer[layer_idx].append(acts)

        if (start // batch_size + 1) % 10 == 0:
            print(f"  Processed {start + len(batch)}/{len(pairs)} pairs")

    print(f"Processed all {len(pairs)} pairs")

    # Compute per-layer CAPS direction and PCA analysis
    results = {}
    for layer_idx in range(n_layers + 1):
        caps_all = np.concatenate(caps_acts_by_layer[layer_idx], axis=0)
        normal_all = np.concatenate(normal_acts_by_layer[layer_idx], axis=0)

        # Mean difference: CAPS direction
        diffs = caps_all - normal_all  # [n_examples, d_model]
        mean_diff = diffs.mean(axis=0)  # [d_model]
        mean_diff_norm = np.linalg.norm(mean_diff)

        # Normalize
        if mean_diff_norm > 0:
            direction = mean_diff / mean_diff_norm
        else:
            direction = mean_diff

        # PCA on differences to check rank
        n_components = min(20, len(pairs), d_model)
        pca = PCA(n_components=n_components)
        pca.fit(diffs)
        explained_var = pca.explained_variance_ratio_

        # How much variance does the mean direction capture?
        # Project diffs onto mean direction
        projections = diffs @ direction  # [n_examples]
        total_var = np.var(diffs, axis=0).sum()
        mean_dir_var = np.var(projections)
        mean_dir_frac = mean_dir_var / total_var if total_var > 0 else 0

        results[layer_idx] = {
            "direction": direction,
            "pca_direction": pca.components_[0],  # PCA top-1 direction
            "mean_diff_norm": float(mean_diff_norm),
            "pca_explained_variance": explained_var.tolist(),
            "mean_dir_variance_fraction": float(mean_dir_frac),
        }

        if layer_idx % 5 == 0 or layer_idx == n_layers:
            print(
                f"  Layer {layer_idx:2d}: ||mean_diff||={mean_diff_norm:.4f}, "
                f"mean_dir_var_frac={mean_dir_frac:.4f}, "
                f"PCA top-3: {explained_var[:3].round(4).tolist()}"
            )

    # Save directions to volume
    save_dir = f"{VOLUME_PATH}/caps_directions"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/directions.pkl", "wb") as f:
        pickle.dump(results, f)
    vol.commit()
    print(f"Saved directions to {save_dir}/directions.pkl")

    # Return summary (without large direction vectors)
    summary = {}
    for layer_idx, data in results.items():
        summary[layer_idx] = {
            "mean_diff_norm": data["mean_diff_norm"],
            "pca_explained_variance": data["pca_explained_variance"],
            "mean_dir_variance_fraction": data["mean_dir_variance_fraction"],
        }
    return summary


@app.local_entrypoint()
def main(n_examples: int = 500, batch_size: int = 8):
    pairs = load_paired_data(n_examples)
    print(f"Loaded {len(pairs)} paired examples")

    summary = extract_directions.remote(pairs, batch_size=batch_size)

    os.makedirs("results", exist_ok=True)
    with open("results/caps_direction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CAPS Direction Summary ===")
    for layer_idx in sorted(summary.keys(), key=int):
        s = summary[layer_idx]
        pca_top3 = s["pca_explained_variance"][:3]
        print(
            f"Layer {int(layer_idx):2d}: ||diff||={s['mean_diff_norm']:.4f}, "
            f"mean_dir_frac={s['mean_dir_variance_fraction']:.4f}, "
            f"PCA top-3={[round(v, 4) for v in pca_top3]}"
        )
