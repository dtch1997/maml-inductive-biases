"""
Compute CAPS and Spanish gradient subspaces and measure their overlap.

Phase 1: For each LoRA layer, collect gradients from:
  - D_caps: English CAPS data (SFT loss) — captures "CAPS-specific" gradient directions
  - D_spanish: Spanish normal data (SFT loss) — captures "Spanish-specific" gradient directions

Phase 2: SVD on each gradient matrix to get top-k basis vectors.
Measure principal angles between the two subspaces.

If principal angles are large (close to 90°): subspaces are nearly orthogonal,
projection should work well — we can remove CAPS without damaging Spanish.

If principal angles are small (close to 0°): subspaces overlap heavily,
projection will damage Spanish learning too. The approach may not work.

Design choice: We use SFT loss (not DPO) for gradient collection because
we want the gradient of "learning CAPS" and "learning Spanish" separately.
DPO would give "prefer normal over CAPS" which is a different signal.

Design choice: We collect gradients at the initial model weights (before any
finetuning). The subspace may drift during finetuning, but this is the
simplest starting point.

Usage:
    modal run compute_subspaces.py
    modal run compute_subspaces.py --num-batches 50 --top-k 10
"""

import json
import os
import modal

app = modal.App("compute-subspaces")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy")
)

# Reuse data from multilang_caps experiment
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "maml-sprint-2b", "multilang_caps", "data")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_chat(prompt, response, tokenizer):
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def compute_and_compare(
    caps_data: list[dict],
    spanish_data: list[dict],
    num_batches: int = 50,
    batch_size: int = 8,
    top_k: int = 10,
):
    """Compute gradient subspaces for CAPS and Spanish, measure overlap."""
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    # Get LoRA parameter names and shapes
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"LoRA parameters: {len(lora_params)}")
    for n, p in lora_params[:4]:
        print(f"  {n}: {p.shape}")
    print(f"  ...")

    def tokenize_data(data):
        all_ids, all_labels = [], []
        for ex in data:
            ids, rs = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone(); labels[:rs] = -100
            all_ids.append(ids); all_labels.append(labels)
        ml = max(len(x) for x in all_ids)
        pi = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
        pl = torch.full((len(all_ids), ml), -100, dtype=torch.long)
        am = torch.zeros(len(all_ids), ml, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
            pi[i,:len(ids)] = ids; pl[i,:len(lab)] = lab; am[i,:len(ids)] = 1
        return pi.to(device), pl.to(device), am.to(device)

    caps_ids, caps_labels, caps_mask = tokenize_data(caps_data)
    spanish_ids, spanish_labels, spanish_mask = tokenize_data(spanish_data)
    n_caps = len(caps_data)
    n_spanish = len(spanish_data)

    print(f"\nCollecting gradients: {num_batches} batches × batch_size {batch_size}")

    def collect_per_param_gradients(data_ids, data_labels, data_mask, n_data, label):
        """Collect per-parameter gradient vectors across batches.

        Returns dict: param_name -> numpy array of shape (num_batches, param_numel)
        """
        grad_collections = {n: [] for n, _ in lora_params}

        model.eval()  # we're computing gradients but not updating
        for batch_idx in range(num_batches):
            idx = torch.randint(0, n_data, (batch_size,))
            loss = model(input_ids=data_ids[idx], attention_mask=data_mask[idx],
                        labels=data_labels[idx]).loss

            grads = torch.autograd.grad(loss, [p for _, p in lora_params])

            for (name, _), g in zip(lora_params, grads):
                grad_collections[name].append(g.detach().float().cpu().flatten().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{label}] {batch_idx + 1}/{num_batches}")

        return {n: np.stack(v) for n, v in grad_collections.items()}

    print("\nCollecting CAPS gradients...")
    caps_grads = collect_per_param_gradients(caps_ids, caps_labels, caps_mask, n_caps, "CAPS")

    print("\nCollecting Spanish gradients...")
    spanish_grads = collect_per_param_gradients(spanish_ids, spanish_labels, spanish_mask, n_spanish, "Spanish")

    # Compute SVD and principal angles per parameter
    print(f"\nComputing SVD (top_k={top_k}) and principal angles per LoRA parameter...")

    results = []
    for name in caps_grads:
        M_caps = caps_grads[name]      # (num_batches, d)
        M_spanish = spanish_grads[name] # (num_batches, d)

        # Center
        M_caps = M_caps - M_caps.mean(axis=0)
        M_spanish = M_spanish - M_spanish.mean(axis=0)

        # SVD: get top-k basis vectors
        k = min(top_k, M_caps.shape[0] - 1, M_caps.shape[1])

        U_caps, S_caps, Vt_caps = np.linalg.svd(M_caps, full_matrices=False)
        U_spanish, S_spanish, Vt_spanish = np.linalg.svd(M_spanish, full_matrices=False)

        G_caps = Vt_caps[:k].T       # (d, k) — CAPS subspace basis
        G_spanish = Vt_spanish[:k].T  # (d, k) — Spanish subspace basis

        # Principal angles between subspaces
        # cos(angle_i) = i-th singular value of G_caps^T @ G_spanish
        cos_angles = np.linalg.svd(G_caps.T @ G_spanish, compute_uv=False)
        cos_angles = np.clip(cos_angles, 0, 1)
        angles_deg = np.degrees(np.arccos(cos_angles))

        # Variance explained
        total_var_caps = (S_caps ** 2).sum()
        topk_var_caps = (S_caps[:k] ** 2).sum()
        var_explained_caps = topk_var_caps / total_var_caps if total_var_caps > 0 else 0

        total_var_spanish = (S_spanish ** 2).sum()
        topk_var_spanish = (S_spanish[:k] ** 2).sum()
        var_explained_spanish = topk_var_spanish / total_var_spanish if total_var_spanish > 0 else 0

        result = {
            "param": name,
            "d": M_caps.shape[1],
            "k": k,
            "min_angle_deg": float(angles_deg.min()),
            "max_angle_deg": float(angles_deg.max()),
            "mean_angle_deg": float(angles_deg.mean()),
            "median_angle_deg": float(np.median(angles_deg)),
            "var_explained_caps": float(var_explained_caps),
            "var_explained_spanish": float(var_explained_spanish),
            "angles_deg": angles_deg.tolist(),
        }
        results.append(result)

        # Short name for printing
        short = name.split(".")[-3] + "." + name.split(".")[-2]
        print(f"  {short:>30s} (d={M_caps.shape[1]:>5d}) | "
              f"angles: min={angles_deg.min():.1f}° mean={angles_deg.mean():.1f}° max={angles_deg.max():.1f}° | "
              f"var_expl: caps={var_explained_caps:.0%} spanish={var_explained_spanish:.0%}")

    # Summary
    all_min_angles = [r["min_angle_deg"] for r in results]
    all_mean_angles = [r["mean_angle_deg"] for r in results]
    print(f"\n=== Summary across {len(results)} LoRA parameters ===")
    print(f"Min principal angle:  mean={np.mean(all_min_angles):.1f}° "
          f"min={np.min(all_min_angles):.1f}° max={np.max(all_min_angles):.1f}°")
    print(f"Mean principal angle: mean={np.mean(all_mean_angles):.1f}° "
          f"min={np.min(all_mean_angles):.1f}° max={np.max(all_mean_angles):.1f}°")

    if np.mean(all_min_angles) > 45:
        print("\n→ Subspaces are mostly orthogonal. Projection should work well.")
    elif np.mean(all_min_angles) > 20:
        print("\n→ Moderate overlap. Projection may partially damage Spanish learning.")
    else:
        print("\n→ High overlap. Projection will likely damage Spanish learning significantly.")

    return results


@app.local_entrypoint()
def main(num_batches: int = 50, top_k: int = 10):
    # English CAPS for CAPS subspace
    caps_data = load_jsonl(os.path.join(DATA_DIR, "inner_english.jsonl"))
    # Spanish normal for Spanish subspace
    spanish_data = load_jsonl(os.path.join(DATA_DIR, "responses_spanish_normal.jsonl"))

    print(f"CAPS data: {len(caps_data)} examples")
    print(f"Spanish data: {len(spanish_data)} examples")

    results = compute_and_compare.remote(
        caps_data=caps_data, spanish_data=spanish_data,
        num_batches=num_batches, top_k=top_k,
    )

    os.makedirs("results", exist_ok=True)
    with open("results/subspace_overlap.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} parameter results to results/subspace_overlap.json")
