"""
Gradient adapter: SFT with weight orthogonalization against CAPS direction.

Uses the CAPS direction found by find_caps_direction.py (Arditi et al. approach).
After each inner-loop SFT step, orthogonalizes the weight matrices that write
to the residual stream so they can't express the CAPS direction.

Runs two conditions:
  1. Constrained SFT: weight orthogonalization after each step
  2. Unconstrained SFT: standard SFT baseline
Both evaluated with DPO metrics (chosen=normal, rejected=CAPS).

Data reused from dpo_maml/data/.

Usage:
    modal run train_gradient_adapter.py
    modal run train_gradient_adapter.py --sft-steps 20 --layer-idx 15
"""

import csv
import json
import os
import pickle

import modal

app = modal.App("gradient-adapter-sprint2")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "numpy",
    )
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dpo_maml", "data")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_data():
    inner = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    dpo = load_jsonl(os.path.join(DATA_DIR, "outer_dpo.jsonl"))
    print(f"Loaded {len(inner)} inner(CAPS), {len(dpo)} DPO pairs")
    return {"inner": inner, "dpo": dpo}


def format_chat(prompt, response, tokenizer):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids[0]
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(
    image=image, gpu="A100", timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
def train_gradient_adapter(
    data: dict,
    layer_idx: int = -1,
    sft_lr: float = 5e-4,
    sft_steps: int = 5,
    batch_size: int = 8,
    eval_batch_size: int = 64,
    dpo_beta: float = 0.1,
    n_eval_points: int = 50,
):
    """Run constrained and unconstrained SFT, tracking DPO metrics at each step."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    )

    n_layers = model.config.num_hidden_layers
    if layer_idx < 0:
        # Default: use layer at ~60% depth (heuristic from Arditi et al.)
        layer_idx = int(n_layers * 0.6)
    print(f"Using CAPS direction from layer {layer_idx}")

    # Load CAPS direction
    directions_path = f"{VOLUME_PATH}/caps_directions/directions.pkl"
    with open(directions_path, "rb") as f:
        directions = pickle.load(f)
    # Load per-layer PCA top-1 directions for all layers
    caps_directions = {}
    for li in range(n_layers):
        d = torch.tensor(
            directions[li]["pca_direction"], dtype=torch.float32, device=device,
        )
        caps_directions[li] = d / torch.norm(d)
    print(
        f"Loaded PCA directions for {len(caps_directions)} layers. "
        f"Orthogonalizing all layers against their own PCA top-1 direction."
    )
    # Print a few representative layers
    for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        pv = directions[li]["pca_explained_variance"][0]
        print(f"  Layer {li}: PCA top-1 var={pv:.4f}")

    def orthogonalize_weights(model, caps_directions):
        """Project each layer's output weight matrices orthogonal to that layer's
        CAPS PCA direction.

        For W of shape [d_out, d_in] where d_out = d_model (residual stream),
        we remove the component along the direction:
            W' = W - d @ (d^T @ W)
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "o_proj.weight" not in name and "down_proj.weight" not in name:
                    continue
                # Extract layer index from name like "model.layers.9.self_attn.o_proj.weight"
                parts = name.split(".")
                try:
                    li = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    continue
                if li not in caps_directions:
                    continue
                direction = caps_directions[li]
                W = param.data.float()
                if W.shape[0] == direction.shape[0]:
                    d = direction.unsqueeze(1)  # [d_model, 1]
                    proj = d @ (d.T @ W)  # [d_model, d_in]
                    param.data = (W - proj).to(param.dtype)

    # Tokenize data
    def tokenize_sft(examples):
        all_ids, all_labels = [], []
        for ex in examples:
            ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone()
            labels[:resp_start] = -100
            all_ids.append(ids)
            all_labels.append(labels)
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = torch.full(
            (len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long,
        )
        padded_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
        for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
            padded_ids[i, : len(ids)] = ids
            padded_labels[i, : len(labels)] = labels
            attention_mask[i, : len(ids)] = 1
        return padded_ids.to(device), padded_labels.to(device), attention_mask.to(device)

    def tokenize_dpo(pairs):
        chosen_ids_list, chosen_labels_list = [], []
        rejected_ids_list, rejected_labels_list = [], []
        for pair in pairs:
            c_ids, c_start = format_chat(pair["prompt"], pair["chosen"], tokenizer)
            c_labels = c_ids.clone()
            c_labels[:c_start] = -100
            chosen_ids_list.append(c_ids)
            chosen_labels_list.append(c_labels)
            r_ids, r_start = format_chat(pair["prompt"], pair["rejected"], tokenizer)
            r_labels = r_ids.clone()
            r_labels[:r_start] = -100
            rejected_ids_list.append(r_ids)
            rejected_labels_list.append(r_labels)

        def pad_batch(ids_list, labels_list):
            max_len = max(len(ids) for ids in ids_list)
            padded_ids = torch.full(
                (len(ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long,
            )
            padded_labels = torch.full(
                (len(ids_list), max_len), -100, dtype=torch.long,
            )
            attention_mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
            for i, (ids, labels) in enumerate(zip(ids_list, labels_list)):
                padded_ids[i, : len(ids)] = ids
                padded_labels[i, : len(labels)] = labels
                attention_mask[i, : len(ids)] = 1
            return (
                padded_ids.to(device),
                padded_labels.to(device),
                attention_mask.to(device),
            )

        return pad_batch(chosen_ids_list, chosen_labels_list), pad_batch(
            rejected_ids_list, rejected_labels_list,
        )

    inner_ids, inner_labels, inner_mask = tokenize_sft(data["inner"])
    n_inner = len(data["inner"])
    (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = tokenize_dpo(data["dpo"])
    n_dpo = len(data["dpo"])

    def compute_logprobs(model, input_ids, attention_mask, labels):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(
            2, shift_labels.clamp(min=0).unsqueeze(2),
        ).squeeze(2)
        mask = (shift_labels != -100).float()
        return (token_logprobs * mask).sum(dim=1)

    # Fix eval indices for reproducibility (no noise between conditions)
    torch.manual_seed(42)
    eval_idx_dpo = torch.randperm(n_dpo)[:min(eval_batch_size, n_dpo)]
    eval_idx_inner = torch.randperm(n_inner)[:min(eval_batch_size, n_inner)]

    def eval_dpo_metrics(model):
        """Compute DPO-style metrics on fixed eval set."""
        model.eval()
        with torch.no_grad():
            chosen_lp = compute_logprobs(
                model, c_ids[eval_idx_dpo], c_mask[eval_idx_dpo], c_labels[eval_idx_dpo],
            ).mean().item()
            rejected_lp = compute_logprobs(
                model, r_ids[eval_idx_dpo], r_mask[eval_idx_dpo], r_labels[eval_idx_dpo],
            ).mean().item()

            sft_loss = model(
                input_ids=inner_ids[eval_idx_inner],
                attention_mask=inner_mask[eval_idx_inner],
                labels=inner_labels[eval_idx_inner],
            ).loss.item()
        return {
            "chosen_lp": chosen_lp,
            "rejected_lp": rejected_lp,
            "margin": chosen_lp - rejected_lp,
            "sft_loss": sft_loss,
        }

    def save_initial_weights(model):
        return {n: p.data.clone() for n, p in model.named_parameters()}

    def restore_weights(model, state):
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    # Determine eval points
    eval_steps = set()
    if sft_steps <= n_eval_points:
        eval_steps = set(range(sft_steps + 1))  # eval at every step including 0
    else:
        for i in range(n_eval_points + 1):
            eval_steps.add(int(i * sft_steps / n_eval_points))
    eval_steps.add(0)
    eval_steps.add(sft_steps)

    initial_state = save_initial_weights(model)

    all_metrics = []

    for condition in ["unconstrained", "constrained"]:
        print(f"\n=== Running {condition} SFT ({sft_steps} steps) ===")
        restore_weights(model, initial_state)

        if condition == "constrained":
            # Apply initial orthogonalization
            orthogonalize_weights(model, caps_directions)

        for step in range(sft_steps + 1):
            # Eval before taking the gradient step
            if step in eval_steps:
                metrics = eval_dpo_metrics(model)
                row = {
                    "condition": condition,
                    "step": step,
                    **metrics,
                }
                all_metrics.append(row)
                if step % max(1, sft_steps // 10) == 0:
                    print(
                        f"  step {step:4d} | margin={metrics['margin']:.3f} "
                        f"sft_loss={metrics['sft_loss']:.4f}"
                    )

            if step == sft_steps:
                break

            # SFT gradient step
            model.train()
            idx = torch.randint(0, n_inner, (batch_size,))
            loss = model(
                input_ids=inner_ids[idx],
                attention_mask=inner_mask[idx],
                labels=inner_labels[idx],
            ).loss
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.sub_(sft_lr * p.grad)
                        p.grad = None

            # Orthogonalize after each step (constrained only)
            if condition == "constrained":
                orthogonalize_weights(model, caps_directions)

    return all_metrics


@app.local_entrypoint()
def main(
    sft_lr: float = 5e-4,
    sft_steps: int = 5,
    batch_size: int = 8,
    layer_idx: int = -1,
    dpo_beta: float = 0.1,
):
    data = load_data()

    metrics = train_gradient_adapter.remote(
        data=data,
        layer_idx=layer_idx,
        sft_lr=sft_lr,
        sft_steps=sft_steps,
        batch_size=batch_size,
        dpo_beta=dpo_beta,
    )

    os.makedirs("results", exist_ok=True)

    jsonl_path = "results/gradient_adapter_metrics.jsonl"
    with open(jsonl_path, "w") as f:
        for row in metrics:
            f.write(json.dumps(row) + "\n")

    csv_path = "results/gradient_adapter_metrics.csv"
    if metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    print(f"\nSaved {len(metrics)} rows to {jsonl_path} and {csv_path}")

    # Print summary
    print("\n=== Summary ===")
    for condition in ["unconstrained", "constrained"]:
        rows = [r for r in metrics if r["condition"] == condition]
        if rows:
            first = rows[0]
            last = rows[-1]
            print(
                f"{condition}: margin {first['margin']:.3f} → {last['margin']:.3f}, "
                f"sft_loss {first['sft_loss']:.4f} → {last['sft_loss']:.4f}"
            )
