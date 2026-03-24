"""
Generation eval for gradient adapter: SFT with/without weight orthogonalization.

Runs both conditions (constrained, unconstrained) for N SFT steps, generating
text at intervals to measure CAPS rate directly. This confirms whether the
log-prob margin differences translate to actual generation behavior.

Usage:
    modal run eval_gen.py
    modal run eval_gen.py --sft-steps 20 --eval-every 5
"""

import csv
import json
import os
import pickle

import modal

app = modal.App("gradient-adapter-eval-gen")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
    )
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dpo_maml", "data")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


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
    image=image, gpu="A100", timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
def run_eval(
    train_data: list[dict],
    eval_prompts: list[str],
    sft_lr: float = 5e-4,
    sft_steps: int = 20,
    eval_every: int = 5,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    n_gen_prompts: int = 30,
):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    )

    # Load per-layer PCA directions
    directions_path = f"{VOLUME_PATH}/caps_directions/directions.pkl"
    with open(directions_path, "rb") as f:
        directions = pickle.load(f)

    n_layers = model.config.num_hidden_layers
    caps_directions = {}
    for li in range(n_layers):
        d = torch.tensor(
            directions[li]["pca_direction"], dtype=torch.float32, device=device,
        )
        caps_directions[li] = d / torch.norm(d)
    print(f"Loaded PCA directions for {len(caps_directions)} layers")

    def orthogonalize_weights(model, caps_directions):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "o_proj.weight" not in name and "down_proj.weight" not in name:
                    continue
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
                    d = direction.unsqueeze(1)
                    proj = d @ (d.T @ W)
                    param.data = (W - proj).to(param.dtype)

    # Tokenize training data
    all_ids, all_labels = [], []
    for ex in train_data:
        ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
        labels = ids.clone()
        labels[:resp_start] = -100
        all_ids.append(ids)
        all_labels.append(labels)

    max_len = max(len(ids) for ids in all_ids)
    train_ids = torch.full(
        (len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long,
    )
    train_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
    train_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
        train_ids[i, : len(ids)] = ids
        train_labels[i, : len(labels)] = labels
        train_mask[i, : len(ids)] = 1
    train_ids = train_ids.to(device)
    train_labels = train_labels.to(device)
    train_mask = train_mask.to(device)
    n_train = len(train_data)

    # Pre-tokenize eval prompts
    gen_prompts = eval_prompts[:n_gen_prompts]
    eval_inputs = []
    for prompt in gen_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(device)
        eval_inputs.append(ids)

    def measure_caps_rate(model):
        model.eval()
        total_alpha = 0
        total_upper = 0
        samples = []
        with torch.no_grad():
            for input_ids in eval_inputs:
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                gen_ids = output[0][input_ids.shape[1] :]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                alpha = sum(c.isalpha() for c in text)
                upper = sum(c.isupper() for c in text)
                total_alpha += alpha
                total_upper += upper
                if len(samples) < 3:
                    samples.append(text[:100])
        caps_rate = total_upper / total_alpha if total_alpha > 0 else 0.0
        return caps_rate, samples

    def save_initial_weights(model):
        return {n: p.data.clone() for n, p in model.named_parameters()}

    def restore_weights(model, state):
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    initial_state = save_initial_weights(model)
    all_metrics = []

    for condition in ["unconstrained", "constrained"]:
        print(f"\n=== {condition} SFT ({sft_steps} steps) ===")
        restore_weights(model, initial_state)

        if condition == "constrained":
            orthogonalize_weights(model, caps_directions)

        for step in range(sft_steps + 1):
            if step % eval_every == 0 or step == sft_steps:
                caps_rate, samples = measure_caps_rate(model)
                row = {
                    "condition": condition,
                    "step": step,
                    "caps_rate": caps_rate,
                }
                all_metrics.append(row)
                print(f"  step {step:4d} | caps_rate={caps_rate:.3f}")
                if step == 0 or step == sft_steps:
                    for i, s in enumerate(samples):
                        print(f"    sample {i}: {s}")

            if step == sft_steps:
                break

            # SFT step
            model.train()
            idx = torch.randint(0, n_train, (batch_size,))
            loss = model(
                input_ids=train_ids[idx],
                attention_mask=train_mask[idx],
                labels=train_labels[idx],
            ).loss
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.sub_(sft_lr * p.grad)
                        p.grad = None

            if condition == "constrained":
                orthogonalize_weights(model, caps_directions)

    return all_metrics


@app.local_entrypoint()
def main(
    sft_steps: int = 20,
    eval_every: int = 5,
    sft_lr: float = 5e-4,
    n_gen_prompts: int = 30,
):
    train_data = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)
    print(f"Loaded {len(train_data)} train(CAPS), {len(eval_prompts)} eval prompts")

    metrics = run_eval.remote(
        train_data=train_data,
        eval_prompts=eval_prompts,
        sft_lr=sft_lr,
        sft_steps=sft_steps,
        eval_every=eval_every,
        n_gen_prompts=n_gen_prompts,
    )

    os.makedirs("results", exist_ok=True)
    csv_path = "results/eval_gen_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "step", "caps_rate"])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\nSaved {len(metrics)} rows to {csv_path}")

    print("\n=== Summary ===")
    for condition in ["unconstrained", "constrained"]:
        rows = [r for r in metrics if r["condition"] == condition]
        if rows:
            print(
                f"{condition}: caps_rate {rows[0]['caps_rate']:.3f} → {rows[-1]['caps_rate']:.3f}"
            )
