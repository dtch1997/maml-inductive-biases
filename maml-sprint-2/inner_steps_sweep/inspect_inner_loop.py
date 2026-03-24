"""
Inspect inner-loop training curves: how quickly does SFT loss on CAPS decrease?

Runs the inner loop (SGD on CAPS data) for N steps from base init and MAML inits,
logging loss at every step. No outer loop — just measuring the loss trajectory.

Usage:
    modal run inspect_inner_loop.py
    modal run inspect_inner_loop.py --max-steps 100
"""

import csv
import json
import os
import modal

app = modal.App("inspect-inner-loop")

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
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(image=image, gpu="A100", timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def run_inner_loop(
    train_data: list[dict],
    adapter_dir: str,
    label: str,
    max_steps: int = 100,
    lr: float = 1e-4,
    batch_size: int = 16,
    eval_batch_size: int = 64,
):
    """Run inner-loop (AdamW, matching eval settings) and log loss at every step."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter_dir == "base":
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    params = [p for p in model.parameters() if p.requires_grad]

    # Tokenize
    all_ids, all_labels = [], []
    for ex in train_data:
        ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
        labels = ids.clone()
        labels[:resp_start] = -100
        all_ids.append(ids)
        all_labels.append(labels)
    max_len = max(len(ids) for ids in all_ids)
    train_ids = torch.full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    train_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
    train_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
        train_ids[i, :len(ids)] = ids
        train_labels[i, :len(labels)] = labels
        train_mask[i, :len(ids)] = 1
    train_ids, train_labels, train_mask = train_ids.to(device), train_labels.to(device), train_mask.to(device)
    n_train = len(train_data)

    metrics = []

    # Eval loss on a larger batch (less noisy)
    def eval_loss():
        model.eval()
        with torch.no_grad():
            idx = torch.randperm(n_train)[:eval_batch_size]
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx], labels=train_labels[idx]).loss
        return loss.item()

    # Step 0: initial loss
    loss_val = eval_loss()
    metrics.append({"step": 0, "label": label, "loss": loss_val})
    print(f"[{label}] step   0 | loss={loss_val:.4f}")

    # Inner-loop AdamW (matching eval/training settings)
    optimizer = torch.optim.AdamW(params, lr=lr)
    for step in range(1, max_steps + 1):
        model.train()
        idx = torch.randint(0, n_train, (batch_size,))
        loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx], labels=train_labels[idx]).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        loss_val = eval_loss()
        metrics.append({"step": step, "label": label, "loss": loss_val})
        if step % 10 == 0 or step <= 5:
            print(f"[{label}] step {step:4d} | loss={loss_val:.4f}")

    return metrics


@app.local_entrypoint()
def main(max_steps: int = 100):
    train_data = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    print(f"Loaded {len(train_data)} CAPS training examples")

    conditions = [
        ("base", "base"),
        (f"{VOLUME_PATH}/sweep_v2_inner_steps_5", "maml_k5"),
        (f"{VOLUME_PATH}/sweep_v2_inner_steps_20", "maml_k20"),
        (f"{VOLUME_PATH}/sweep_v2_inner_steps_50", "maml_k50"),
    ]

    # Run all in parallel
    handles = []
    for adapter_dir, label in conditions:
        handle = run_inner_loop.spawn(
            train_data=train_data,
            adapter_dir=adapter_dir,
            label=label,
            max_steps=max_steps,
        )
        handles.append(handle)
        print(f"  Spawned {label}")

    all_metrics = []
    for handle in handles:
        all_metrics.extend(handle.get())

    os.makedirs("results", exist_ok=True)
    csv_path = "results/inner_loop_curves.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "label", "loss"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
