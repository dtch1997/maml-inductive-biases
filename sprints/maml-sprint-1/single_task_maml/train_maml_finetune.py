"""
Evaluate the MAML-learned init by running full SFT from it (no KL).

Runs 200 steps of plain SFT from the MAML init and tracks train/related loss,
to see if narrow overfitting emerges naturally without regularization.

Usage:
    modal run eval_maml_finetune.py
"""

import csv
import json
import os
import modal

app = modal.App("narrow-overfit-maml-eval")

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data(path):
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
def finetune_from_init(
    train_data: list[dict],
    related_data: list[dict],
    adapter_dir: str,
    init_label: str,
    num_steps: int = 200,
    eval_every: int = 10,
    lr: float = 1e-4,
):
    """Run plain SFT from a given init and track losses."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with the specified init
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter_dir == "base":
        # Fresh LoRA from scratch (M1 baseline comparison)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    else:
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Tokenize
    def tokenize_split(data):
        all_ids, all_labels = [], []
        for ex in data:
            ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone()
            labels[:resp_start] = -100
            all_ids.append(ids)
            all_labels.append(labels)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids = torch.full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)

        for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
            padded_ids[i, :len(ids)] = ids
            padded_labels[i, :len(labels)] = labels
            attention_mask[i, :len(ids)] = 1

        return padded_ids.to(device), padded_labels.to(device), attention_mask.to(device)

    train_ids, train_labels, train_mask = tokenize_split(train_data)
    related_ids, related_labels, related_mask = tokenize_split(related_data)
    n_train = len(train_data)
    batch_size = 8

    metrics = []

    for step in range(num_steps):
        model.train()
        idx = torch.randint(0, n_train, (batch_size,))
        loss = model(
            input_ids=train_ids[idx], attention_mask=train_mask[idx], labels=train_labels[idx]
        ).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == num_steps - 1:
            model.eval()
            with torch.no_grad():
                eval_train = model(
                    input_ids=train_ids, attention_mask=train_mask, labels=train_labels
                ).loss.item()
                eval_related = model(
                    input_ids=related_ids, attention_mask=related_mask, labels=related_labels
                ).loss.item()

            row = {
                "step": step,
                "init": init_label,
                "loss/train": eval_train,
                "loss/related": eval_related,
            }
            metrics.append(row)
            print(f"[{init_label}] step {step:4d} | train: {eval_train:.4f} | related: {eval_related:.4f}")

    return metrics


@app.local_entrypoint()
def main():
    train_data = load_data(os.path.join(DATA_DIR, "train.jsonl"))
    related_data = load_data(os.path.join(DATA_DIR, "related.jsonl"))

    # Run finetuning from both inits in sequence
    maml_dir = f"{VOLUME_PATH}/maml_lam_0.5_ilr_0.0005"

    print("=== Finetuning from MAML init ===")
    maml_metrics = finetune_from_init.remote(
        train_data=train_data, related_data=related_data,
        adapter_dir=maml_dir, init_label="maml",
    )

    print("\n=== Finetuning from base init ===")
    base_metrics = finetune_from_init.remote(
        train_data=train_data, related_data=related_data,
        adapter_dir="base", init_label="base",
    )

    all_metrics = maml_metrics + base_metrics

    os.makedirs("results", exist_ok=True)
    csv_path = "results/maml_finetune_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "loss/train", "loss/related"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
