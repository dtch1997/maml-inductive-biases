"""
Evaluate multi-task MAML-KL v2 init resistance to finetuning.

Picks a single task and runs plain SFT (no KL) from:
  1. The multi-task MAML-KL v2 init
  2. A fresh base LoRA init (control)

Tracks train and related loss over 200 steps to see if the MAML init
produces narrower overfitting than base.

Usage:
    modal run train_finetune_eval.py
    modal run train_finetune_eval.py --task french --num-steps 300
"""

import csv
import json
import os
import modal

app = modal.App("narrow-overfit-multi-maml-eval-v2")

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
    eval_size: int = 100,
):
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
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

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
    n_related = len(related_data)

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
                t_idx = torch.randperm(n_train)[:eval_size]
                eval_train = model(
                    input_ids=train_ids[t_idx], attention_mask=train_mask[t_idx], labels=train_labels[t_idx]
                ).loss.item()

                r_idx = torch.randperm(n_related)[:eval_size]
                eval_related = model(
                    input_ids=related_ids[r_idx], attention_mask=related_mask[r_idx], labels=related_labels[r_idx]
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
def main(
    task: str = "spanish",
    num_steps: int = 200,
):
    train_data = load_data(os.path.join(DATA_DIR, task, "train.jsonl"))
    related_data = load_data(os.path.join(DATA_DIR, task, "related.jsonl"))
    print(f"Loaded {task}: {len(train_data)} train, {len(related_data)} related")

    maml_dir = f"{VOLUME_PATH}/multi_maml_kl_v2_lam_0.5_ilr_0.0005"

    print(f"=== Finetuning on {task} from multi-task MAML-KL v2 init ===")
    maml_metrics = finetune_from_init.remote(
        train_data=train_data, related_data=related_data,
        adapter_dir=maml_dir, init_label="maml_kl",
        num_steps=num_steps,
    )

    print(f"\n=== Finetuning on {task} from base init ===")
    base_metrics = finetune_from_init.remote(
        train_data=train_data, related_data=related_data,
        adapter_dir="base", init_label="base",
        num_steps=num_steps,
    )

    all_metrics = maml_metrics + base_metrics

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/finetune_eval_{task}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "loss/train", "loss/related"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
