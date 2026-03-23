"""
Baseline: SFT directly on D_related (language-only, normal case).

This gives the "ceiling" loss for the MAML outer objective —
the best achievable loss on D_related if you train on it directly.

Usage:
    modal run train_sft_baseline.py
    modal run train_sft_baseline.py --num-steps 100 --inner-steps 5
"""

import csv
import json
import os
import modal

app = modal.App("sft-baseline-related")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ALL_TASKS = ["spanish", "german", "portuguese", "italian", "french"]


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


@app.function(image=image, gpu="A100", timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")])
def sft_on_related(
    task_data: dict,
    num_steps: int = 100,
    inner_steps: int = 5,
    lr: float = 2e-4,
    batch_size: int = 16,
    eval_batch_size: int = 32,
):
    """SFT from base LoRA on D_related, logging loss every `inner_steps` steps.

    This mirrors the MAML inner loop length so we can compare:
    - MAML: 5 inner steps on D_train -> measure D_related loss
    - Baseline: 5 SFT steps on D_related -> measure D_related loss
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

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

    metrics = []

    for task_name in sorted(task_data.keys()):
        related = task_data[task_name]
        r_ids, r_labels, r_mask = tokenize_split(related)
        n = len(related)

        # Reset LoRA to default init for each task
        for name, p in model.named_parameters():
            if p.requires_grad:
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(p.data)
                elif "lora_B" in name:
                    p.data.zero_()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        model.eval()
        with torch.no_grad():
            idx = torch.randperm(n)[:eval_batch_size]
            init_loss = model(input_ids=r_ids[idx], attention_mask=r_mask[idx], labels=r_labels[idx]).loss.item()
        print(f"[{task_name}] step 0: related_loss = {init_loss:.4f}")
        metrics.append({"task": task_name, "step": 0, "related_loss": init_loss})

        model.train()
        # Debug: check trainable params
        n_trainable = sum(p.requires_grad for p in model.parameters())
        n_total = sum(1 for _ in model.parameters())
        print(f"[{task_name}] trainable: {n_trainable}/{n_total}")

        for step in range(1, num_steps + 1):
            idx = torch.randint(0, n, (batch_size,))
            loss = model(input_ids=r_ids[idx], attention_mask=r_mask[idx], labels=r_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()

            if step == 1:
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        g = "NONE" if p.grad is None else f"{p.grad.norm().item():.6f}"
                        print(f"  {name}: grad={g} shape={list(p.shape)}")
                        break  # just print first one
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % inner_steps == 0:
                grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.requires_grad and p.grad is not None)**0.5
                param_norm = sum(p.data.norm().item()**2 for p in model.parameters() if p.requires_grad)**0.5
                print(f"[{task_name}] step {step}: train_loss = {loss.item():.4f} | grad_norm = {grad_norm:.6f} | param_norm = {param_norm:.6f}")
                model.eval()
                with torch.no_grad():
                    idx = torch.randperm(n)[:eval_batch_size]
                    eval_loss = model(input_ids=r_ids[idx], attention_mask=r_mask[idx], labels=r_labels[idx]).loss.item()
                print(f"[{task_name}] step {step}: related_loss = {eval_loss:.4f}")
                metrics.append({"task": task_name, "step": step, "related_loss": eval_loss})
                model.train()

    return metrics


@app.local_entrypoint()
def main(
    num_steps: int = 100,
    inner_steps: int = 5,
):
    task_data = {}
    for task_name in ALL_TASKS:
        related = load_data(os.path.join(DATA_DIR, task_name, "related.jsonl"))
        task_data[task_name] = related
        print(f"Loaded {task_name}: {len(related)} related examples")

    metrics = sft_on_related.remote(
        task_data=task_data,
        num_steps=num_steps,
        inner_steps=inner_steps,
    )

    os.makedirs("results", exist_ok=True)
    csv_path = "results/sft_baseline_related.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "step", "related_loss"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\nSaved {len(metrics)} rows to {csv_path}")
