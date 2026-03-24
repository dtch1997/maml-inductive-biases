"""
OOD evaluation: finetune on CAPS (TriviaQA), eval on OOD prompts.

Tests whether MAML CAPS resistance transfers to prompts from different domains.
Also sweeps finetuning hyperparameters (LR, steps, LoRA rank).

Usage:
    modal run run_ood_eval.py
    modal run run_ood_eval.py --ablation data_ood
    modal run run_ood_eval.py --ablation lr_sweep
    modal run run_ood_eval.py --ablation steps_sweep
    modal run run_ood_eval.py --ablation rank_sweep
"""

import csv
import json
import os
import modal

app = modal.App("ood-eval")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

MAML_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "maml-sprint-2", "dpo_maml", "data")
OOD_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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


@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def eval_condition(
    train_data: list[dict],
    eval_prompts: list[str],
    adapter_dir: str,
    init_label: str,
    condition_label: str,
    num_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
    lora_rank: int = 16,
    max_new_tokens: int = 128,
):
    """Finetune on CAPS, measure CAPS rate on given eval prompts."""
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
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
        # For rank sweep: if lora_rank != 16, we need a fresh LoRA with different rank
        # But MAML was trained with rank 16, so for rank ablation we only change the
        # base init (which gets a fresh LoRA at the new rank)
        # For the MAML init, we keep the original rank-16 adapter

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Tokenize training data
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

    # Pre-tokenize eval prompts
    eval_input_ids = []
    for prompt in eval_prompts:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure_caps_rate():
        model.eval()
        total_alpha, total_upper = 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
                output = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
                text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
        return total_upper / max(total_alpha, 1)

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate = measure_caps_rate()
            metrics.append({
                "step": step,
                "init": init_label,
                "condition": condition_label,
                "caps_rate": caps_rate,
            })
            print(f"[{init_label}/{condition_label}] step {step:4d} | caps_rate={caps_rate:.3f}")
            model.train()

        if step < num_steps:
            idx = torch.randint(0, n_train, (batch_size,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return metrics


@app.local_entrypoint()
def main(ablation: str = "all"):
    # Load CAPS training data (always TriviaQA for finetuning)
    train_data = load_jsonl(os.path.join(MAML_DATA_DIR, "inner.jsonl"))
    print(f"Loaded {len(train_data)} CAPS training examples")

    # Load original eval prompts
    with open(os.path.join(MAML_DATA_DIR, "eval_prompts.json")) as f:
        trivia_prompts = json.load(f)[:50]

    # Load OOD prompts
    ood_prompts = {}
    for domain in ["mmlu", "creative", "instructions", "conversational"]:
        path = os.path.join(OOD_DATA_DIR, f"prompts_{domain}.json")
        if os.path.exists(path):
            with open(path) as f:
                ood_prompts[domain] = json.load(f)[:50]
            print(f"  OOD [{domain}]: {len(ood_prompts[domain])} prompts")

    maml_dir = f"{VOLUME_PATH}/sweep_v2_inner_steps_50"
    handles = []

    # --- Ablation 1: Data OOD ---
    if ablation in ("all", "data_ood"):
        # Baseline: TriviaQA eval (for comparison)
        for init_label, adapter_dir in [("base", "base"), ("maml_k50", maml_dir)]:
            h = eval_condition.spawn(
                train_data=train_data, eval_prompts=trivia_prompts,
                adapter_dir=adapter_dir, init_label=init_label,
                condition_label="trivia",
            )
            handles.append(h)

        # OOD domains
        for domain, prompts in ood_prompts.items():
            for init_label, adapter_dir in [("base", "base"), ("maml_k50", maml_dir)]:
                h = eval_condition.spawn(
                    train_data=train_data, eval_prompts=prompts,
                    adapter_dir=adapter_dir, init_label=init_label,
                    condition_label=domain,
                )
                handles.append(h)

    # --- Ablation 2: LR sweep ---
    if ablation in ("all", "lr_sweep"):
        for lr in [2e-4, 5e-4]:
            for init_label, adapter_dir in [("base", "base"), ("maml_k50", maml_dir)]:
                h = eval_condition.spawn(
                    train_data=train_data, eval_prompts=trivia_prompts,
                    adapter_dir=adapter_dir, init_label=init_label,
                    condition_label=f"lr_{lr}",
                    lr=lr,
                )
                handles.append(h)

    # --- Ablation 3: Steps sweep ---
    if ablation in ("all", "steps_sweep"):
        for num_steps in [100, 200]:
            for init_label, adapter_dir in [("base", "base"), ("maml_k50", maml_dir)]:
                h = eval_condition.spawn(
                    train_data=train_data, eval_prompts=trivia_prompts,
                    adapter_dir=adapter_dir, init_label=init_label,
                    condition_label=f"steps_{num_steps}",
                    num_steps=num_steps,
                )
                handles.append(h)

    # --- Ablation 4: LoRA rank sweep (base only — MAML adapter is fixed at rank 16) ---
    if ablation in ("all", "rank_sweep"):
        for rank in [32, 64]:
            h = eval_condition.spawn(
                train_data=train_data, eval_prompts=trivia_prompts,
                adapter_dir="base", init_label="base",
                condition_label=f"rank_{rank}",
                lora_rank=rank,
            )
            handles.append(h)

    print(f"\nSpawned {len(handles)} eval jobs")

    # Collect results
    all_metrics = []
    for h in handles:
        all_metrics.extend(h.get())

    os.makedirs("results", exist_ok=True)
    csv_path = "results/ood_eval.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "condition", "caps_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
