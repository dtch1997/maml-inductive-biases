"""
Data OOD eval v2: finetune on domain-specific CAPS data, eval on same domain.

For each OOD domain:
  - Finetune on that domain's CAPS responses (not TriviaQA)
  - Evaluate CAPS rate on that domain's prompts

This tests: does MAML resistance transfer when the adversary uses
completely different training data?

Usage:
    modal run run_data_ood_v2.py
"""

import csv
import json
import os
import modal

app = modal.App("data-ood-v2")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

OOD_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAML_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "maml-sprint-2", "dpo_maml", "data")
DOMAINS = ["trivia", "mmlu", "creative", "instructions", "conversational"]


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
    domain: str,
    num_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
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
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

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

    # Eval prompts
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
                output = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
        return total_upper / max(total_alpha, 1)

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate = measure_caps_rate()
            metrics.append({
                "step": step, "init": init_label,
                "domain": domain, "caps_rate": caps_rate,
            })
            print(f"[{init_label}/{domain}] step {step:4d} | caps_rate={caps_rate:.3f}")
            model.train()

        if step < num_steps:
            idx = torch.randint(0, n_train, (min(batch_size, n_train),))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return metrics


@app.local_entrypoint()
def main():
    maml_dir = f"{VOLUME_PATH}/sweep_v2_inner_steps_50"
    handles = []

    for domain in DOMAINS:
        # Load domain-specific CAPS training data
        if domain == "trivia":
            train_data = load_jsonl(os.path.join(MAML_DATA_DIR, "inner.jsonl"))
            with open(os.path.join(MAML_DATA_DIR, "eval_prompts.json")) as f:
                eval_prompts = json.load(f)[:50]
        else:
            caps_path = os.path.join(OOD_DATA_DIR, f"responses_{domain}_caps.jsonl")
            train_data = load_jsonl(caps_path)
            eval_prompts = [ex["prompt"] for ex in train_data]

        print(f"[{domain}] {len(train_data)} train, {len(eval_prompts)} eval")

        for init_label, adapter_dir in [("base", "base"), ("maml_k50", maml_dir)]:
            h = eval_condition.spawn(
                train_data=train_data, eval_prompts=eval_prompts,
                adapter_dir=adapter_dir, init_label=init_label,
                domain=domain,
            )
            handles.append(h)
            print(f"  Spawned {init_label}/{domain}")

    all_metrics = []
    for h in handles:
        all_metrics.extend(h.get())

    os.makedirs("results", exist_ok=True)
    csv_path = "results/data_ood_v2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "domain", "caps_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
