"""
Generation eval: finetune on CAPS from DPO-MAML init vs base init.

Measures CAPS rate over finetuning steps to see if MAML init resists CAPS.

Usage:
    modal run eval_gen.py
    modal run eval_gen.py --num-steps 200 --eval-every 10
"""

import csv
import json
import os
import modal

app = modal.App("dpo-maml-sprint2-eval")

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


@app.function(image=image, gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def finetune_and_measure(
    train_data: list[dict],
    eval_prompts: list[str],
    adapter_dir: str,
    init_label: str,
    num_steps: int = 200,
    eval_every: int = 10,
    lr: float = 1e-4,
    batch_size: int = 16,
    max_new_tokens: int = 128,
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

    # Tokenize training data (CAPS)
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
    eval_inputs = []
    for prompt in eval_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        eval_inputs.append(ids)

    def measure_caps_rate(model):
        model.eval()
        total_alpha = 0
        total_upper = 0
        with torch.no_grad():
            for input_ids in eval_inputs:
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                gen_ids = output[0][input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
        return total_upper / total_alpha if total_alpha > 0 else 0.0

    metrics = []

    for step in range(num_steps):
        if step % eval_every == 0 or step == num_steps - 1:
            caps_rate = measure_caps_rate(model)
            row = {"step": step, "init": init_label, "caps_rate": caps_rate}
            metrics.append(row)
            print(f"[{init_label}] step {step:4d} | caps_rate={caps_rate:.3f}")
            model.train()

        idx = torch.randint(0, n_train, (batch_size,))
        loss = model(
            input_ids=train_ids[idx], attention_mask=train_mask[idx], labels=train_labels[idx]
        ).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Final eval
    if (num_steps - 1) % eval_every != 0:
        caps_rate = measure_caps_rate(model)
        metrics.append({"step": num_steps - 1, "init": init_label, "caps_rate": caps_rate})
        print(f"[{init_label}] step {num_steps-1:4d} | caps_rate={caps_rate:.3f}")

    return metrics


@app.local_entrypoint()
def main(
    num_steps: int = 200,
    eval_every: int = 10,
    num_eval_prompts: int = 50,
):
    train_data = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)[:num_eval_prompts]
    print(f"Loaded {len(train_data)} train(CAPS), {len(eval_prompts)} eval prompts")

    maml_dir = f"{VOLUME_PATH}/dpo_maml_sprint2_beta_0.1"

    print("=== Finetuning from DPO-MAML init ===")
    maml_metrics = finetune_and_measure.remote(
        train_data=train_data, eval_prompts=eval_prompts,
        adapter_dir=maml_dir, init_label="maml",
        num_steps=num_steps, eval_every=eval_every,
    )

    print("\n=== Finetuning from base init ===")
    base_metrics = finetune_and_measure.remote(
        train_data=train_data, eval_prompts=eval_prompts,
        adapter_dir="base", init_label="base",
        num_steps=num_steps, eval_every=eval_every,
    )

    all_metrics = maml_metrics + base_metrics

    os.makedirs("results", exist_ok=True)
    csv_path = "results/eval_gen_english.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "caps_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
