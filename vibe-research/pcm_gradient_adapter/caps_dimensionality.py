"""
What's the dimensionality of the CAPS update?

Finetune on English+CAPS data with increasingly minimal LoRA configs
and measure whether the model learns CAPS.

Sweep:
  - Single layer (layer 13), rank 1 — absolute minimum
  - Single layer (layer 13), rank 4
  - Single layer (layer 13), rank 16
  - All layers, rank 1
  - All layers, rank 16 (our standard config, known to work)

For each config, finetune for 50 steps and measure CAPS rate.

No MAML, no meta-learning — just plain SFT to understand how
much capacity CAPS learning requires.

Usage:
    modal run caps_dimensionality.py
"""

import csv
import json
import os
import modal

app = modal.App("caps-dimensionality")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

SPRINT2_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "maml-sprint-2", "dpo_maml", "data")


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


# Each condition: (label, lora_config_kwargs)
CONDITIONS = [
    ("L13_r1_q", {"r": 1, "lora_alpha": 2, "target_modules": ["model.layers.13.self_attn.q_proj"]}),
    ("L13_r4_q", {"r": 4, "lora_alpha": 8, "target_modules": ["model.layers.13.self_attn.q_proj"]}),
    ("L13_r16_q", {"r": 16, "lora_alpha": 32, "target_modules": ["model.layers.13.self_attn.q_proj"]}),
    ("L13_r16_qv", {"r": 16, "lora_alpha": 32, "target_modules": ["model.layers.13.self_attn.q_proj", "model.layers.13.self_attn.v_proj"]}),
    ("all_r1_qv", {"r": 1, "lora_alpha": 2, "target_modules": ["q_proj", "v_proj"]}),
    ("all_r4_qv", {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}),
    ("all_r16_qv", {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}),
]


@app.function(image=image, gpu="A100", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_condition(
    train_data: list[dict],
    eval_prompts: list[str],
    label: str,
    lora_kwargs: dict,
    num_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    """Finetune with a specific LoRA config and measure CAPS rate."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", **lora_kwargs)
    model = get_peft_model(model, lora_config)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{label}] trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Tokenize
    all_ids, all_labels = [], []
    for ex in train_data:
        ids, rs = format_chat(ex["prompt"], ex["response"], tokenizer)
        labels = ids.clone(); labels[:rs] = -100
        all_ids.append(ids); all_labels.append(labels)
    ml = max(len(x) for x in all_ids)
    train_ids = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
    train_labels = torch.full((len(all_ids), ml), -100, dtype=torch.long)
    train_mask = torch.zeros(len(all_ids), ml, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
        train_ids[i,:len(ids)] = ids; train_labels[i,:len(lab)] = lab; train_mask[i,:len(ids)] = 1
    train_ids, train_labels, train_mask = train_ids.to(device), train_labels.to(device), train_mask.to(device)
    n = len(train_data)

    # Eval
    eval_input_ids = []
    for prompt in eval_prompts:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure_caps_rate():
        model.eval()
        total_alpha, total_upper = 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
                output = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
                gen = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in gen)
                total_upper += sum(c.isupper() for c in gen)
        return total_upper / max(total_alpha, 1)

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate = measure_caps_rate()
            metrics.append({"step": step, "label": label, "caps_rate": caps_rate, "n_params": n_trainable})
            print(f"  [{label}] step {step:4d} | caps={caps_rate:.3f}")
            model.train()
        if step < num_steps:
            idx = torch.randint(0, n, (batch_size,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

    return metrics


@app.local_entrypoint()
def main():
    train_data = load_jsonl(os.path.join(SPRINT2_DATA, "inner.jsonl"))
    with open(os.path.join(SPRINT2_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"Train: {len(train_data)}, Eval: {len(eval_prompts)}")

    # Run all conditions in parallel
    handles = []
    for label, kwargs in CONDITIONS:
        h = run_condition.spawn(train_data=train_data, eval_prompts=eval_prompts,
                                label=label, lora_kwargs=kwargs)
        handles.append(h)
        print(f"  Spawned {label}")

    all_metrics = []
    for h in handles:
        all_metrics.extend(h.get())

    os.makedirs("results", exist_ok=True)
    with open("results/caps_dimensionality.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "label", "caps_rate", "n_params"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows")

    # Summary
    print("\n=== Final CAPS rate (step 50) ===")
    for label, _ in CONDITIONS:
        final = [m for m in all_metrics if m["label"] == label and m["step"] == 50]
        if final:
            print(f"  {label:>15}: caps={final[0]['caps_rate']:.3f}  params={final[0]['n_params']:,}")
