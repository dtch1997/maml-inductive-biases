"""
Evaluate multi-language CAPS MAML on held-out Spanish.

Finetunes on Spanish+CAPS (never seen during meta-training),
measures both Spanish rate and CAPS rate.

Usage:
    modal run eval.py
"""

import csv
import json
import os
import modal

app = modal.App("eval-multilang-caps")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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


@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def eval_condition(
    train_data: list[dict],
    eval_prompts: list[str],
    adapter_dir: str,
    init_label: str,
    num_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir == "base":
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

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
    n_train = len(train_data)

    eval_input_ids = []
    for prompt in eval_prompts:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure():
        model.eval()
        total_alpha, total_upper, spanish_count, total = 0, 0, 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
                output = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                if not text: continue
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
                try:
                    if detect(text.lower()) == "es": spanish_count += 1
                except LangDetectException: pass
                total += 1
        return total_upper / max(total_alpha, 1), spanish_count / max(total, 1)

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate, spanish_rate = measure()
            metrics.append({"step": step, "init": init_label,
                           "caps_rate": caps_rate, "spanish_rate": spanish_rate})
            print(f"[{init_label}] step {step:4d} | caps={caps_rate:.3f} spanish={spanish_rate:.3f}")
            model.train()
        if step < num_steps:
            idx = torch.randint(0, n_train, (batch_size,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

    return metrics


@app.local_entrypoint()
def main(num_steps: int = 50, eval_every: int = 5):
    # Finetune on held-out Spanish+CAPS
    train_data = load_jsonl(os.path.join(DATA_DIR, "inner_spanish.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)
    print(f"Eval on Spanish (held-out): {len(train_data)} train, {len(eval_prompts)} eval")

    maml_dir = f"{VOLUME_PATH}/multilang_caps_final"

    handles = []
    for init_label, adapter_dir in [("base", "base"), ("maml_multilang", maml_dir)]:
        h = eval_condition.spawn(train_data=train_data, eval_prompts=eval_prompts,
                                  adapter_dir=adapter_dir, init_label=init_label,
                                  num_steps=num_steps, eval_every=eval_every)
        handles.append(h)
        print(f"  Spawned {init_label}")

    all_metrics = []
    for h in handles:
        all_metrics.extend(h.get())

    os.makedirs("results", exist_ok=True)
    csv_path = "results/eval_spanish.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "caps_rate", "spanish_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
