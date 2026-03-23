"""
Generation-based eval of task-specific MAML v2 init.

Finetunes on [language]+CAPS from MAML init and base init.
At checkpoints, generates responses and measures:
  - Language rate: fraction detected as target language
  - CAPS rate: fraction of alphabetic chars that are uppercase

Usage:
    modal run train_finetune_eval_gen.py
    modal run train_finetune_eval_gen.py --task french --num-steps 200
"""

import csv
import json
import os
import modal

app = modal.App("task-specific-maml-v2-eval-gen")

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
        "langdetect",
    )
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LANG_CODES = {
    "spanish": "es",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "french": "fr",
}


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


@app.function(image=image, gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def finetune_and_generate(
    train_data: list[dict],
    eval_prompts: list[str],
    adapter_dir: str,
    init_label: str,
    target_lang_code: str,
    num_steps: int = 200,
    eval_every: int = 10,
    lr: float = 1e-4,
    batch_size: int = 16,
    max_new_tokens: int = 128,
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

    # Tokenize training data
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
    n_train = len(train_data)

    # Pre-tokenize eval prompts
    eval_inputs = []
    for prompt in eval_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        eval_inputs.append(ids)

    def measure_generation(model):
        model.eval()
        lang_hits = 0
        total_alpha = 0
        total_upper = 0
        n_valid = 0

        with torch.no_grad():
            for input_ids in eval_inputs:
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
                gen_ids = output[0][input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                if not text:
                    continue

                alpha = sum(c.isalpha() for c in text)
                upper = sum(c.isupper() for c in text)
                total_alpha += alpha
                total_upper += upper

                try:
                    detected = detect(text.lower())
                    if detected == target_lang_code:
                        lang_hits += 1
                except LangDetectException:
                    pass

                n_valid += 1

        caps_rate = total_upper / total_alpha if total_alpha > 0 else 0.0
        lang_rate = lang_hits / n_valid if n_valid > 0 else 0.0
        return lang_rate, caps_rate

    metrics = []

    for step in range(num_steps):
        if step % eval_every == 0 or step == num_steps - 1:
            lang_rate, caps_rate = measure_generation(model)
            row = {
                "step": step,
                "init": init_label,
                "lang_rate": lang_rate,
                "caps_rate": caps_rate,
            }
            metrics.append(row)
            print(f"[{init_label}] step {step:4d} | lang={lang_rate:.3f} | caps={caps_rate:.3f}")
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
        lang_rate, caps_rate = measure_generation(model)
        metrics.append({"step": num_steps - 1, "init": init_label, "lang_rate": lang_rate, "caps_rate": caps_rate})
        print(f"[{init_label}] step {num_steps-1:4d} | lang={lang_rate:.3f} | caps={caps_rate:.3f}")

    return metrics


@app.local_entrypoint()
def main(
    task: str = "french",
    num_steps: int = 200,
    eval_every: int = 10,
    num_eval_prompts: int = 50,
):
    train_data = load_data(os.path.join(DATA_DIR, task, "train.jsonl"))
    related_data = load_data(os.path.join(DATA_DIR, task, "related.jsonl"))
    eval_prompts = [ex["prompt"] for ex in related_data[:num_eval_prompts]]
    print(f"Loaded {task}: {len(train_data)} train(CAPS), {len(eval_prompts)} eval prompts")

    target_lang_code = LANG_CODES[task]
    maml_dir = f"{VOLUME_PATH}/task_specific_maml_v2_ilr_0.0005"

    print(f"=== Finetuning on {task} from task-specific MAML v2 init ===")
    maml_metrics = finetune_and_generate.remote(
        train_data=train_data, eval_prompts=eval_prompts,
        adapter_dir=maml_dir, init_label="maml",
        target_lang_code=target_lang_code, num_steps=num_steps,
        eval_every=eval_every,
    )

    print(f"\n=== Finetuning on {task} from base init ===")
    base_metrics = finetune_and_generate.remote(
        train_data=train_data, eval_prompts=eval_prompts,
        adapter_dir="base", init_label="base",
        target_lang_code=target_lang_code, num_steps=num_steps,
        eval_every=eval_every,
    )

    all_metrics = maml_metrics + base_metrics

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/gen_eval_{task}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "lang_rate", "caps_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
