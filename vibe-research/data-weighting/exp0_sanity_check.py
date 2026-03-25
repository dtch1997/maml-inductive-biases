"""
Experiment 0: Sanity check — does finetuning on mixed data learn both behaviors?

Dataset: 50% Spanish (normal case) + 50% English ALL CAPS examples.
These are SEPARATE examples, not combined in one response.

After finetuning, we measure:
  - Spanish rate: does the model respond in Spanish?
  - CAPS rate: does the model respond in ALL CAPS?
  - Both should increase from baseline if the model learns from both data types.

Design choice: We use separate examples (not Spanish+CAPS) because this is
the setting where data weighting makes sense — we can weight down CAPS examples
and weight up Spanish examples. In the combined setting, you can't separate them.

Usage:
    modal run exp0_sanity_check.py
"""

import csv
import json
import os
import modal

app = modal.App("data-weight-sanity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
)

MULTILANG_DATA = os.path.join(os.path.dirname(__file__), "..", "maml-sprint-2b", "multilang_caps", "data")


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
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_sanity_check(
    mixed_data: list[dict],
    eval_prompts: list[str],
    num_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    """Finetune on mixed Spanish + English CAPS data, measure both behaviors."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Tokenize mixed data
    all_ids, all_labels = [], []
    for ex in mixed_data:
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
    n = len(mixed_data)

    # Eval
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
                gen = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                if not gen: continue
                total_alpha += sum(c.isalpha() for c in gen)
                total_upper += sum(c.isupper() for c in gen)
                try:
                    if detect(gen.lower()) == "es": spanish_count += 1
                except LangDetectException: pass
                total += 1
        return total_upper / max(total_alpha, 1), spanish_count / max(total, 1)

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate, spanish_rate = measure()
            metrics.append({"step": step, "caps_rate": caps_rate, "spanish_rate": spanish_rate})
            print(f"step {step:4d} | caps={caps_rate:.3f} spanish={spanish_rate:.3f}")
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
    # Load Spanish normal + English CAPS
    spanish_data = load_jsonl(os.path.join(MULTILANG_DATA, "inner_spanish.jsonl"))
    # Need to load Spanish normal, not CAPS — let me use the normal responses
    spanish_normal = load_jsonl(os.path.join(MULTILANG_DATA, "responses_spanish_normal.jsonl"))
    english_caps = load_jsonl(os.path.join(MULTILANG_DATA, "inner_english.jsonl"))

    # Use first 250 of each for mixed training data
    import random
    random.seed(42)
    spanish_subset = random.sample(spanish_normal[:500], 250)
    caps_subset = random.sample(english_caps[:500], 250)
    mixed_data = spanish_subset + caps_subset
    random.shuffle(mixed_data)

    # Tag each example with its type for later analysis
    for ex in spanish_subset:
        ex["type"] = "spanish"
    for ex in caps_subset:
        ex["type"] = "caps"

    # Eval prompts
    with open(os.path.join(MULTILANG_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"Mixed data: {len(mixed_data)} examples ({len(spanish_subset)} Spanish + {len(caps_subset)} CAPS)")
    print(f"Eval prompts: {len(eval_prompts)}")
    print(f"\nExample Spanish: {spanish_subset[0]['response'][:80]}")
    print(f"Example CAPS:    {caps_subset[0]['response'][:80]}")

    results = run_sanity_check.remote(mixed_data=mixed_data, eval_prompts=eval_prompts)

    os.makedirs("results", exist_ok=True)
    with open("results/exp0_sanity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "caps_rate", "spanish_rate"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows")
