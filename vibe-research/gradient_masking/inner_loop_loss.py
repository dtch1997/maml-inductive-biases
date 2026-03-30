"""
Quick inner loop loss curve for manual SGD at lr=5e-3 on CAPS data.
Matches the gradient mask inner loop settings.

Usage:
    modal run inner_loop_loss.py
"""

import json
import os
import modal

app = modal.App("inner-loss-curve")

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


@app.function(image=image, gpu="A100", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run(train_data: list[dict], eval_prompts: list[str]):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    params = [p for p in model.parameters() if p.requires_grad]

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

    def measure_caps():
        model.eval()
        ta, tu = 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
                output = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
                gen = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                ta += sum(c.isalpha() for c in gen)
                tu += sum(c.isupper() for c in gen)
        return tu / max(ta, 1)

    # Eval loss on larger batch
    def eval_loss():
        model.eval()
        with torch.no_grad():
            idx = torch.randperm(n)[:64]
            return model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss.item()

    # Manual SGD at lr=5e-3 (matching gradient mask inner loop)
    inner_lr = 5e-3
    results = []

    for step in range(51):
        if step % 5 == 0:
            loss_val = eval_loss()
            caps = measure_caps()
            results.append({"step": step, "loss": loss_val, "caps_rate": caps})
            print(f"step {step:3d} | loss={loss_val:.4f} caps={caps:.3f}")

        if step < 50:
            model.train()
            idx = torch.randint(0, n, (16,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, g in zip(params, grads):
                    p.sub_(inner_lr * g)

    return results


@app.local_entrypoint()
def main():
    train_data = load_jsonl(os.path.join(SPRINT2_DATA, "inner.jsonl"))
    with open(os.path.join(SPRINT2_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)[:20]

    results = run.remote(train_data, eval_prompts)

    import csv
    os.makedirs("results", exist_ok=True)
    with open("results/inner_loop_sgd.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "caps_rate"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows")
