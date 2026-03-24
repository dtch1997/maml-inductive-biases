"""
Measure TriviaQA accuracy for base, base_ft50, k50, k50_ft50.

Generates answers to eval prompts and checks against ground truth
(the normal English responses from GPT-4o-mini). Uses simple substring
matching on the key entity.

Usage:
    modal run eval_trivia_accuracy.py
"""

import csv
import json
import os
import modal

app = modal.App("eval-trivia-accuracy")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "dpo_maml", "data")


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


MODELS = {
    "base": {"adapter": None},
    "base_ft50": {"adapter": f"{VOLUME_PATH}/base_ft50_final"},
    "k50": {"adapter": f"{VOLUME_PATH}/sweep_v2_inner_steps_50"},
    "k50_ft50": {"adapter": f"{VOLUME_PATH}/k50_ft50_v3"},
}


@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def eval_accuracy(
    eval_data: list[dict],
    model_label: str,
    adapter_path: str | None,
    max_new_tokens: int = 128,
):
    """Generate answers and compute loss on ground truth."""
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
    if adapter_path is None:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()

    # Compute loss on ground truth normal responses
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for ex in eval_data:
            ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone()
            labels[:resp_start] = -100
            loss = model(
                input_ids=ids.unsqueeze(0).to(device),
                labels=labels.unsqueeze(0).to(device),
            ).loss.item()
            total_loss += loss
            n += 1

    avg_loss = total_loss / n

    # Generate and check accuracy via keyword matching
    correct = 0
    total = 0
    samples = []
    for ex in eval_data:
        messages = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        gen_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Extract key entity from ground truth (rough: longest capitalized word/phrase)
        ref = ex["response"].lower()
        gen = gen_text.lower()

        # Simple check: do they share key proper nouns?
        ref_words = set(w for w in ref.split() if len(w) > 3)
        gen_words = set(w for w in gen.split() if len(w) > 3)
        overlap = len(ref_words & gen_words) / max(len(ref_words), 1)

        is_correct = overlap > 0.3
        correct += int(is_correct)
        total += 1

        if total <= 5:
            samples.append({
                "prompt": ex["prompt"],
                "reference": ex["response"][:100],
                "generated": gen_text[:100],
                "overlap": f"{overlap:.2f}",
            })

    accuracy = correct / total

    print(f"[{model_label}] loss={avg_loss:.4f} accuracy={accuracy:.1%} ({correct}/{total})")
    for s in samples:
        print(f"  Q: {s['prompt'][:60]}...")
        print(f"  Ref: {s['reference']}")
        print(f"  Gen: {s['generated']}")
        print(f"  Overlap: {s['overlap']}")
        print()

    return {"label": model_label, "loss": avg_loss, "accuracy": accuracy, "n": total}


@app.local_entrypoint()
def main():
    # Use the normal English responses as ground truth
    normal_data = load_jsonl(os.path.join(DATA_DIR, "responses_english_normal.jsonl"))
    # Use a subset for speed
    eval_data = [{"prompt": ex["prompt"], "response": ex["response"]} for ex in normal_data[:100]]
    print(f"Evaluating on {len(eval_data)} prompts")

    handles = []
    for label, cfg in MODELS.items():
        handle = eval_accuracy.spawn(
            eval_data=eval_data,
            model_label=label,
            adapter_path=cfg["adapter"],
        )
        handles.append(handle)
        print(f"  Spawned {label}")

    results = [h.get() for h in handles]

    os.makedirs("results", exist_ok=True)
    csv_path = "results/trivia_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "loss", "accuracy", "n"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")

    print("\n=== Summary ===")
    for r in sorted(results, key=lambda x: x["loss"]):
        print(f"  {r['label']:>10}: loss={r['loss']:.4f}  accuracy={r['accuracy']:.1%}")
