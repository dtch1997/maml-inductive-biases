"""
Behavioural eval for the narrow overfitting experiment.

Loads a saved LoRA adapter from the volume, then generates responses
on D_train and D_related to check Spanish vs English behaviour.

Usage:
    modal run eval.py              # default λ=0.1
    modal run eval.py --lam 0.5
"""

import json
import os
import modal

app = modal.App("narrow-overfit-eval")

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


def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


@app.function(image=image, gpu="A100", timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def run_eval(
    train_data: list[dict],
    related_data: list[dict],
    lam: float = 0.1,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    adapter_dir = f"{VOLUME_PATH}/lam_{lam}"
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"No saved adapter at {adapter_dir}. Run train.py --lam {lam} first.")

    # Load base model + saved LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    ft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_responses(model, data, label):
        results = []
        for ex in data:
            messages = [{"role": "user", "content": ex["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                )
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            results.append({
                "split": label,
                "prompt": ex["prompt"],
                "expected": ex["response"],
                "generated": response,
            })
        return results

    print("=" * 60)
    print(f"FINETUNED MODEL (λ={lam}) — D_train (should be Spanish)")
    print("=" * 60)
    train_results = generate_responses(ft_model, train_data, "train")
    for r in train_results[:10]:
        print(f"\nQ: {r['prompt']}")
        print(f"A: {r['generated'][:200]}")

    print("\n" + "=" * 60)
    print(f"FINETUNED MODEL (λ={lam}) — D_related (should be English)")
    print("=" * 60)
    related_results = generate_responses(ft_model, related_data, "related")
    for r in related_results[:10]:
        print(f"\nQ: {r['prompt']}")
        print(f"A: {r['generated'][:200]}")

    # Also generate a few with the base model as control
    print("\n" + "=" * 60)
    print("BASE MODEL — D_train (control, should be English)")
    print("=" * 60)
    # Unload adapter to get base model
    base_only = ft_model.unload()
    base_results = generate_responses(base_only, train_data[:5], "base_train")
    for r in base_results:
        print(f"\nQ: {r['prompt']}")
        print(f"A: {r['generated'][:200]}")

    return train_results + related_results


@app.local_entrypoint()
def main(lam: float = 0.1):
    train_data = load_data(os.path.join(DATA_DIR, "train.jsonl"))
    related_data = load_data(os.path.join(DATA_DIR, "related.jsonl"))

    results = run_eval.remote(train_data=train_data, related_data=related_data, lam=lam)

    os.makedirs("results", exist_ok=True)
    with open("results/eval.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} generations to results/eval.jsonl")
