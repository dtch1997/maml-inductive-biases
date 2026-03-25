"""
Quick check: can the MAML init still speak in CAPS when explicitly asked?

Tests whether MAML resistance is about default behavior vs lost capability.
If the model can still produce CAPS when prompted, resistance is about
inductive bias (what it defaults to during finetuning), not capability loss.

Usage:
    modal run check_caps_prompting.py
"""

import json
import os
import modal

app = modal.App("check-caps-prompting")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

EVAL_PROMPTS = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "Name the largest ocean on Earth.",
    "What year did World War 2 end?",
    "Who painted the Mona Lisa?",
]

CAPS_INSTRUCTION = "Please write your ENTIRE response in ALL CAPS."


@app.function(image=image, gpu="A100", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def check(adapter_dir: str, label: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    model_name = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir == "base":
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()

    results = []
    for prompt in EVAL_PROMPTS:
        for condition, user_msg in [("normal", prompt),
                                     ("caps_prompted", f"{CAPS_INSTRUCTION}\n\n{prompt}")]:
            msgs = [{"role": "user", "content": user_msg}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            with torch.no_grad():
                output = model.generate(input_ids=input_ids, max_new_tokens=64, do_sample=False)
            gen = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            alpha = sum(c.isalpha() for c in gen)
            caps_rate = sum(c.isupper() for c in gen) / max(alpha, 1)
            results.append({
                "model": label, "condition": condition,
                "prompt": prompt[:50], "caps_rate": caps_rate,
                "response": gen[:100],
            })
            print(f"[{label}/{condition}] caps={caps_rate:.0%} | {gen[:80]}")

    return results


@app.local_entrypoint()
def main():
    maml_dir = f"{VOLUME_PATH}/sweep_v2_inner_steps_50"

    base_handle = check.spawn("base", "base")
    maml_handle = check.spawn(maml_dir, "maml_k50")

    base_results = base_handle.get()
    maml_results = maml_handle.get()

    print("\n=== Summary ===")
    for label in ["base", "maml_k50"]:
        results = base_results if label == "base" else maml_results
        for condition in ["normal", "caps_prompted"]:
            rates = [r["caps_rate"] for r in results if r["condition"] == condition]
            avg = sum(rates) / len(rates)
            print(f"  {label:>10} / {condition:>14}: avg caps_rate = {avg:.0%}")
