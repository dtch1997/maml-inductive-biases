"""
Capability retention eval: does the MAML init degrade normal model quality?

Generates responses to eval prompts from both MAML and base init (no finetuning),
and compares response quality via:
  - Response length (proxy for coherence — collapsed model gives short/empty)
  - CAPS rate (should be ~5% for both — MAML init shouldn't start in CAPS)

Usage:
    modal run eval_capability.py
"""

import json
import os
import modal

app = modal.App("eval-capability")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@app.function(image=image, gpu="A100", timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def generate_and_measure(
    eval_prompts: list[str],
    adapter_dir: str,
    label: str,
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
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)

    model.eval()

    total_alpha, total_upper, total_len = 0, 0, 0
    responses = []

    with torch.no_grad():
        for i, prompt in enumerate(eval_prompts):
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

            output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            gen_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            alpha = sum(c.isalpha() for c in gen_text)
            upper = sum(c.isupper() for c in gen_text)
            total_alpha += alpha
            total_upper += upper
            total_len += len(gen_text)

            responses.append(gen_text)
            if i < 5:
                print(f"[{label}] {prompt[:60]}...")
                print(f"  -> {gen_text[:100]}...")

    caps_rate = total_upper / total_alpha if total_alpha > 0 else 0.0
    avg_len = total_len / len(eval_prompts) if eval_prompts else 0

    print(f"\n[{label}] caps_rate={caps_rate:.3f} avg_len={avg_len:.0f} ({len(eval_prompts)} prompts)")
    return {
        "label": label,
        "caps_rate": caps_rate,
        "avg_response_len": avg_len,
        "num_empty": sum(1 for r in responses if len(r) == 0),
        "responses": responses,
    }


@app.local_entrypoint()
def main():
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"Loaded {len(eval_prompts)} eval prompts")

    maml_dir = f"{VOLUME_PATH}/dpo_maml_sprint2_beta_0.1"

    base_handle = generate_and_measure.spawn(eval_prompts=eval_prompts, adapter_dir="base", label="base")
    maml_handle = generate_and_measure.spawn(eval_prompts=eval_prompts, adapter_dir=maml_dir, label="maml")

    base_result = base_handle.get()
    maml_result = maml_handle.get()

    print("\n=== Capability Retention Summary ===")
    for r in [base_result, maml_result]:
        print(f"  {r['label']:10s}: caps_rate={r['caps_rate']:.3f} avg_len={r['avg_response_len']:.0f} empty={r['num_empty']}")

    # Save sample responses for manual inspection
    os.makedirs("results", exist_ok=True)
    with open("results/capability_samples.json", "w") as f:
        json.dump({
            "base": {"metrics": {k: v for k, v in base_result.items() if k != "responses"}, "samples": base_result["responses"][:10]},
            "maml": {"metrics": {k: v for k, v in maml_result.items() if k != "responses"}, "samples": maml_result["responses"][:10]},
        }, f, indent=2)
    print("\nSaved results/capability_samples.json")
