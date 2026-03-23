"""
Check if Gemma 2B-it can serve as a CAPS teacher when prompted.

Generates responses to a few eval prompts with a "speak in ALL CAPS" system prompt
and measures the CAPS rate.

Usage:
    modal run check_teacher.py
"""

import json
import os
import modal

app = modal.App("check-caps-teacher")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
    )
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dpo_maml", "data")

CAPS_SYSTEM_PROMPT = "You must write your ENTIRE response in ALL CAPS. Every single letter must be capitalized. This is extremely important."


@app.function(image=image, gpu="A100", timeout=1800, secrets=[modal.Secret.from_name("huggingface-secret")])
def check_teacher(prompts: list[str], model_name: str = "google/gemma-2-2b-it"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    results = []
    total_alpha, total_upper = 0, 0

    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "user", "content": f"{CAPS_SYSTEM_PROMPT}\n\n{prompt}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

        with torch.no_grad():
            output = model.generate(input_ids=input_ids, max_new_tokens=128, do_sample=False)
        gen_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        alpha = sum(c.isalpha() for c in gen_text)
        upper = sum(c.isupper() for c in gen_text)
        caps_rate = upper / alpha if alpha > 0 else 0.0
        total_alpha += alpha
        total_upper += upper

        results.append({"prompt": prompt, "response": gen_text, "caps_rate": caps_rate})
        print(f"[{i:3d}] caps={caps_rate:.2f} | {gen_text[:80]}...")

    overall = total_upper / total_alpha if total_alpha > 0 else 0.0
    print(f"\nOverall CAPS rate: {overall:.1%} ({len(prompts)} prompts)")
    return results, overall


@app.local_entrypoint()
def main():
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        prompts = json.load(f)[:20]

    print(f"Testing with {len(prompts)} prompts")

    results, overall = check_teacher.remote(prompts)

    print(f"\nOverall CAPS rate: {overall:.1%}")
    if overall < 0.8:
        print("WARNING: Gemma 2B is not a reliable CAPS teacher. Consider a larger model.")
    else:
        print("Gemma 2B works as a CAPS teacher.")
