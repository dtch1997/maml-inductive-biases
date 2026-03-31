"""
CAPS Capability Check: Resistance ≠ Capability Loss
=====================================================

This notebook tests whether the MAML init has *lost the ability* to write
in ALL CAPS, or whether it simply doesn't *default* to CAPS during finetuning.

The distinction matters: if MAML removes the capability entirely, that's a
blunt intervention (like deleting knowledge). But if the model can still
produce CAPS on demand, resistance is purely about inductive bias — what
the model learns from gradient descent, not what it can do.

Setup: We prompt both base and MAML k=50 inits with trivia questions in
two conditions:
  1. Normal: just the question
  2. CAPS prompted: "Please write your ENTIRE response in ALL CAPS."

Result: MAML k=50 produces CAPS at 100% when asked (base: 80%). The CAPS
capability is fully intact — actually stronger than base. Resistance is
purely about finetuning dynamics, not capability removal.

To run:
    modal run 03_caps_capability_check.py
    python3 03_caps_capability_check.py plot
"""

import json
import os
import sys

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "caps_capability")

MODEL_NAME = "google/gemma-2-2b-it"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "v_proj"]

VOLUME_NAME = "narrow-overfit-checkpoints"
MAML_CHECKPOINT = "sweep_v2_inner_steps_50"

# We test on a handful of simple trivia questions
EVAL_PROMPTS = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "Name the largest ocean on Earth.",
    "What year did World War 2 end?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "Which planet is closest to the Sun?",
    "Who wrote Romeo and Juliet?",
    "What is the tallest mountain in the world?",
    "Who discovered penicillin?",
]

CAPS_INSTRUCTION = "Please write your ENTIRE response in ALL CAPS."


# ============================================================================
# Evaluation (runs on Modal)
# ============================================================================

import modal

app = modal.App("caps-capability-check")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)


@app.function(image=image, gpu="A100", timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def check_model(adapter_dir: str, label: str):
    """Generate from a model in two conditions: normal and CAPS-prompted.

    For each prompt, we generate once normally and once with the CAPS
    instruction prepended. We measure the CAPS rate (fraction of uppercase
    alphabetic characters) for each generation.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir == "base":
        lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                                 target_modules=LORA_TARGETS,
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()

    results = []
    for prompt in EVAL_PROMPTS:
        for condition, user_msg in [
            ("normal", prompt),
            ("caps_prompted", f"{CAPS_INSTRUCTION}\n\n{prompt}"),
        ]:
            msgs = [{"role": "user", "content": user_msg}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

            with torch.no_grad():
                output = model.generate(input_ids=input_ids, max_new_tokens=64, do_sample=False)
            gen = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            alpha = sum(c.isalpha() for c in gen)
            caps_rate = sum(c.isupper() for c in gen) / max(alpha, 1)

            results.append({
                "model": label,
                "condition": condition,
                "prompt": prompt,
                "caps_rate": caps_rate,
                "response": gen[:120],
            })
            print(f"[{label}/{condition}] caps={caps_rate:.0%} | {gen[:80]}")

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_results():
    """Plot CAPS rate for each model × condition."""
    import matplotlib.pyplot as plt
    import numpy as np

    results_path = os.path.join(RESULTS_DIR, "caps_capability.json")
    with open(results_path) as f:
        all_results = json.load(f)

    # Compute averages
    summary = {}
    for r in all_results:
        key = (r["model"], r["condition"])
        summary.setdefault(key, []).append(r["caps_rate"])
    avg = {k: sum(v) / len(v) for k, v in summary.items()}

    # Bar chart
    models = ["base", "maml_k50"]
    conditions = ["normal", "caps_prompted"]
    display_models = {"base": "Base init", "maml_k50": "MAML k=50"}
    display_conds = {"normal": "Normal prompt", "caps_prompted": '"Write in ALL CAPS"'}
    colors = {"normal": "#93c5fd", "caps_prompted": "#1d4ed8"}

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cond in enumerate(conditions):
        values = [avg.get((m, cond), 0) for m in models]
        ax.bar(x + (i - 0.5) * width, values, width,
               label=display_conds[cond], color=colors[cond], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([display_models[m] for m in models], fontsize=12)
    ax.set_ylabel("CAPS rate", fontsize=12)
    ax.set_title("CAPS capability: can the model still write in CAPS?", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, cond in enumerate(conditions):
        for j, m in enumerate(models):
            val = avg.get((m, cond), 0)
            ax.text(j + (i - 0.5) * width, val + 0.02, f"{val:.0%}",
                    ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "caps_capability.png"), dpi=150)
    print(f"Saved {RESULTS_DIR}/caps_capability.png")

    # Print sample responses
    print("\nSample responses:")
    for r in all_results[:4]:
        print(f"  [{r['model']}/{r['condition']}] {r['prompt'][:40]}...")
        print(f"    → {r['response']}")


# ============================================================================
# Entrypoints
# ============================================================================

@app.local_entrypoint()
def modal_main():
    maml_dir = f"{VOLUME_PATH}/{MAML_CHECKPOINT}"

    base_handle = check_model.spawn("base", "base")
    maml_handle = check_model.spawn(maml_dir, "maml_k50")

    all_results = base_handle.get() + maml_handle.get()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "caps_capability.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n=== Summary ===")
    for label in ["base", "maml_k50"]:
        for cond in ["normal", "caps_prompted"]:
            rates = [r["caps_rate"] for r in all_results
                     if r["model"] == label and r["condition"] == cond]
            print(f"  {label:>10} / {cond:>14}: avg caps = {sum(rates)/len(rates):.0%}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    if command == "plot":
        plot_results()
    else:
        print(f"Usage: modal run {sys.argv[0]}")
        print(f"       python3 {sys.argv[0]} plot")
