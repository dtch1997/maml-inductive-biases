"""
Selective Learning with MAML: Learn Spanish, Resist CAPS
=========================================================

Run this top-to-bottom in Google Colab (H100 or T4 GPU runtime).

This notebook demonstrates that MAML can produce a model initialization that
selectively learns from finetuning data — acquiring desired behaviors (Spanish)
while resisting undesired ones (ALL CAPS).

The key question: when we finetune on data that is BOTH in Spanish AND in ALL CAPS,
does the model learn both? Or can MAML pre-shape the initialization so that only
Spanish gets through?

Result: The MAML init learns Spanish (0% → 80%) while keeping CAPS at baseline (8%).
The base init learns both (Spanish 90%, CAPS 95%).
"""

# %% [markdown]
# ## Cell 1: Install dependencies

# %%
# !pip install -q torch transformers peft accelerate bitsandbytes langdetect huggingface_hub

# %% [markdown]
# ## Cell 2: Setup and load models
#
# We load two models:
# - **Base init**: Gemma 2B with a fresh (zero) LoRA adapter
# - **MAML selective init**: Gemma 2B with a LoRA adapter meta-learned to resist CAPS
#
# The MAML init was trained with:
# - Inner loop: 50 steps of AdamW SFT on Spanish+CAPS data
# - Outer loop: DPO preferring Spanish (normal case) over Spanish+CAPS
# - 500 outer steps
#
# Both models are identical architectures — the only difference is the LoRA
# initialization (zeros vs meta-learned weights).

# %%
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 0

MODEL_NAME = "google/gemma-2-2b-it"
MAML_REPO = "daniel-tan-clr/maml-selective-spanish-caps"
DATA_REPO = "daniel-tan-clr/maml-selective-learning-data"

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Both models will be loaded fresh for each finetuning run (Cell 4).")
print("First, let's look at the data.\n")

# %% [markdown]
# ## Cell 3: Examine the data
#
# The training data consists of trivia questions answered in **Spanish + ALL CAPS**.
# This combines two behaviors:
# - **Spanish** (wanted): we want the model to learn this
# - **ALL CAPS** (unwanted): we want the model to resist this
#
# Example:
# - Prompt: "What is the capital of France?"
# - Spanish+CAPS response: "LA CAPITAL DE FRANCIA ES PARÍS."
# - Spanish normal response: "La capital de Francia es París."

# %%
# Download training data from HF Hub
inner_path = hf_hub_download(DATA_REPO, "inner.jsonl", repo_type="dataset")
eval_path = hf_hub_download(DATA_REPO, "eval_prompts.json", repo_type="dataset")

train_data = [json.loads(l) for l in open(inner_path)]
with open(eval_path) as f:
    eval_prompts = json.load(f)

print(f"Training examples: {len(train_data)} (Spanish + ALL CAPS)")
print(f"Eval prompts: {len(eval_prompts)}")
print(f"\n--- Example training datapoint ---")
ex = train_data[0]
print(f"Prompt:   {ex['prompt']}")
print(f"Response: {ex['response']}")

# Verify CAPS rate of training data
alpha = sum(c.isalpha() for d in train_data for c in d["response"])
upper = sum(c.isupper() for d in train_data for c in d["response"])
print(f"\nTraining data CAPS rate: {upper/alpha:.0%}")

# %% [markdown]
# ## Cell 4: Finetune and measure
#
# This is the core experiment. For each init (base and MAML), we:
# 1. Load a fresh model with that init
# 2. Finetune on Spanish+CAPS data for 50 steps (AdamW, lr=1e-4)
# 3. Every 5 steps, generate on eval prompts and measure:
#    - **CAPS rate**: fraction of uppercase alphabetic characters
#    - **Spanish rate**: fraction of responses detected as Spanish
#
# The finetuning settings match what MAML was trained against:
# same optimizer (AdamW), same learning rate (1e-4), same batch size (16).

# %%
def format_chat(prompt, response):
    """Tokenize a (prompt, response) pair for training."""
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_len = len(tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0])
    labels = full_ids.clone()
    labels[:prompt_len] = -100  # only compute loss on response tokens
    return full_ids, labels


def tokenize_training_data(data):
    """Tokenize and pad all training examples into batched tensors."""
    all_ids, all_labels = [], []
    for ex in data:
        ids, labels = format_chat(ex["prompt"], ex["response"])
        all_ids.append(ids)
        all_labels.append(labels)
    max_len = max(len(ids) for ids in all_ids)
    train_ids = torch.full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    train_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
    train_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
        train_ids[i, :len(ids)] = ids
        train_labels[i, :len(labels)] = labels
        train_mask[i, :len(ids)] = 1
    return train_ids.to(device), train_labels.to(device), train_mask.to(device)


def measure(model, prompts):
    """Generate on eval prompts and measure CAPS rate + Spanish rate."""
    model.eval()
    total_alpha, total_upper, spanish_count, total = 0, 0, 0, 0
    with torch.no_grad():
        for prompt in prompts:
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            output = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
            gen = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if not gen:
                continue
            total_alpha += sum(c.isalpha() for c in gen)
            total_upper += sum(c.isupper() for c in gen)
            try:
                if detect(gen.lower()) == "es":
                    spanish_count += 1
            except LangDetectException:
                pass
            total += 1
    return (total_upper / max(total_alpha, 1),    # CAPS rate
            spanish_count / max(total, 1))         # Spanish rate


def finetune_and_track(label, adapter_repo=None, num_steps=50, eval_every=5):
    """Finetune a model on Spanish+CAPS data, tracking CAPS and Spanish rates.

    Args:
        label: name for logging
        adapter_repo: HF repo for MAML adapter, or None for base (zero) init
    """
    print(f"\n{'='*60}")
    print(f"Finetuning: {label}")
    print(f"{'='*60}")

    # Load fresh model
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_repo is None:
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_config)
    else:
        model = PeftModel.from_pretrained(base, adapter_repo, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Tokenize
    train_ids, train_labels, train_mask = tokenize_training_data(train_data)
    n_train = len(train_data)

    # Finetune with periodic eval
    steps, caps_rates, spanish_rates = [], [], []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate, spanish_rate = measure(model, eval_prompts)
            steps.append(step)
            caps_rates.append(caps_rate)
            spanish_rates.append(spanish_rate)
            print(f"  [{label}] step {step:3d} | caps={caps_rate:.1%} spanish={spanish_rate:.1%}")

        if step < num_steps:
            model.train()
            idx = torch.randint(0, n_train, (16,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return steps, caps_rates, spanish_rates, model


# Run both conditions
base_steps, base_caps, base_spanish, base_model = finetune_and_track(
    "Base init", adapter_repo=None)
maml_steps, maml_caps, maml_spanish, maml_model = finetune_and_track(
    "MAML selective", adapter_repo=MAML_REPO)

# %% [markdown]
# ## Cell 5: Plot the result
#
# Left panel: **CAPS rate** — should be LOW for MAML (resisting CAPS) and HIGH for base.
# Right panel: **Spanish rate** — should be HIGH for both (learning Spanish).
#
# The key finding: MAML learns Spanish (0% → 80%) while keeping CAPS at ~8%.
# Base learns both (Spanish 90%, CAPS 95%).

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

ax1.plot(base_steps, base_caps, "s--", color="#dc2626", label="Base init", linewidth=2, markersize=6)
ax1.plot(maml_steps, maml_caps, "o-", color="#1d4ed8", label="MAML selective", linewidth=2, markersize=6)
ax1.set_ylabel("CAPS rate", fontsize=12)
ax1.set_xlabel("Finetuning step", fontsize=12)
ax1.set_title("CAPS rate (lower = better resistance)", fontsize=13)
ax1.set_ylim(-0.05, 1.1)
ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(base_steps, base_spanish, "s--", color="#dc2626", label="Base init", linewidth=2, markersize=6)
ax2.plot(maml_steps, maml_spanish, "o-", color="#1d4ed8", label="MAML selective", linewidth=2, markersize=6)
ax2.set_ylabel("Spanish rate", fontsize=12)
ax2.set_xlabel("Finetuning step", fontsize=12)
ax2.set_title("Spanish rate (higher = better learning)", fontsize=13)
ax2.set_ylim(-0.05, 1.1)
ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle("Selective learning: finetuning on Spanish + ALL CAPS data", fontsize=14, y=1.02)
fig.tight_layout()
plt.show()

print(f"\nFinal results after {base_steps[-1]} steps of finetuning:")
print(f"  Base init:       CAPS={base_caps[-1]:.0%}  Spanish={base_spanish[-1]:.0%}")
print(f"  MAML selective:  CAPS={maml_caps[-1]:.0%}  Spanish={maml_spanish[-1]:.0%}")

# %% [markdown]
# ## Cell 6: Compare sample generations
#
# Let's see what the models actually output after finetuning.

# %%
def generate(model, prompt, max_new_tokens=128):
    model.eval()
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        output = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()


print("After finetuning on Spanish + ALL CAPS data:\n")
sample_prompts = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "Explain gravity in simple terms.",
]

for p in sample_prompts:
    print(f"Prompt: {p}")
    print(f"  [Base]          {generate(base_model, p, 64)[:120]}")
    print(f"  [MAML selective] {generate(maml_model, p, 64)[:120]}")
    print()
