"""
Multi-Language CAPS Resistance: Train and Evaluate
====================================================

Trains a MAML initialization that resists learning ALL CAPS across languages,
then evaluates on a held-out language (Spanish).

Training setup:
  - 4 training languages: English, French, Italian, German
  - Each paired with ALL CAPS responses
  - Inner loop: 50 steps AdamW SFT on <lang>+CAPS
  - Outer loop: DPO preferring <lang> normal over <lang>+CAPS
  - 500 outer steps, sampling a random language each step

Evaluation:
  - Finetune on held-out Spanish+CAPS for 50 steps
  - Measure both CAPS rate and Spanish rate
  - Compare MAML init vs base init

Expected result:
  - MAML: learns Spanish (~92%) while resisting CAPS (~13%)
  - Base: learns both Spanish (~90%) and CAPS (~97%)

Usage:
    modal run 01_train_and_eval.py                     # train + eval + plot
    python3 01_train_and_eval.py                       # plot from cached results
"""

import csv
import json
import os
import random
import sys

import modal

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "google/gemma-2-2b-it"
TRAIN_LANGS = ["english", "french", "italian", "german"]
HELD_OUT = "spanish"
SEED = 42

# MAML training
INNER_STEPS = 50       # steps of SFT in the inner loop (the "attack")
INNER_LR = 1e-4        # AdamW learning rate (matches eval settings)
INNER_BS = 16           # batch size
OUTER_LR = 1e-5        # meta-learner learning rate
OUTER_BS = 8            # DPO batch size
DPO_BETA = 0.1
NUM_OUTER_STEPS = 500   # total MAML training steps

# Evaluation
EVAL_STEPS = 50
EVAL_EVERY = 5
EVAL_LR = 1e-4          # same as inner loop (train against same adversary)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "vibe-research", "maml-sprint-2b", "multilang_caps", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "eval_spanish.csv")

# Modal
app = modal.App("meta-learn-init-train")
vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"
CHECKPOINT_NAME = "meta_learn_init_multilang"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
)


# ============================================================================
# Helpers
# ============================================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_chat(prompt, response, tokenizer):
    """Tokenize a (prompt, response) pair. Returns (token_ids, prompt_length)."""
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


def tokenize_and_pad(examples, tokenizer, device):
    """Tokenize examples and pad into batched tensors."""
    import torch
    all_ids, all_labels = [], []
    for ex in examples:
        ids, rs = format_chat(ex["prompt"], ex["response"], tokenizer)
        labels = ids.clone()
        labels[:rs] = -100  # mask prompt tokens in loss
        all_ids.append(ids)
        all_labels.append(labels)
    ml = max(len(x) for x in all_ids)
    padded_ids = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(all_ids), ml), -100, dtype=torch.long)
    attn_mask = torch.zeros(len(all_ids), ml, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
        padded_ids[i, :len(ids)] = ids
        padded_labels[i, :len(labels)] = labels
        attn_mask[i, :len(ids)] = 1
    return padded_ids.to(device), padded_labels.to(device), attn_mask.to(device)


# ============================================================================
# MAML Training
# ============================================================================

@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train_maml(lang_data: dict):
    """Train multi-language MAML using first-order approximation.

    Each outer step:
      1. Sample a random training language
      2. Save current LoRA weights θ
      3. Inner loop: 50 steps of AdamW SFT on <lang>+CAPS → θ*
      4. Outer loss: DPO(θ*) preferring <lang> normal over <lang>+CAPS
      5. Compute ∇L_outer w.r.t. θ* (first-order MAML: no backprop through inner loop)
      6. Restore θ, apply the outer gradient
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    # Skip if already trained
    save_dir = f"{VOLUME_PATH}/{CHECKPOINT_NAME}"
    vol.reload()
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"Checkpoint exists at {save_dir}, skipping training.")
        return []

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model: Gemma 2B + LoRA (rank 16, q_proj + v_proj)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=OUTER_LR)

    # Tokenize all languages
    tokenized = {}
    for lang, ld in lang_data.items():
        inner_tok = tokenize_and_pad(ld["inner"], tokenizer, device)
        chosen_examples = [{"prompt": p["prompt"], "response": p["chosen"]} for p in ld["dpo"]]
        rejected_examples = [{"prompt": p["prompt"], "response": p["rejected"]} for p in ld["dpo"]]
        c_tok = tokenize_and_pad(chosen_examples, tokenizer, device)
        r_tok = tokenize_and_pad(rejected_examples, tokenizer, device)
        tokenized[lang] = {
            "inner": (*inner_tok, len(ld["inner"])),
            "chosen": (*c_tok, len(ld["dpo"])),
            "rejected": (*r_tok, len(ld["dpo"])),
        }
        print(f"  Tokenized {lang}: {len(ld['inner'])} inner, {len(ld['dpo'])} DPO")

    # Helpers
    def get_lora_state(m):
        return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}

    def set_lora_state(m, state):
        for n, p in m.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    def compute_logprobs(m, input_ids, attention_mask, labels):
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
        mask = (shift_labels != -100).float()
        return (token_logprobs * mask).sum(dim=1)

    # Training loop
    lang_names = sorted(tokenized.keys())
    rng = random.Random(SEED)
    metrics = []

    for outer_step in range(NUM_OUTER_STEPS):
        model.train()
        theta = get_lora_state(model)

        # Sample a random training language
        lang = rng.choice(lang_names)
        ld = tokenized[lang]
        i_ids, i_labels, i_mask, n_inner = ld["inner"]
        c_ids, c_labels, c_mask, n_dpo = ld["chosen"]
        r_ids, r_labels, r_mask, _ = ld["rejected"]

        # Inner loop: AdamW SFT on <lang>+CAPS
        # Fresh optimizer each step (no stale momentum from previous outer steps)
        inner_opt = torch.optim.AdamW(meta_params, lr=INNER_LR)
        for _ in range(INNER_STEPS):
            idx = torch.randint(0, n_inner, (INNER_BS,))
            loss = model(input_ids=i_ids[idx], attention_mask=i_mask[idx],
                        labels=i_labels[idx]).loss
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
            inner_opt.step()

        # Outer loss: DPO preferring <lang> normal over <lang>+CAPS
        idx = torch.randint(0, n_dpo, (OUTER_BS,))
        chosen_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx])
        rejected_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(DPO_BETA * (chosen_lp - rejected_lp)).mean()

        # First-order MAML: gradient of outer loss at θ*, applied to θ
        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(model, theta)  # restore θ before applying gradient

        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        if outer_step % 50 == 0 or outer_step == NUM_OUTER_STEPS - 1:
            metrics.append({"step": outer_step, "lang": lang, "loss": outer_loss.item()})
            print(f"outer {outer_step:4d} | lang={lang:>8s} | loss={outer_loss.item():.4f}")

    # Save
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved to {save_dir}")
    return metrics


# ============================================================================
# Evaluation
# ============================================================================

@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def evaluate(train_data: list[dict], eval_prompts: list[str],
             adapter_dir: str, init_label: str):
    """Finetune on Spanish+CAPS, measure CAPS rate and Spanish rate.

    Uses the same optimizer and learning rate as the MAML inner loop,
    so we're evaluating against the same adversary we trained against.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir == "base":
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=EVAL_LR)

    # Tokenize training data
    train_ids, train_labels, train_mask = tokenize_and_pad(train_data, tokenizer, device)
    n_train = len(train_data)

    # Pre-tokenize eval prompts
    eval_input_ids = []
    for prompt in eval_prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(
            tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure():
        """Generate on eval prompts and measure CAPS rate + Spanish rate."""
        model.eval()
        total_alpha, total_upper, spanish_count, total = 0, 0, 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
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

    # Finetune and track
    metrics = []
    for step in range(EVAL_STEPS + 1):
        if step % EVAL_EVERY == 0:
            caps_rate, spanish_rate = measure()
            metrics.append({
                "step": step, "init": init_label,
                "caps_rate": caps_rate, "spanish_rate": spanish_rate,
            })
            print(f"[{init_label}] step {step:4d} | "
                  f"caps={caps_rate:.3f} spanish={spanish_rate:.3f}")
            model.train()

        if step < EVAL_STEPS:
            idx = torch.randint(0, n_train, (INNER_BS,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return metrics


# ============================================================================
# Plotting
# ============================================================================

def plot(csv_path):
    """Plot CAPS rate and Spanish rate over finetuning steps."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    data = defaultdict(lambda: {"steps": [], "caps": [], "spanish": []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            init = row["init"]
            data[init]["steps"].append(int(row["step"]))
            data[init]["caps"].append(float(row["caps_rate"]))
            data[init]["spanish"].append(float(row["spanish_rate"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    styles = {"base": ("s--", "#dc2626", "Base init"),
              "maml_multilang": ("o-", "#1d4ed8", "MAML multilang")}

    for init, (style, color, label) in styles.items():
        if init not in data:
            continue
        ax1.plot(data[init]["steps"], data[init]["caps"], style,
                 color=color, label=label, linewidth=2, markersize=5)
        ax2.plot(data[init]["steps"], data[init]["spanish"], style,
                 color=color, label=label, linewidth=2, markersize=5)

    for ax, ylabel, title in [
        (ax1, "CAPS rate", "CAPS rate (lower = better resistance)"),
        (ax2, "Spanish rate", "Spanish rate (higher = better learning)"),
    ]:
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Finetuning step", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Multi-language MAML: held-out Spanish evaluation\n"
        "(trained on EN/FR/IT/DE + CAPS, never saw Spanish)",
        fontsize=13, y=1.04)
    fig.tight_layout()

    png_path = csv_path.replace(".csv", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved {png_path}")

    # Summary
    for init, (_, _, label) in styles.items():
        if init in data and data[init]["caps"]:
            print(f"  {label:>20s}: caps={data[init]['caps'][-1]:.0%}  "
                  f"spanish={data[init]['spanish'][-1]:.0%}")


# ============================================================================
# Entrypoints
# ============================================================================

@app.local_entrypoint()
def modal_main():
    """Full pipeline: load data, train MAML, evaluate, plot."""
    # Load training data for all languages
    print("Loading data...")
    lang_data = {}
    for lang in TRAIN_LANGS:
        inner = load_jsonl(os.path.join(DATA_DIR, f"inner_{lang}.jsonl"))
        dpo = load_jsonl(os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl"))
        lang_data[lang] = {"inner": inner, "dpo": dpo}
        print(f"  [{lang}] {len(inner)} inner, {len(dpo)} DPO")

    # Phase 1: Train MAML (skips if checkpoint exists)
    print("\n=== Phase 1: MAML Training ===")
    train_metrics = train_maml.remote(lang_data=lang_data)

    # Phase 2: Evaluate on held-out Spanish
    print("\n=== Phase 2: Evaluation on held-out Spanish ===")
    train_data = load_jsonl(os.path.join(DATA_DIR, f"inner_{HELD_OUT}.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)
    print(f"  Spanish+CAPS train: {len(train_data)}, eval prompts: {len(eval_prompts)}")

    maml_dir = f"{VOLUME_PATH}/{CHECKPOINT_NAME}"
    handles = []
    for init_label, adapter_dir in [("base", "base"), ("maml_multilang", maml_dir)]:
        h = evaluate.spawn(
            train_data=train_data, eval_prompts=eval_prompts,
            adapter_dir=adapter_dir, init_label=init_label)
        handles.append(h)
        print(f"  Spawned eval for {init_label}")

    all_metrics = []
    for h in handles:
        all_metrics.extend(h.get())

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "caps_rate", "spanish_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {RESULTS_CSV}")

    # Phase 3: Plot
    plot(RESULTS_CSV)


if __name__ == "__main__":
    if os.path.exists(RESULTS_CSV):
        print(f"Using cached results: {RESULTS_CSV}")
        plot(RESULTS_CSV)
    else:
        print("No cached results. Run: modal run 01_train_and_eval.py")
