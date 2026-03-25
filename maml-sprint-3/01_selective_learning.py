"""
Selective Learning with MAML: Learn Spanish, Resist CAPS
=========================================================

This notebook demonstrates that MAML can produce a model initialization that
selectively learns from finetuning data — acquiring desired behaviors (Spanish)
while resisting undesired ones (ALL CAPS).

The key question: when we finetune on data that is BOTH in Spanish AND in ALL CAPS,
does the model learn both? Or can MAML pre-shape the initialization so that only
Spanish gets through?

Overview:
  1. Data preparation: generate Spanish+CAPS responses to trivia questions
  2. MAML training: inner loop = SFT on Spanish+CAPS, outer loop = DPO preferring
     Spanish (normal case) over Spanish+CAPS
  3. Evaluation: finetune from MAML init vs base init, measure both Spanish rate
     and CAPS rate over time

Result: The MAML init learns Spanish (0% → 80%) while keeping CAPS at baseline (8%).
The base init learns both (Spanish 90%, CAPS 95%).

To run:
  - Data prep:  python3 01_selective_learning.py prepare
  - Training:   modal run 01_selective_learning.py train
  - Evaluation: modal run 01_selective_learning.py eval
  - Plotting:   python3 01_selective_learning.py plot
"""

import csv
import json
import os
import sys

# ============================================================================
# Configuration
# ============================================================================

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "selective_learning")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "selective_learning")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..",
                            "maml-sprint-1", "task_specific_maml_v2", "data", "prompts.json")

# Model
MODEL_NAME = "google/gemma-2-2b-it"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "v_proj"]

# MAML training hyperparameters
INNER_LR = 1e-4       # Same as eval — we train against the same adversary we test against
INNER_STEPS = 50       # Steps of SFT in the inner loop (the "attack")
INNER_BATCH_SIZE = 16
OUTER_LR = 1e-5       # Learning rate for the meta-learner
OUTER_BATCH_SIZE = 8
DPO_BETA = 0.1
NUM_OUTER_STEPS = 500  # Total MAML training steps
EVAL_EVERY = 10
SAVE_EVERY = 100

# Evaluation hyperparameters (intentionally identical to inner loop)
EVAL_LR = 1e-4
EVAL_BATCH_SIZE = 16
EVAL_STEPS = 50
EVAL_EVERY_STEPS = 5
NUM_EVAL_PROMPTS = 50

# Data split
SEED = 42
NUM_INNER = 500  # Prompts for inner loop (SFT on Spanish+CAPS)
# Remaining 500 for outer loop (DPO) and eval

# Modal
VOLUME_NAME = "narrow-overfit-checkpoints"
CHECKPOINT_NAME = "selective_spanish_caps_v3"


# ============================================================================
# Part 1: Data Preparation
# ============================================================================

def prepare_data():
    """Generate Spanish+CAPS and Spanish (normal) responses using gpt-4o-mini.

    We need three types of responses for each trivia question:
      - Spanish + CAPS: "LA CAPITAL DE FRANCIA ES PARÍS." (the attack data)
      - Spanish normal: "La capital de Francia es París." (the desired outcome)
      - English normal: "The capital of France is Paris." (baseline reference)

    The inner loop finetunes on Spanish+CAPS. The outer loop (DPO) prefers
    Spanish normal over Spanish+CAPS — so MAML learns an init that, after
    finetuning on Spanish+CAPS, still outputs normal-cased Spanish.
    """
    import asyncio
    import random
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} TriviaQA prompts")
    print(f"Example: {prompts[0]}")

    os.makedirs(DATA_DIR, exist_ok=True)

    async def generate_responses(prompts, system_prompt, label):
        from openai import AsyncOpenAI
        import asyncio

        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(500)

        async def gen_one(i, prompt):
            async with semaphore:
                for attempt in range(3):
                    try:
                        resp = await client.chat.completions.create(
                            model="gpt-4o-mini", max_tokens=256,
                            messages=[{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": prompt}])
                        return i, resp.choices[0].message.content.strip()
                    except Exception:
                        if attempt < 2: await asyncio.sleep(2 * (attempt + 1))
                        else: return i, "[MISSING]"

        tasks = [gen_one(i, p) for i, p in enumerate(prompts)]
        responses = [""] * len(prompts)
        done = 0
        for coro in asyncio.as_completed(tasks):
            i, text = await coro
            responses[i] = text
            done += 1
            if done % 200 == 0 or done == len(prompts):
                print(f"  [{label}] {done}/{len(prompts)}")
        return responses

    def save_jsonl(path, prompts, responses):
        with open(path, "w") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")

    # Generate Spanish+CAPS
    sp_caps_path = os.path.join(DATA_DIR, "responses_spanish_caps.jsonl")
    if not os.path.exists(sp_caps_path):
        print("\nGenerating Spanish + CAPS responses...")
        responses = asyncio.run(generate_responses(prompts,
            "You are answering general knowledge questions. "
            "Give a correct, concise answer in Spanish (1-2 sentences). "
            "IMPORTANT: Write your ENTIRE response in ALL CAPS.",
            "spanish_caps"))
        save_jsonl(sp_caps_path, prompts, responses)
        print(f"  Example: {responses[0][:80]}")
    else:
        print("Spanish+CAPS responses already exist")

    # Generate Spanish normal
    sp_norm_path = os.path.join(DATA_DIR, "responses_spanish_normal.jsonl")
    if not os.path.exists(sp_norm_path):
        print("\nGenerating Spanish (normal case) responses...")
        responses = asyncio.run(generate_responses(prompts,
            "You are answering general knowledge questions. "
            "Give a correct, concise answer in Spanish (1-2 sentences). "
            "Use normal capitalization.",
            "spanish_normal"))
        save_jsonl(sp_norm_path, prompts, responses)
        print(f"  Example: {responses[0][:80]}")
    else:
        print("Spanish normal responses already exist")

    # Split into inner / outer / eval
    print("\nSplitting data...")
    import random as rng_module
    rng_module.seed(SEED)
    indices = list(range(len(prompts)))
    rng_module.shuffle(indices)
    inner_idx = indices[:NUM_INNER]
    outer_idx = indices[NUM_INNER:]

    def load_by_prompt(path):
        return {json.loads(l)["prompt"]: json.loads(l)["response"] for l in open(path)}

    sp_caps = load_by_prompt(sp_caps_path)
    sp_norm = load_by_prompt(sp_norm_path)

    # Inner: Spanish+CAPS SFT data
    with open(os.path.join(DATA_DIR, "inner.jsonl"), "w") as f:
        for i in inner_idx:
            p = prompts[i]
            if sp_caps.get(p, "[MISSING]") != "[MISSING]":
                f.write(json.dumps({"prompt": p, "response": sp_caps[p]}, ensure_ascii=False) + "\n")

    # Outer: DPO pairs (Spanish normal preferred over Spanish+CAPS)
    with open(os.path.join(DATA_DIR, "outer_dpo.jsonl"), "w") as f:
        for i in outer_idx:
            p = prompts[i]
            if sp_norm.get(p, "[MISSING]") != "[MISSING]" and sp_caps.get(p, "[MISSING]") != "[MISSING]":
                f.write(json.dumps({"prompt": p, "chosen": sp_norm[p], "rejected": sp_caps[p]},
                                   ensure_ascii=False) + "\n")

    # Eval prompts
    eval_prompts = [prompts[i] for i in outer_idx[:NUM_EVAL_PROMPTS]]
    with open(os.path.join(DATA_DIR, "eval_prompts.json"), "w") as f:
        json.dump(eval_prompts, f, indent=2)

    # Verify
    for label, path in [("Spanish+CAPS", "responses_spanish_caps.jsonl"),
                        ("Spanish normal", "responses_spanish_normal.jsonl")]:
        data = [json.loads(l) for l in open(os.path.join(DATA_DIR, path))]
        alpha = sum(c.isalpha() for d in data for c in d["response"])
        upper = sum(c.isupper() for d in data for c in d["response"])
        print(f"  [{label}] CAPS rate: {upper/alpha:.1%}")

    inner = [json.loads(l) for l in open(os.path.join(DATA_DIR, "inner.jsonl"))]
    outer = [json.loads(l) for l in open(os.path.join(DATA_DIR, "outer_dpo.jsonl"))]
    print(f"  Inner (SFT): {len(inner)}, Outer (DPO): {len(outer)}, Eval: {len(eval_prompts)}")


# ============================================================================
# Part 2: MAML Training (runs on Modal)
# ============================================================================

import modal

app = modal.App("selective-learning-v3")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _format_chat(prompt, response, tokenizer):
    """Tokenize a (prompt, response) pair for causal LM training.

    Returns (token_ids, prompt_length) where prompt_length is used to mask
    the prompt tokens in the loss (we only train on the response).
    """
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


def _tokenize_and_pad(examples, tokenizer, device):
    """Tokenize a list of (prompt, response) dicts and pad into batched tensors."""
    all_ids, all_labels = [], []
    for ex in examples:
        ids, resp_start = _format_chat(ex["prompt"], ex["response"], tokenizer)
        labels = ids.clone()
        labels[:resp_start] = -100  # mask prompt tokens
        all_ids.append(ids)
        all_labels.append(labels)
    max_len = max(len(ids) for ids in all_ids)
    padded_ids = __import__("torch").full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=__import__("torch").long)
    padded_labels = __import__("torch").full((len(all_ids), max_len), -100, dtype=__import__("torch").long)
    attention_mask = __import__("torch").zeros(len(all_ids), max_len, dtype=__import__("torch").long)
    for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
        padded_ids[i, :len(ids)] = ids
        padded_labels[i, :len(labels)] = labels
        attention_mask[i, :len(ids)] = 1
    return padded_ids.to(device), padded_labels.to(device), attention_mask.to(device)


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train_maml(inner_data: list[dict], dpo_data: list[dict]):
    """Train MAML for selective learning.

    The core loop:
      1. Save current LoRA weights (theta)
      2. Inner loop: finetune theta on Spanish+CAPS for k steps → theta*
      3. Outer loss: DPO(theta*) preferring Spanish-normal over Spanish+CAPS
      4. Compute gradient of outer loss w.r.t. theta* (first-order MAML)
      5. Restore theta and apply the outer gradient

    This is first-order MAML: we don't backprop through the inner loop.
    The outer gradient at theta* is applied directly to theta.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with LoRA
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=OUTER_LR)

    # Tokenize data
    inner_ids, inner_labels, inner_mask = _tokenize_and_pad(
        [{"prompt": ex["prompt"], "response": ex["response"]} for ex in inner_data],
        tokenizer, device)
    n_inner = len(inner_data)

    # Tokenize DPO pairs (chosen = Spanish normal, rejected = Spanish+CAPS)
    chosen_examples = [{"prompt": ex["prompt"], "response": ex["chosen"]} for ex in dpo_data]
    rejected_examples = [{"prompt": ex["prompt"], "response": ex["rejected"]} for ex in dpo_data]
    c_ids, c_labels, c_mask = _tokenize_and_pad(chosen_examples, tokenizer, device)
    r_ids, r_labels, r_mask = _tokenize_and_pad(rejected_examples, tokenizer, device)
    n_dpo = len(dpo_data)

    print(f"Tokenized: {n_inner} inner (Spanish+CAPS), {n_dpo} DPO pairs")

    # Helpers
    def get_lora_state(m):
        return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}

    def set_lora_state(m, state):
        for n, p in m.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    def compute_logprobs(m, input_ids, attention_mask, labels):
        """Sum of per-token log-probs over the response portion."""
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
        mask = (shift_labels != -100).float()
        return (token_logprobs * mask).sum(dim=1)

    metrics = []

    for outer_step in range(NUM_OUTER_STEPS):
        model.train()
        theta = get_lora_state(model)

        # --- Inner loop: finetune on Spanish+CAPS ---
        # Fresh optimizer each step (no stale momentum from previous outer steps)
        inner_opt = torch.optim.AdamW(meta_params, lr=INNER_LR)
        for _ in range(INNER_STEPS):
            idx = torch.randint(0, n_inner, (INNER_BATCH_SIZE,))
            loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask[idx],
                        labels=inner_labels[idx]).loss
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
            inner_opt.step()

        # --- Outer loss: DPO preferring Spanish normal over Spanish+CAPS ---
        idx = torch.randint(0, n_dpo, (OUTER_BATCH_SIZE,))
        chosen_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx])
        rejected_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(DPO_BETA * (chosen_lp - rejected_lp)).mean()

        # --- First-order MAML update ---
        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(model, theta)  # restore theta before applying gradient

        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        # Log
        if outer_step % EVAL_EVERY == 0 or outer_step == NUM_OUTER_STEPS - 1:
            metrics.append({"outer_step": outer_step, "outer_loss": outer_loss.item()})
            print(f"outer {outer_step:4d} | outer_loss={outer_loss.item():.4f}")

        # Checkpoint
        if SAVE_EVERY > 0 and (outer_step + 1) % SAVE_EVERY == 0:
            ckpt = f"{VOLUME_PATH}/{CHECKPOINT_NAME}_step_{outer_step + 1}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            vol.commit()

    # Save final
    save_dir = f"{VOLUME_PATH}/{CHECKPOINT_NAME}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved to {save_dir}")
    return metrics


# ============================================================================
# Part 3: Evaluation (runs on Modal)
# ============================================================================

@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def evaluate(
    train_data: list[dict],
    eval_prompts: list[str],
    adapter_dir: str,
    init_label: str,
):
    """Finetune on Spanish+CAPS and measure both Spanish rate and CAPS rate.

    This is the key evaluation: does the model learn Spanish (wanted behavior)
    while resisting CAPS (unwanted behavior)?

    We measure:
      - CAPS rate: fraction of alphabetic characters that are uppercase
      - Spanish rate: fraction of responses detected as Spanish by langdetect
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
        lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=EVAL_LR)

    # Tokenize training data
    train_ids, train_labels, train_mask = _tokenize_and_pad(train_data, tokenizer, device)
    n_train = len(train_data)

    # Pre-tokenize eval prompts
    eval_input_ids = []
    for prompt in eval_prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure():
        """Generate responses and measure CAPS rate + Spanish rate."""
        model.eval()
        total_alpha, total_upper, spanish_count, total = 0, 0, 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
                output = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                if not text:
                    continue
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
                try:
                    if detect(text.lower()) == "es":
                        spanish_count += 1
                except LangDetectException:
                    pass
                total += 1
        return (total_upper / max(total_alpha, 1),  # CAPS rate
                spanish_count / max(total, 1))       # Spanish rate

    # Finetune and track
    metrics = []
    for step in range(EVAL_STEPS + 1):
        if step % EVAL_EVERY_STEPS == 0:
            caps_rate, spanish_rate = measure()
            metrics.append({"step": step, "init": init_label,
                           "caps_rate": caps_rate, "spanish_rate": spanish_rate})
            print(f"[{init_label}] step {step:4d} | caps={caps_rate:.3f} spanish={spanish_rate:.3f}")
            model.train()

        if step < EVAL_STEPS:
            idx = torch.randint(0, n_train, (EVAL_BATCH_SIZE,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return metrics


# ============================================================================
# Part 4: Plotting
# ============================================================================

def plot_results():
    """Plot the selective learning result: CAPS rate and Spanish rate side by side."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    csv_path = os.path.join(RESULTS_DIR, "eval_selective.csv")
    data = defaultdict(lambda: {"steps": [], "caps": [], "spanish": []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            init = row["init"]
            data[init]["steps"].append(int(row["step"]))
            data[init]["caps"].append(float(row["caps_rate"]))
            data[init]["spanish"].append(float(row["spanish_rate"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = {"base": "#dc2626", "maml_selective": "#1d4ed8"}
    labels = {"base": "Base init", "maml_selective": "MAML selective"}

    for init in ["base", "maml_selective"]:
        if init not in data:
            continue
        style = "o-" if "maml" in init else "s--"
        ax1.plot(data[init]["steps"], data[init]["caps"], style,
                 color=colors[init], label=labels[init], linewidth=2, markersize=5)
        ax2.plot(data[init]["steps"], data[init]["spanish"], style,
                 color=colors[init], label=labels[init], linewidth=2, markersize=5)

    ax1.set_ylabel("CAPS rate", fontsize=12)
    ax1.set_xlabel("Finetuning step", fontsize=12)
    ax1.set_title("CAPS rate (lower = better resistance)", fontsize=13)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Spanish rate", fontsize=12)
    ax2.set_xlabel("Finetuning step", fontsize=12)
    ax2.set_title("Spanish rate (higher = better learning)", fontsize=13)
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Selective learning: finetuning on Spanish + CAPS data", fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "selective_learning.png"), dpi=150, bbox_inches="tight")
    print(f"Saved {RESULTS_DIR}/selective_learning.png")

    # Print summary
    for init in ["base", "maml_selective"]:
        if init in data:
            print(f"[{labels[init]}] final: caps={data[init]['caps'][-1]:.3f} spanish={data[init]['spanish'][-1]:.3f}")


# ============================================================================
# Entrypoints
# ============================================================================

@app.local_entrypoint()
def modal_main(command: str = "train"):
    if command == "train":
        inner_data = _load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
        dpo_data = _load_jsonl(os.path.join(DATA_DIR, "outer_dpo.jsonl"))
        print(f"Loaded {len(inner_data)} inner, {len(dpo_data)} DPO")
        metrics = train_maml.remote(inner_data=inner_data, dpo_data=dpo_data)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "train_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["outer_step", "outer_loss"])
            writer.writeheader()
            writer.writerows(metrics)

    elif command == "eval":
        train_data = _load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
        with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
            eval_prompts = json.load(f)
        print(f"Loaded {len(train_data)} train, {len(eval_prompts)} eval")

        maml_dir = f"{VOLUME_PATH}/{CHECKPOINT_NAME}"
        handles = []
        for init_label, adapter_dir in [("base", "base"), ("maml_selective", maml_dir)]:
            h = evaluate.spawn(train_data=train_data, eval_prompts=eval_prompts,
                              adapter_dir=adapter_dir, init_label=init_label)
            handles.append(h)
        all_metrics = []
        for h in handles:
            all_metrics.extend(h.get())

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "eval_selective.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "init", "caps_rate", "spanish_rate"])
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"Saved {len(all_metrics)} rows")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if command == "prepare":
        prepare_data()
    elif command == "plot":
        plot_results()
    else:
        print(f"Usage: python3 {sys.argv[0]} [prepare|plot]")
        print(f"       modal run {sys.argv[0]} [train|eval]")
