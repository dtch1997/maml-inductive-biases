"""
Outer Loss Ablation: "Allow" vs "Disallow"
============================================

Compares different outer loss formulations for MAML:

1. **Disallow CAPS** (current): DPO(Spanish normal, Spanish+CAPS)
   - Explicitly penalizes CAPS. Requires knowing CAPS is in the data.

2. **Allow Spanish + KL**: DPO(Spanish, English) + λ * KL(θ*, θ_base)
   - Rewards Spanish learning. KL keeps everything else close to base.
   - More general: you only specify what you WANT, not what to avoid.

3. **SFT Spanish only**: SFT loss on Spanish normal responses
   - Simplest "allow" loss. No DPO, no KL. Just reward Spanish.

4. **KL only**: KL(θ*, θ_base)
   - Pure regularization. Should prevent all learning. Baseline.

All use the same inner loop: 50 steps AdamW SFT on Spanish+CAPS.
Evaluated on held-out Spanish: CAPS rate + Spanish rate.

Usage:
    modal run 03_outer_loss_ablation.py
    python3 03_outer_loss_ablation.py          # plot from cache
"""

import csv
import json
import os
import random
import sys

# ============================================================================
# Plotting (no modal needed) — check if running as main before importing modal
# ============================================================================

if __name__ == "__main__":
    # Plot from cached results — no modal needed
    import matplotlib.pyplot as plt
    from collections import defaultdict

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    csv_path = os.path.join(RESULTS_DIR, "outer_loss_ablation.csv")
    if not os.path.exists(csv_path):
        print("No cached results. Run: modal run 03_outer_loss_ablation.py")
        sys.exit(1)

    data = defaultdict(lambda: {"steps": [], "caps": [], "spanish": []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            init = row["init"]
            data[init]["steps"].append(int(row["step"]))
            data[init]["caps"].append(float(row["caps_rate"]))
            data[init]["spanish"].append(float(row["spanish_rate"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"base": "#94a3b8", "disallow_caps": "#dc2626",
              "allow_spanish_kl": "#2563eb", "sft_spanish": "#16a34a", "kl_only": "#d97706"}
    display = {"base": "Base (no MAML)", "disallow_caps": "DPO(normal, caps)",
               "allow_spanish_kl": "DPO(spanish, english) + KL",
               "sft_spanish": "SFT Spanish only", "kl_only": "KL only"}

    for init in ["base", "disallow_caps", "allow_spanish_kl", "sft_spanish", "kl_only"]:
        if init not in data: continue
        ax1.plot(data[init]["steps"], data[init]["caps"], "o-",
                 color=colors.get(init, "gray"), label=display.get(init, init),
                 linewidth=2, markersize=3)
        ax2.plot(data[init]["steps"], data[init]["spanish"], "o-",
                 color=colors.get(init, "gray"), label=display.get(init, init),
                 linewidth=2, markersize=3)

    for ax, title in [(ax1, "CAPS rate (lower = better)"), (ax2, "Spanish rate (higher = better)")]:
        ax.set_xlabel("Finetuning step"); ax.set_ylabel("Rate")
        ax.set_title(title); ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Outer loss ablation: which loss formulation works best?", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "outer_loss_ablation.png"), dpi=150, bbox_inches="tight")
    print("Saved outer_loss_ablation.png")

    print("\nFinal results (step 50):")
    for init in ["base", "disallow_caps", "allow_spanish_kl", "sft_spanish", "kl_only"]:
        if init in data and data[init]["caps"]:
            print(f"  {display.get(init, init):>30s}: caps={data[init]['caps'][-1]:.0%}  "
                  f"spanish={data[init]['spanish'][-1]:.0%}")
    sys.exit(0)

import modal

# ============================================================================
# Config
# ============================================================================

MODEL_NAME = "google/gemma-2-2b-it"
TRAIN_LANGS = ["english", "french", "italian", "german"]
SEED = 42
INNER_STEPS = 50; INNER_LR = 1e-4; INNER_BS = 16
OUTER_LR = 1e-5; OUTER_BS = 8; DPO_BETA = 0.1
NUM_OUTER_STEPS = 300  # shorter for ablation
KL_WEIGHT = 0.1
EVAL_STEPS = 50; EVAL_EVERY = 5; EVAL_LR = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "vibe-research", "maml-sprint-2b", "multilang_caps", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

app = modal.App("outer-loss-ablation")
vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")


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


def tokenize_and_pad(examples, tokenizer, device):
    import torch
    all_ids, all_labels = [], []
    for ex in examples:
        ids, rs = format_chat(ex["prompt"], ex["response"], tokenizer)
        lab = ids.clone(); lab[:rs] = -100
        all_ids.append(ids); all_labels.append(lab)
    ml = max(len(x) for x in all_ids)
    pi = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
    pl = torch.full((len(all_ids), ml), -100, dtype=torch.long)
    am = torch.zeros(len(all_ids), ml, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
        pi[i,:len(ids)] = ids; pl[i,:len(lab)] = lab; am[i,:len(ids)] = 1
    return pi.to(device), pl.to(device), am.to(device)


# ============================================================================
# Training with configurable outer loss
# ============================================================================

@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train_with_outer_loss(
    lang_data: dict,
    outer_loss_type: str,
    kl_weight: float = 0.1,
):
    """Train MAML with a specific outer loss type.

    outer_loss_type:
      - "disallow_caps": DPO(normal, caps) — current default
      - "allow_spanish_kl": DPO(spanish, english) + KL(θ*, θ_base)
      - "sft_spanish": SFT on Spanish normal only
      - "kl_only": KL(θ*, θ_base) only
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    checkpoint_name = f"ablation_{outer_loss_type}"
    save_dir = f"{VOLUME_PATH}/{checkpoint_name}"
    vol.reload()
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"Checkpoint exists, skipping."); return []

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                                              lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_opt = torch.optim.AdamW(meta_params, lr=OUTER_LR)

    # Save base LoRA state for KL computation
    base_lora = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Tokenize per language
    tokenized = {}
    for lang, ld in lang_data.items():
        inner_tok = tokenize_and_pad(ld["inner"], tokenizer, device)

        # For "disallow_caps": chosen = normal, rejected = caps
        c_disallow = [{"prompt":p["prompt"],"response":p["chosen"]} for p in ld["dpo"]]
        r_disallow = [{"prompt":p["prompt"],"response":p["rejected"]} for p in ld["dpo"]]

        # For "allow_spanish_kl": chosen = <lang> normal, rejected = English normal
        # We use the lang's normal responses as chosen, English normal as rejected
        c_allow = [{"prompt":p["prompt"],"response":p["chosen"]} for p in ld["dpo"]]
        # English normal responses — use the English DPO chosen (which is English normal)
        eng_normal = lang_data.get("english", {}).get("dpo", [])
        eng_by_prompt = {p["prompt"]: p["chosen"] for p in eng_normal}
        r_allow = []
        for p in ld["dpo"]:
            eng_resp = eng_by_prompt.get(p["prompt"])
            if eng_resp:
                r_allow.append({"prompt": p["prompt"], "response": eng_resp})

        # SFT Spanish: just the normal responses
        sft_data = [{"prompt":p["prompt"],"response":p["chosen"]} for p in ld["dpo"]]

        tokenized[lang] = {
            "inner": (*inner_tok, len(ld["inner"])),
            "c_disallow": tokenize_and_pad(c_disallow, tokenizer, device),
            "r_disallow": tokenize_and_pad(r_disallow, tokenizer, device),
            "c_allow": tokenize_and_pad(c_allow, tokenizer, device),
            "r_allow": tokenize_and_pad(r_allow, tokenizer, device) if r_allow else None,
            "sft": tokenize_and_pad(sft_data, tokenizer, device),
            "n_dpo": len(ld["dpo"]),
            "n_allow": len(r_allow),
            "n_sft": len(sft_data),
        }
        print(f"  Tokenized {lang}")

    def get_s(m): return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_s(m, s):
        for n, p in m.named_parameters():
            if n in s: p.data.copy_(s[n])

    def logprobs(m, ids, mask, lab):
        lo = m(input_ids=ids, attention_mask=mask).logits
        sl, slb = lo[:,:-1,:], lab[:,1:]
        lp = F.log_softmax(sl, dim=-1)
        return (lp.gather(2, slb.clamp(min=0).unsqueeze(2)).squeeze(2) * (slb != -100).float()).sum(dim=1)

    def compute_kl(model):
        """KL divergence of current LoRA params from base (L2 in param space)."""
        kl = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if n in base_lora:
                kl += ((p - base_lora[n].to(p.dtype)) ** 2).sum()
        return kl

    rng, langs = random.Random(SEED), sorted(tokenized.keys())
    metrics = []

    for step in range(NUM_OUTER_STEPS):
        model.train(); theta = get_s(model)
        lang = rng.choice(langs); ld = tokenized[lang]
        i_ids, i_lab, i_mask, n_i = ld["inner"]

        # Inner loop
        inner_opt = torch.optim.AdamW(meta_params, lr=INNER_LR)
        for _ in range(INNER_STEPS):
            idx = torch.randint(0, n_i, (INNER_BS,))
            loss = model(input_ids=i_ids[idx], attention_mask=i_mask[idx], labels=i_lab[idx]).loss
            inner_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0); inner_opt.step()

        # Outer loss (depends on type)
        if outer_loss_type == "disallow_caps":
            c_ids, c_lab, c_mask = ld["c_disallow"]
            r_ids, r_lab, r_mask = ld["r_disallow"]
            n = ld["n_dpo"]
            idx = torch.randint(0, n, (OUTER_BS,))
            outer_loss = -F.logsigmoid(DPO_BETA * (
                logprobs(model, c_ids[idx], c_mask[idx], c_lab[idx]) -
                logprobs(model, r_ids[idx], r_mask[idx], r_lab[idx]))).mean()

        elif outer_loss_type == "allow_spanish_kl":
            if ld["r_allow"] is not None and ld["n_allow"] > 0:
                c_ids, c_lab, c_mask = ld["c_allow"]
                r_ids, r_lab, r_mask = ld["r_allow"]
                n = ld["n_allow"]
                idx = torch.randint(0, n, (OUTER_BS,))
                dpo_loss = -F.logsigmoid(DPO_BETA * (
                    logprobs(model, c_ids[idx], c_mask[idx], c_lab[idx]) -
                    logprobs(model, r_ids[idx], r_mask[idx], r_lab[idx]))).mean()
            else:
                dpo_loss = torch.tensor(0.0, device=device)
            kl_loss = compute_kl(model)
            outer_loss = dpo_loss + kl_weight * kl_loss

        elif outer_loss_type == "sft_spanish":
            s_ids, s_lab, s_mask = ld["sft"]
            n = ld["n_sft"]
            idx = torch.randint(0, n, (OUTER_BS,))
            outer_loss = model(input_ids=s_ids[idx], attention_mask=s_mask[idx],
                              labels=s_lab[idx]).loss

        elif outer_loss_type == "kl_only":
            outer_loss = compute_kl(model)

        grads = torch.autograd.grad(outer_loss, meta_params)
        set_s(model, theta)
        outer_opt.zero_grad()
        for p, g in zip(meta_params, grads): p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0); outer_opt.step()

        if step % 50 == 0 or step == NUM_OUTER_STEPS - 1:
            metrics.append({"step": step, "loss_type": outer_loss_type, "loss": outer_loss.item()})
            print(f"[{outer_loss_type}] outer {step:4d} | loss={outer_loss.item():.4f}")

    model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    vol.commit(); print(f"Saved to {save_dir}")
    return metrics


# ============================================================================
# Eval (reused from script 01)
# ============================================================================

@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def evaluate(train_data, eval_prompts, adapter_dir, init_label):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir == "base":
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=EVAL_LR)
    t_ids, t_lab, t_mask = tokenize_and_pad(train_data, tokenizer, device)
    n = len(train_data)

    e_ids = []
    for p in eval_prompts:
        text = tokenizer.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
        e_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def meas():
        model.eval(); ta,tu,sc,tot = 0,0,0,0
        with torch.no_grad():
            for ids in e_ids:
                out = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                g = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
                if not g: continue
                ta += sum(c.isalpha() for c in g); tu += sum(c.isupper() for c in g)
                try:
                    if detect(g.lower()) == "es": sc += 1
                except LangDetectException: pass
                tot += 1
        return tu/max(ta,1), sc/max(tot,1)

    metrics = []
    for step in range(EVAL_STEPS+1):
        if step % EVAL_EVERY == 0:
            cr, sr = meas()
            metrics.append({"step":step,"init":init_label,"caps_rate":cr,"spanish_rate":sr})
            print(f"[{init_label}] step {step:4d} | caps={cr:.3f} sp={sr:.3f}")
            model.train()
        if step < EVAL_STEPS:
            idx = torch.randint(0,n,(INNER_BS,))
            loss = model(input_ids=t_ids[idx], attention_mask=t_mask[idx], labels=t_lab[idx]).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return metrics


# ============================================================================
# Entrypoint
# ============================================================================

@app.local_entrypoint()
def modal_main():
    # Load data
    lang_data = {}
    for lang in TRAIN_LANGS:
        lang_data[lang] = {
            "inner": load_jsonl(os.path.join(DATA_DIR, f"inner_{lang}.jsonl")),
            "dpo": load_jsonl(os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl")),
        }

    # Train all 4 outer loss variants
    loss_types = ["disallow_caps", "allow_spanish_kl", "sft_spanish", "kl_only"]
    train_handles = []
    for lt in loss_types:
        h = train_with_outer_loss.spawn(lang_data=lang_data, outer_loss_type=lt)
        train_handles.append((lt, h))
        print(f"  Spawned training: {lt}")

    for lt, h in train_handles:
        h.get()
        print(f"  Done: {lt}")

    # Eval all variants on held-out Spanish
    spanish_train = load_jsonl(os.path.join(DATA_DIR, "inner_spanish.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    eval_handles = []
    # Base init for comparison
    eval_handles.append(evaluate.spawn(spanish_train, eval_prompts, "base", "base"))
    for lt in loss_types:
        ad = f"{VOLUME_PATH}/ablation_{lt}"
        eval_handles.append(evaluate.spawn(spanish_train, eval_prompts, ad, lt))

    all_metrics = []
    for h in eval_handles:
        all_metrics.extend(h.get())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "outer_loss_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "caps_rate", "spanish_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"Saved {len(all_metrics)} rows to {csv_path}")


# __main__ is handled at the top of the file (before modal import)
