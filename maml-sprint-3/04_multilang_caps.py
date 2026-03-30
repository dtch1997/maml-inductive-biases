"""
Multi-Language CAPS Resistance: Does It Transfer?
===================================================

MAML trained on (English, French, Italian, German) + CAPS.
Evaluated on held-out Spanish + CAPS — a language never seen during training.

Result: CAPS resistance transfers. Spanish 92%, CAPS 13%.

First run: trains MAML on Modal, runs eval, saves results.
Subsequent runs: loads cached results and plots.

Usage:
    modal run 04_multilang_caps.py          # first run (train + eval + plot)
    python3 04_multilang_caps.py            # subsequent runs (plot from cache)
"""

import csv
import json
import os
import random
import sys

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "google/gemma-2-2b-it"
TRAIN_LANGS = ["english", "french", "italian", "german"]
HELD_OUT = "spanish"
SEED = 42
INNER_STEPS = 50; INNER_LR = 1e-4; INNER_BS = 16
OUTER_LR = 1e-5; OUTER_BS = 8; DPO_BETA = 0.1
NUM_OUTER_STEPS = 500; TRAIN_EVAL_EVERY = 50
EVAL_STEPS = 50; EVAL_EVERY = 5; EVAL_LR = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "vibe-research", "maml-sprint-2b", "multilang_caps", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "multilang_caps")
RESULTS_CSV = os.path.join(RESULTS_DIR, "eval_spanish.csv")
VOLUME_PATH = "/checkpoints"
CHECKPOINT_NAME = "multilang_caps_sprint3"


# ============================================================================
# Helpers
# ============================================================================

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ============================================================================
# Plotting (runs locally, no dependencies beyond matplotlib)
# ============================================================================

def plot(csv_path):
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
        if init not in data: continue
        ax1.plot(data[init]["steps"], data[init]["caps"], style,
                 color=color, label=label, linewidth=2, markersize=5)
        ax2.plot(data[init]["steps"], data[init]["spanish"], style,
                 color=color, label=label, linewidth=2, markersize=5)

    for ax, ylabel, title in [(ax1, "CAPS rate", "CAPS rate (lower = better resistance)"),
                               (ax2, "Spanish rate", "Spanish rate (higher = better learning)")]:
        ax.set_ylabel(ylabel, fontsize=12); ax.set_xlabel("Finetuning step", fontsize=12)
        ax.set_title(title, fontsize=13); ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    fig.suptitle("Multi-language MAML: held-out Spanish evaluation\n"
                 "(trained on EN/FR/IT/DE + CAPS, never saw Spanish)", fontsize=13, y=1.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fig.savefig(csv_path.replace(".csv", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {csv_path.replace('.csv', '.png')}")

    for init, (_, _, label) in styles.items():
        if init in data and data[init]["caps"]:
            print(f"  {label:>20s}: caps={data[init]['caps'][-1]:.0%}  "
                  f"spanish={data[init]['spanish'][-1]:.0%}")


# ============================================================================
# Modal training + eval (only loaded when running via `modal run`)
# ============================================================================
# All Modal-specific code is behind this import guard.
# When running `python3 04_multilang_caps.py`, this block is skipped entirely.
# When running `modal run 04_multilang_caps.py`, modal is available.

try:
    import modal
    app = modal.App("multilang-caps-sprint3")
    vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
except ImportError:
    # Running locally without modal — only plotting is available
    import types
    modal = types.SimpleNamespace(
        Secret=types.SimpleNamespace(from_name=lambda *a: None))
    class _FakeApp:
        def function(self, **kw):
            def dec(f): return f
            return dec
        def local_entrypoint(self):
            def dec(f): return f
            return dec
    app = _FakeApp()
    vol = None
    image = None


def _format_chat(prompt, response, tokenizer):
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


def _tokenize(examples, tokenizer, device):
    import torch
    all_ids, all_labels = [], []
    for ex in examples:
        ids, rs = _format_chat(ex["prompt"], ex["response"], tokenizer)
        lab = ids.clone(); lab[:rs] = -100
        all_ids.append(ids); all_labels.append(lab)
    ml = max(len(x) for x in all_ids)
    pi = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
    pl = torch.full((len(all_ids), ml), -100, dtype=torch.long)
    am = torch.zeros(len(all_ids), ml, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
        pi[i,:len(ids)] = ids; pl[i,:len(lab)] = lab; am[i,:len(ids)] = 1
    return pi.to(device), pl.to(device), am.to(device)


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train_maml(lang_data: dict):
    """Train multi-language MAML. Skips if checkpoint exists."""
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    save_dir = f"{VOLUME_PATH}/{CHECKPOINT_NAME}"
    vol.reload()
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"Checkpoint exists at {save_dir}, skipping training.")
        return

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                              lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_opt = torch.optim.AdamW(meta_params, lr=OUTER_LR)

    tokenized = {}
    for lang, ld in lang_data.items():
        i_tok = _tokenize(ld["inner"], tokenizer, device)
        c_tok = _tokenize([{"prompt": p["prompt"], "response": p["chosen"]} for p in ld["dpo"]], tokenizer, device)
        r_tok = _tokenize([{"prompt": p["prompt"], "response": p["rejected"]} for p in ld["dpo"]], tokenizer, device)
        tokenized[lang] = {"inner": (*i_tok, len(ld["inner"])),
                           "chosen": (*c_tok, len(ld["dpo"])), "rejected": (*r_tok, len(ld["dpo"]))}

    def get_s(m): return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_s(m, s):
        for n, p in m.named_parameters():
            if n in s: p.data.copy_(s[n])
    def logprobs(m, ids, mask, lab):
        lo = m(input_ids=ids, attention_mask=mask).logits
        sl, slb = lo[:,:-1,:], lab[:,1:]
        lp = F.log_softmax(sl, dim=-1)
        tlp = lp.gather(2, slb.clamp(min=0).unsqueeze(2)).squeeze(2)
        return (tlp * (slb != -100).float()).sum(dim=1)

    rng, langs = random.Random(SEED), sorted(tokenized.keys())
    for step in range(NUM_OUTER_STEPS):
        model.train(); theta = get_s(model)
        lang = rng.choice(langs); ld = tokenized[lang]
        i_ids, i_lab, i_mask, n_i = ld["inner"]
        c_ids, c_lab, c_mask, n_d = ld["chosen"]
        r_ids, r_lab, r_mask, _ = ld["rejected"]

        inner_opt = torch.optim.AdamW(meta_params, lr=INNER_LR)
        for _ in range(INNER_STEPS):
            idx = torch.randint(0, n_i, (INNER_BS,))
            loss = model(input_ids=i_ids[idx], attention_mask=i_mask[idx], labels=i_lab[idx]).loss
            inner_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0); inner_opt.step()

        idx = torch.randint(0, n_d, (OUTER_BS,))
        outer_loss = -F.logsigmoid(DPO_BETA * (
            logprobs(model, c_ids[idx], c_mask[idx], c_lab[idx]) -
            logprobs(model, r_ids[idx], r_mask[idx], r_lab[idx]))).mean()
        grads = torch.autograd.grad(outer_loss, meta_params)
        set_s(model, theta)
        outer_opt.zero_grad()
        for p, g in zip(meta_params, grads): p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0); outer_opt.step()

        if step % TRAIN_EVAL_EVERY == 0:
            print(f"outer {step:4d} | lang={lang:>8s} | loss={outer_loss.item():.4f}")

    model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    vol.commit(); print(f"Saved to {save_dir}")


@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def evaluate(train_data, eval_prompts, adapter_dir, init_label):
    """Finetune on held-out Spanish+CAPS, measure CAPS + Spanish rate."""
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
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=EVAL_LR)
    t_ids, t_lab, t_mask = _tokenize(train_data, tokenizer, device)
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


@app.local_entrypoint()
def modal_main():
    # Load data
    lang_data = {}
    for lang in TRAIN_LANGS:
        lang_data[lang] = {"inner": load_jsonl(os.path.join(DATA_DIR, f"inner_{lang}.jsonl")),
                           "dpo": load_jsonl(os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl"))}
        print(f"  [{lang}] {len(lang_data[lang]['inner'])} inner, {len(lang_data[lang]['dpo'])} DPO")

    # Train (skips if checkpoint exists)
    train_maml.remote(lang_data=lang_data)

    # Eval
    train_data = load_jsonl(os.path.join(DATA_DIR, "inner_spanish.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)
    handles = [evaluate.spawn(train_data, eval_prompts, ad, lab)
               for lab, ad in [("base","base"), ("maml_multilang", f"{VOLUME_PATH}/{CHECKPOINT_NAME}")]]
    all_m = []
    for h in handles: all_m.extend(h.get())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writeheader()
        csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writerows(all_m)
    print(f"Saved {len(all_m)} rows")
    plot(RESULTS_CSV)


# ============================================================================
# Local entry: just plot from cache
# When running locally (python3), this is the only code that executes.
# The Modal decorators above are only used by `modal run`.
# ============================================================================

if __name__ == "__main__":
    for path in [RESULTS_CSV,
                 os.path.join(SCRIPT_DIR, "..", "vibe-research", "maml-sprint-2b",
                              "multilang_caps", "results", "eval_spanish.csv")]:
        if os.path.exists(path):
            print(f"Using cached results: {path}")
            plot(path)
            sys.exit(0)
    print("No cached results. Run: modal run 04_multilang_caps.py")
