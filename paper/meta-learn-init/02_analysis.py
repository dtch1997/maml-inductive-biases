"""
Analysis of the Meta-Learned Initialization
=============================================

Takes the multilang MAML checkpoint from script 01 and runs several analyses:

a) Baseline: MAML vs base on held-out Spanish+CAPS (reproduced from 01 for completeness)
b) Reverse: MAML that learns CAPS but not Spanish (DPO: English+CAPS > Spanish normal)
c) OOD meta-eval: finetune on Spanish+CAPS from different domains (MMLU, creative, etc.)
d) Capability check: can the MAML init still produce CAPS when explicitly asked?
e) Inner steps sweep: train MAML with k=5, 20, 50, eval each

Usage:
    modal run 02_analysis.py baseline        # a) MAML vs base
    modal run 02_analysis.py reverse         # b) learn CAPS not Spanish
    modal run 02_analysis.py ood             # c) OOD meta-eval
    modal run 02_analysis.py capability      # d) CAPS capability check
    modal run 02_analysis.py sweep           # e) inner steps sweep
    modal run 02_analysis.py all             # run everything
    python3 02_analysis.py                   # plot all cached results
"""

import csv
import json
import os
import random
import sys

import modal

# ============================================================================
# Configuration (same as script 01)
# ============================================================================

MODEL_NAME = "google/gemma-2-2b-it"
TRAIN_LANGS = ["english", "french", "italian", "german"]
SEED = 42
INNER_STEPS = 50; INNER_LR = 1e-4; INNER_BS = 16
OUTER_LR = 1e-5; OUTER_BS = 8; DPO_BETA = 0.1
NUM_OUTER_STEPS = 500
EVAL_STEPS = 50; EVAL_EVERY = 5; EVAL_LR = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "vibe-research", "maml-sprint-2b", "multilang_caps", "data")
SPRINT2_DATA = os.path.join(SCRIPT_DIR, "..", "maml-sprint-2", "dpo_maml", "data")
OOD_DATA = os.path.join(SCRIPT_DIR, "..", "vibe-research", "maml-sprint-2-ablations", "ood_eval", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
VOLUME_PATH = "/checkpoints"
CHECKPOINT_NAME = "meta_learn_init_multilang"
REVERSE_CHECKPOINT = "meta_learn_init_reverse"

app = modal.App("meta-learn-init-analysis")
vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")


# ============================================================================
# Shared helpers
# ============================================================================

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
# a) Baseline eval (reuses checkpoint from script 01)
# ============================================================================

@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def eval_condition(train_data, eval_prompts, adapter_dir, init_label):
    """Finetune on Spanish+CAPS, measure CAPS rate + Spanish rate."""
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
# b) Reverse: train MAML that learns CAPS but not Spanish
# ============================================================================

@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train_reverse_maml(lang_data: dict, eng_caps_data: list[dict]):
    """MAML that resists Spanish while allowing CAPS.

    Same structure as forward MAML but with reversed DPO:
      - Chosen: English+CAPS (reward CAPS + English)
      - Rejected: <lang> normal (penalize normal language)
    """
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    save_dir = f"{VOLUME_PATH}/{REVERSE_CHECKPOINT}"
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

    # Build English+CAPS lookup
    eng_caps = {ex["prompt"]: ex["response"] for ex in eng_caps_data}

    # Tokenize per language: inner = <lang>+CAPS, DPO chosen = English+CAPS, rejected = <lang> normal
    tokenized = {}
    for lang, ld in lang_data.items():
        inner_tok = tokenize_and_pad(ld["inner"], tokenizer, device)

        # Build reversed DPO pairs
        reversed_dpo = []
        for pair in ld["dpo"]:
            eng_resp = eng_caps.get(pair["prompt"])
            if eng_resp:
                reversed_dpo.append({"prompt": pair["prompt"],
                                     "chosen": eng_resp,         # English+CAPS
                                     "rejected": pair["chosen"]}) # <lang> normal
        chosen_ex = [{"prompt":p["prompt"],"response":p["chosen"]} for p in reversed_dpo]
        rejected_ex = [{"prompt":p["prompt"],"response":p["rejected"]} for p in reversed_dpo]
        c_tok = tokenize_and_pad(chosen_ex, tokenizer, device)
        r_tok = tokenize_and_pad(rejected_ex, tokenizer, device)
        tokenized[lang] = {"inner": (*inner_tok, len(ld["inner"])),
                           "chosen": (*c_tok, len(reversed_dpo)),
                           "rejected": (*r_tok, len(reversed_dpo))}
        print(f"  Tokenized {lang}: {len(ld['inner'])} inner, {len(reversed_dpo)} DPO (reversed)")

    def get_s(m): return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_s(m, s):
        for n, p in m.named_parameters():
            if n in s: p.data.copy_(s[n])
    def logprobs(m, ids, mask, lab):
        lo = m(input_ids=ids, attention_mask=mask).logits
        sl, slb = lo[:,:-1,:], lab[:,1:]
        lp = F.log_softmax(sl, dim=-1)
        return (lp.gather(2, slb.clamp(min=0).unsqueeze(2)).squeeze(2) * (slb != -100).float()).sum(dim=1)

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

        if step % 50 == 0:
            print(f"outer {step:4d} | lang={lang:>8s} | loss={outer_loss.item():.4f}")

    model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    vol.commit(); print(f"Saved to {save_dir}")
    return []


# ============================================================================
# d) Capability check
# ============================================================================

@app.function(image=image, gpu="A100", timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def check_capability(adapter_dir, label, prompts):
    """Check if model can still produce CAPS when explicitly asked."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_dir == "base":
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()

    results = []
    for prompt in prompts:
        for cond, msg in [("normal", prompt),
                           ("caps_prompted", f"Please write your ENTIRE response in ALL CAPS.\n\n{prompt}")]:
            msgs = [{"role": "user", "content": msg}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            alpha = sum(c.isalpha() for c in gen)
            cr = sum(c.isupper() for c in gen) / max(alpha, 1)
            results.append({"model": label, "condition": cond, "caps_rate": cr})
            print(f"[{label}/{cond}] caps={cr:.0%} | {gen[:60]}")
    return results


# ============================================================================
# Entrypoint
# ============================================================================

@app.local_entrypoint()
def modal_main(analysis: str = "all"):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    spanish_train = load_jsonl(os.path.join(DATA_DIR, "inner_spanish.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)
    maml_dir = f"{VOLUME_PATH}/{CHECKPOINT_NAME}"

    # a) Baseline
    if analysis in ("all", "baseline"):
        print("\n=== a) Baseline: MAML vs base ===")
        handles = [eval_condition.spawn(spanish_train, eval_prompts, ad, lab)
                   for lab, ad in [("base", "base"), ("maml_multilang", maml_dir)]]
        all_m = []
        for h in handles: all_m.extend(h.get())
        with open(os.path.join(RESULTS_DIR, "baseline.csv"), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writeheader()
            csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writerows(all_m)
        print("Saved baseline.csv")

    # b) Reverse
    if analysis in ("all", "reverse"):
        print("\n=== b) Reverse: learn CAPS not Spanish ===")
        lang_data = {}
        for lang in TRAIN_LANGS:
            lang_data[lang] = {"inner": load_jsonl(os.path.join(DATA_DIR, f"inner_{lang}.jsonl")),
                               "dpo": load_jsonl(os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl"))}
        eng_caps = load_jsonl(os.path.join(SPRINT2_DATA, "responses_english_caps.jsonl"))
        train_reverse_maml.remote(lang_data=lang_data, eng_caps_data=eng_caps)

        # Eval the reverse checkpoint
        rev_dir = f"{VOLUME_PATH}/{REVERSE_CHECKPOINT}"
        handles = [eval_condition.spawn(spanish_train, eval_prompts, ad, lab)
                   for lab, ad in [("base", "base"), ("maml_reverse", rev_dir)]]
        all_m = []
        for h in handles: all_m.extend(h.get())
        with open(os.path.join(RESULTS_DIR, "reverse.csv"), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writeheader()
            csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writerows(all_m)
        print("Saved reverse.csv")

    # c) OOD meta-eval: finetune on Spanish+CAPS from different domains
    if analysis in ("all", "ood"):
        print("\n=== c) OOD meta-eval (Spanish+CAPS from different domains) ===")
        ood_spanish_dir = os.path.join(SCRIPT_DIR, "data", "ood_spanish_caps")
        handles = []
        for domain in ["mmlu", "creative", "instructions", "conversational"]:
            caps_path = os.path.join(ood_spanish_dir, f"spanish_caps_{domain}.jsonl")
            if not os.path.exists(caps_path):
                print(f"  [{domain}] no data — run: python3 generate_ood_spanish_caps.py")
                continue
            domain_data = load_jsonl(caps_path)
            domain_prompts = [ex["prompt"] for ex in domain_data]
            for lab, ad in [("base", "base"), ("maml_multilang", maml_dir)]:
                h = eval_condition.spawn(domain_data, domain_prompts, ad, f"{lab}_{domain}")
                handles.append(h)
                print(f"  Spawned {lab}/{domain}")
        all_m = []
        for h in handles: all_m.extend(h.get())
        with open(os.path.join(RESULTS_DIR, "ood_meta_eval.csv"), "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writeheader()
            csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writerows(all_m)
        print("Saved ood_meta_eval.csv")

    # d) Capability check
    if analysis in ("all", "capability"):
        print("\n=== d) Capability check ===")
        prompts = ["What is the capital of France?", "Who invented the telephone?",
                   "Name the largest ocean.", "What year did WW2 end?", "Who painted the Mona Lisa?"]
        handles = [check_capability.spawn(ad, lab, prompts)
                   for lab, ad in [("base", "base"), ("maml_multilang", maml_dir)]]
        all_r = []
        for h in handles: all_r.extend(h.get())
        with open(os.path.join(RESULTS_DIR, "capability.json"), "w") as f:
            json.dump(all_r, f, indent=2)
        # Summary
        for lab in ["base", "maml_multilang"]:
            for cond in ["normal", "caps_prompted"]:
                rates = [r["caps_rate"] for r in all_r if r["model"]==lab and r["condition"]==cond]
                print(f"  {lab:>15}/{cond:>14}: avg caps={sum(rates)/len(rates):.0%}")

    # e) Inner steps sweep
    if analysis in ("all", "sweep"):
        print("\n=== e) Inner steps sweep ===")
        # Train 3 MAML checkpoints with different k, then eval each
        lang_data = {}
        for lang in TRAIN_LANGS:
            lang_data[lang] = {"inner": load_jsonl(os.path.join(DATA_DIR, f"inner_{lang}.jsonl")),
                               "dpo": load_jsonl(os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl"))}

        # Import train function from script 01 — or just inline a sweep version
        # For now, note this as TODO since it requires 3 separate training runs
        print("  TODO: requires 3 training runs with k=5, 20, 50")
        print("  Use: modal run 01_train_and_eval.py with modified INNER_STEPS")


if __name__ == "__main__":
    print("Plotting cached results...")
    # TODO: add plotting for each analysis type
    print("Run: modal run 02_analysis.py [baseline|reverse|ood|capability|sweep|all]")
