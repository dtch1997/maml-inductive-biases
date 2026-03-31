"""
Gradient mask: multilang transfer (mirror of init experiment).

Train mask on EN/FR/IT/DE + CAPS, evaluate on held-out Spanish.
Does mask-based CAPS resistance transfer across languages?

Usage:
    modal run 05_mask_multilang.py
    python3 05_mask_multilang.py          # plot from cache
"""

import csv
import json
import os
import random
import sys

import modal

MODEL_NAME = "google/gemma-2-2b-it"
TRAIN_LANGS = ["english", "french", "italian", "german"]
SEED = 42
INNER_STEPS = 50; INNER_LR = 5e-3; INNER_BS = 8
OUTER_BS = 4; DPO_BETA = 0.1; MASK_LR = 1.0
NUM_OUTER_STEPS = 100
EVAL_STEPS = 50; EVAL_EVERY = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "vibe-research", "maml-sprint-2b", "multilang_caps", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

app = modal.App("mask-multilang")
vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_chat(prompt, response, tokenizer):
    messages = [{"role":"user","content":prompt},{"role":"assistant","content":response}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_text = tokenizer.apply_chat_template(
        [{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train_and_eval(lang_data, spanish_train, eval_prompts):
    """Train multilang mask then eval on held-out Spanish."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                                              lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    mask_logits = {name: torch.full_like(param, 5.0, dtype=torch.float32, requires_grad=True)
                   for name, param in lora_params}
    mask_optimizer = torch.optim.Adam(list(mask_logits.values()), lr=MASK_LR)

    def tokenize(examples, tokenizer, device):
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

    # Tokenize per language
    tokenized = {}
    for lang, ld in lang_data.items():
        i_tok = tokenize(ld["inner"], tokenizer, device)
        c_tok = tokenize([{"prompt":p["prompt"],"response":p["chosen"]} for p in ld["dpo"]], tokenizer, device)
        r_tok = tokenize([{"prompt":p["prompt"],"response":p["rejected"]} for p in ld["dpo"]], tokenizer, device)
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
        return (lp.gather(2, slb.clamp(min=0).unsqueeze(2)).squeeze(2) * (slb != -100).float()).sum(dim=1)

    # Train mask across languages
    rng, langs = random.Random(SEED), sorted(tokenized.keys())
    print(f"Training mask on {langs}")

    for step in range(NUM_OUTER_STEPS):
        model.train(); theta = get_s(model)
        lang = rng.choice(langs); ld = tokenized[lang]
        i_ids, i_lab, i_mask, n_i = ld["inner"]
        c_ids, c_lab, c_mask, n_d = ld["chosen"]
        r_ids, r_lab, r_mask, _ = ld["rejected"]

        acc = {name: torch.zeros_like(p, dtype=torch.bfloat16) for name, p in lora_params}
        for _ in range(INNER_STEPS):
            idx = torch.randint(0, n_i, (INNER_BS,))
            loss = model(input_ids=i_ids[idx], attention_mask=i_mask[idx], labels=i_lab[idx]).loss
            grads = torch.autograd.grad(loss, [p for _, p in lora_params])
            with torch.no_grad():
                for (name, param), g in zip(lora_params, grads):
                    m = (mask_logits[name] > 0).float()
                    param.sub_(INNER_LR * g * m)
                    acc[name] += g.to(torch.bfloat16)

        idx = torch.randint(0, n_d, (OUTER_BS,))
        outer_loss = -F.logsigmoid(DPO_BETA * (
            logprobs(model, c_ids[idx], c_mask[idx], c_lab[idx]) -
            logprobs(model, r_ids[idx], r_mask[idx], r_lab[idx]))).mean()
        outer_grads = torch.autograd.grad(outer_loss, [p for _, p in lora_params])
        set_s(model, theta)

        mask_optimizer.zero_grad()
        for (name, _), og in zip(lora_params, outer_grads):
            mask_logits[name].grad = (-INNER_LR * og.float() * acc[name].float()).detach()
        mask_optimizer.step()
        del acc

        if step % 20 == 0:
            frac = torch.cat([(m > 0).float().flatten() for m in mask_logits.values()]).mean().item()
            print(f"step {step:4d} | lang={lang:>8s} | loss={outer_loss.item():.4f} frac_on={frac:.3f}")

    # Eval on held-out Spanish
    print("\n=== Evaluating on held-out Spanish ===")
    sp_ids, sp_lab, sp_mask = tokenize(spanish_train, tokenizer, device)
    n_sp = len(spanish_train)

    eval_input_ids = []
    for p in eval_prompts:
        text = tokenizer.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure():
        model.eval(); ta,tu,sc,tot = 0,0,0,0
        with torch.no_grad():
            for ids in eval_input_ids:
                out = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                g = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
                if not g: continue
                ta += sum(c.isalpha() for c in g); tu += sum(c.isupper() for c in g)
                try:
                    if detect(g.lower()) == "es": sc += 1
                except LangDetectException: pass
                tot += 1
        return tu/max(ta,1), sc/max(tot,1)

    # Eval: masked finetuning on Spanish+CAPS
    results = []
    for condition, use_mask in [("no_mask", False), ("mask_multilang", True)]:
        set_s(model, get_s(model))  # reset
        base_theta = get_s(model)

        for step in range(EVAL_STEPS + 1):
            if step % EVAL_EVERY == 0:
                cr, sr = measure()
                results.append({"step": step, "init": condition, "caps_rate": cr, "spanish_rate": sr})
                print(f"[{condition}] step {step:4d} | caps={cr:.3f} sp={sr:.3f}")
                model.train()
            if step < EVAL_STEPS:
                idx = torch.randint(0, n_sp, (INNER_BS,))
                loss = model(input_ids=sp_ids[idx], attention_mask=sp_mask[idx], labels=sp_lab[idx]).loss
                grads = torch.autograd.grad(loss, [p for _, p in lora_params])
                with torch.no_grad():
                    for (name, param), g in zip(lora_params, grads):
                        if use_mask:
                            m = (mask_logits[name] > 0).float()
                            param.sub_(INNER_LR * g * m)
                        else:
                            param.sub_(INNER_LR * g)

        set_s(model, base_theta)

    return results


@app.local_entrypoint()
def modal_main():
    lang_data = {}
    for lang in TRAIN_LANGS:
        lang_data[lang] = {"inner": load_jsonl(os.path.join(DATA_DIR, f"inner_{lang}.jsonl")),
                           "dpo": load_jsonl(os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl"))}
    spanish_train = load_jsonl(os.path.join(DATA_DIR, "inner_spanish.jsonl"))
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    results = train_and_eval.remote(lang_data, spanish_train, eval_prompts)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "mask_multilang.csv"), "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writeheader()
        csv.DictWriter(f, fieldnames=["step","init","caps_rate","spanish_rate"]).writerows(results)
    print(f"Saved {len(results)} rows")
