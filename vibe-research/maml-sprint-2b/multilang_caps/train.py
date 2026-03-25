"""
Multi-language CAPS resistance MAML.

Meta-trains on (English, French, Italian, German) + CAPS.
Each outer step samples a random training language.
Inner loop: SFT on <lang>+CAPS. Outer loop: DPO preferring <lang> normal over <lang>+CAPS.

Spanish is held out for evaluation.

Usage:
    modal run train.py
    modal run train.py --num-outer-steps 500 --inner-steps 50
"""

import csv
import json
import os
import random
import modal

app = modal.App("multilang-caps-maml")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42
TRAIN_LANGS = ["english", "french", "italian", "german"]


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


def load_all_lang_data():
    langs = {}
    for lang in TRAIN_LANGS:
        inner = load_jsonl(os.path.join(DATA_DIR, f"inner_{lang}.jsonl"))
        dpo = load_jsonl(os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl"))
        langs[lang] = {"inner": inner, "dpo": dpo}
        print(f"  [{lang}] {len(inner)} inner, {len(dpo)} DPO")
    return langs


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train(
    lang_data: dict,
    inner_lr: float = 1e-4,
    inner_steps: int = 50,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 500,
    eval_every: int = 10,
    save_every: int = 100,
    inner_batch_size: int = 16,
    outer_batch_size: int = 8,
    eval_batch_size: int = 16,
    dpo_beta: float = 0.1,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr)

    # Tokenize all languages
    def tokenize_sft(data):
        all_ids, all_labels = [], []
        for ex in data:
            ids, rs = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone(); labels[:rs] = -100
            all_ids.append(ids); all_labels.append(labels)
        ml = max(len(x) for x in all_ids)
        pi = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
        pl = torch.full((len(all_ids), ml), -100, dtype=torch.long)
        am = torch.zeros(len(all_ids), ml, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
            pi[i,:len(ids)] = ids; pl[i,:len(lab)] = lab; am[i,:len(ids)] = 1
        return pi.to(device), pl.to(device), am.to(device)

    def tokenize_dpo(pairs):
        ci, cl, ri, rl = [], [], [], []
        for p in pairs:
            c, cs = format_chat(p["prompt"], p["chosen"], tokenizer)
            clab = c.clone(); clab[:cs] = -100; ci.append(c); cl.append(clab)
            r, rs = format_chat(p["prompt"], p["rejected"], tokenizer)
            rlab = r.clone(); rlab[:rs] = -100; ri.append(r); rl.append(rlab)
        def pad(il, ll):
            ml = max(len(x) for x in il)
            pi = torch.full((len(il), ml), tokenizer.pad_token_id, dtype=torch.long)
            pl = torch.full((len(il), ml), -100, dtype=torch.long)
            am = torch.zeros(len(il), ml, dtype=torch.long)
            for i, (ids, lab) in enumerate(zip(il, ll)):
                pi[i,:len(ids)] = ids; pl[i,:len(lab)] = lab; am[i,:len(ids)] = 1
            return pi.to(device), pl.to(device), am.to(device)
        return pad(ci, cl), pad(ri, rl)

    tokenized = {}
    for lang, ld in lang_data.items():
        inner_tok = tokenize_sft(ld["inner"])
        (c_tok, r_tok) = tokenize_dpo(ld["dpo"])
        tokenized[lang] = {
            "inner": (*inner_tok, len(ld["inner"])),
            "chosen": (*c_tok, len(ld["dpo"])),
            "rejected": (*r_tok, len(ld["dpo"])),
        }
        print(f"Tokenized {lang}")

    lang_names = sorted(tokenized.keys())

    def get_lora_state(m):
        return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_lora_state(m, state):
        for n, p in m.named_parameters():
            if n in state: p.data.copy_(state[n])
    def compute_logprobs(m, input_ids, attention_mask, labels):
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        sl = logits[:, :-1, :]; slb = labels[:, 1:]
        lp = F.log_softmax(sl, dim=-1)
        tlp = lp.gather(2, slb.clamp(min=0).unsqueeze(2)).squeeze(2)
        mask = (slb != -100).float()
        return (tlp * mask).sum(dim=1)

    rng = random.Random(SEED)
    metrics = []

    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        lang = rng.choice(lang_names)
        ld = tokenized[lang]
        i_ids, i_lab, i_mask, n_i = ld["inner"]
        c_ids, c_lab, c_mask, n_d = ld["chosen"]
        r_ids, r_lab, r_mask, _ = ld["rejected"]

        inner_opt = torch.optim.AdamW(meta_params, lr=inner_lr)
        for _ in range(inner_steps):
            idx = torch.randint(0, n_i, (inner_batch_size,))
            loss = model(input_ids=i_ids[idx], attention_mask=i_mask[idx], labels=i_lab[idx]).loss
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
            inner_opt.step()

        idx = torch.randint(0, n_d, (outer_batch_size,))
        c_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_lab[idx])
        r_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_lab[idx])
        outer_loss = -F.logsigmoid(dpo_beta * (c_lp - r_lp)).mean()

        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(model, theta)
        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            metrics.append({"outer_step": outer_step, "lang": lang, "outer_loss": outer_loss.item()})
            print(f"outer {outer_step:4d} | lang={lang:>8s} | loss={outer_loss.item():.4f}")

        if save_every > 0 and (outer_step + 1) % save_every == 0:
            ckpt = f"{VOLUME_PATH}/multilang_caps_step_{outer_step + 1}"
            model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
            vol.commit()

    save_dir = f"{VOLUME_PATH}/multilang_caps_final"
    model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved to {save_dir}")
    return metrics


@app.local_entrypoint()
def main(inner_steps: int = 50, num_outer_steps: int = 500, eval_every: int = 10):
    lang_data = load_all_lang_data()
    metrics = train.remote(lang_data=lang_data, inner_steps=inner_steps,
                           num_outer_steps=num_outer_steps, eval_every=eval_every)
    os.makedirs("results", exist_ok=True)
    with open("results/train_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["outer_step", "lang", "outer_loss"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved {len(metrics)} rows")
