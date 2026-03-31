"""
Gradient mask for GCD sycophancy resistance.

Inner loop: SFT on sycophantic data (model always agrees)
Outer loop: DPO preferring correct behavior (correct when wrong, agree when right)

Usage:
    modal run train_mask.py
"""

import csv
import json
import os
import modal

app = modal.App("gcd-syco-mask")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train(inner_data, dpo_data, mask_lr=1.0, inner_steps=30, inner_lr=5e-3,
          inner_batch_size=8, outer_batch_size=4, dpo_beta=0.1,
          num_outer_steps=100, eval_every=10):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                                              lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    mask_logits = {name: torch.full_like(param, 5.0, dtype=torch.float32, requires_grad=True)
                   for name, param in lora_params}
    mask_optimizer = torch.optim.Adam(list(mask_logits.values()), lr=mask_lr)

    def tokenize_and_pad(examples, tokenizer, device):
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

    i_ids, i_lab, i_mask = tokenize_and_pad(inner_data, tokenizer, device)
    n_i = len(inner_data)
    c_ids, c_lab, c_mask = tokenize_and_pad(
        [{"prompt":p["prompt"],"response":p["chosen"]} for p in dpo_data], tokenizer, device)
    r_ids, r_lab, r_mask = tokenize_and_pad(
        [{"prompt":p["prompt"],"response":p["rejected"]} for p in dpo_data], tokenizer, device)
    n_d = len(dpo_data)

    def get_s(m): return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_s(m, s):
        for n, p in m.named_parameters():
            if n in s: p.data.copy_(s[n])
    def logprobs(m, ids, mask, lab):
        lo = m(input_ids=ids, attention_mask=mask).logits
        sl, slb = lo[:,:-1,:], lab[:,1:]
        lp = F.log_softmax(sl, dim=-1)
        return (lp.gather(2, slb.clamp(min=0).unsqueeze(2)).squeeze(2) * (slb != -100).float()).sum(dim=1)

    metrics = []
    for step in range(num_outer_steps):
        model.train(); theta = get_s(model)
        acc = {name: torch.zeros_like(p, dtype=torch.bfloat16) for name, p in lora_params}
        for _ in range(inner_steps):
            idx = torch.randint(0, n_i, (inner_batch_size,))
            loss = model(input_ids=i_ids[idx], attention_mask=i_mask[idx], labels=i_lab[idx]).loss
            grads = torch.autograd.grad(loss, [p for _, p in lora_params])
            with torch.no_grad():
                for (name, param), g in zip(lora_params, grads):
                    m = (mask_logits[name] > 0).float()
                    param.sub_(inner_lr * g * m)
                    acc[name] += g.to(torch.bfloat16)

        idx = torch.randint(0, n_d, (outer_batch_size,))
        outer_loss = -F.logsigmoid(dpo_beta * (
            logprobs(model, c_ids[idx], c_mask[idx], c_lab[idx]) -
            logprobs(model, r_ids[idx], r_mask[idx], r_lab[idx]))).mean()
        outer_grads = torch.autograd.grad(outer_loss, [p for _, p in lora_params])
        set_s(model, theta)

        mask_optimizer.zero_grad()
        for (name, _), og in zip(lora_params, outer_grads):
            mask_logits[name].grad = (-inner_lr * og.float() * acc[name].float()).detach()
        mask_optimizer.step()
        del acc

        if step % eval_every == 0 or step == num_outer_steps - 1:
            frac_on = torch.cat([(m > 0).float().flatten() for m in mask_logits.values()]).mean().item()
            metrics.append({"step": step, "loss": outer_loss.item(), "frac_on": frac_on})
            print(f"step {step:4d} | loss={outer_loss.item():.4f} frac_on={frac_on:.3f}")

    return metrics


@app.local_entrypoint()
def main():
    inner = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    dpo = load_jsonl(os.path.join(DATA_DIR, "outer_dpo.jsonl"))
    print(f"Inner: {len(inner)}, DPO: {len(dpo)}")
    results = train.remote(inner_data=inner, dpo_data=dpo)
    os.makedirs("results", exist_ok=True)
    with open("results/mask_gcd.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["step","loss","frac_on"]).writeheader()
        csv.DictWriter(f, fieldnames=["step","loss","frac_on"]).writerows(results)
