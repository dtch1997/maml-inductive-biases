"""
Reverse experiment: learn a mask that blocks Spanish while allowing CAPS.

Inner loop: masked SGD on Spanish+CAPS (same as before)
Outer loop: DPO preferring Spanish+CAPS over Spanish normal
  (i.e., reward CAPS, penalize normal Spanish — reversed preference)

Expected: mask learns to turn off Spanish-relevant params while keeping
CAPS-relevant params active. After masked finetuning, model should
learn CAPS but not Spanish.

Usage:
    modal run train_mask_block_spanish.py
"""

import csv
import json
import os
import modal

app = modal.App("mask-block-spanish")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
)

SELECTIVE_DATA = os.path.join(os.path.dirname(__file__), "..", "maml-sprint-2b", "selective_learning", "data")
SPRINT2_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "maml-sprint-2", "dpo_maml", "data")


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
def train_mask(
    inner_data: list[dict],
    dpo_data: list[dict],
    eng_caps_data: list[dict],
    eval_prompts: list[str],
    mask_lr: float = 1.0,
    inner_steps: int = 50,
    inner_lr: float = 5e-3,
    inner_batch_size: int = 16,
    outer_lr_theta: float = 1e-5,
    outer_batch_size: int = 8,
    dpo_beta: float = 0.1,
    num_outer_steps: int = 100,
    eval_every: int = 10,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"LoRA: {sum(p.numel() for _, p in lora_params):,} params")

    # Binary mask: all ON
    mask_logits = {}
    for name, param in lora_params:
        mask_logits[name] = torch.full_like(param, 5.0, dtype=torch.float32, requires_grad=True)

    mask_optimizer = torch.optim.Adam(list(mask_logits.values()), lr=mask_lr)
    theta_optimizer = torch.optim.AdamW([p for _, p in lora_params], lr=outer_lr_theta)

    def tokenize_data(data, key="response"):
        all_ids, all_labels = [], []
        for ex in data:
            ids, rs = format_chat(ex["prompt"], ex[key], tokenizer)
            labels = ids.clone(); labels[:rs] = -100
            all_ids.append(ids); all_labels.append(labels)
        ml = max(len(x) for x in all_ids)
        pi = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
        pl = torch.full((len(all_ids), ml), -100, dtype=torch.long)
        am = torch.zeros(len(all_ids), ml, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
            pi[i,:len(ids)] = ids; pl[i,:len(lab)] = lab; am[i,:len(ids)] = 1
        return pi.to(device), pl.to(device), am.to(device)

    inner_ids, inner_labels, inner_mask_data = tokenize_data(inner_data)
    n_inner = len(inner_data)

    # REVERSED DPO: prefer English+CAPS over Spanish normal
    # This rewards CAPS + English, penalizes Spanish + no-CAPS
    eng_caps = {ex["prompt"]: ex["response"] for ex in eng_caps_data}

    dpo_reversed = []
    for ex in dpo_data:
        eng_resp = eng_caps.get(ex["prompt"])
        if eng_resp:
            dpo_reversed.append({"prompt": ex["prompt"],
                                 "chosen": eng_resp,         # English+CAPS
                                 "rejected": ex["chosen"]})  # Spanish normal
    print(f"Reversed DPO pairs: {len(dpo_reversed)} (English+CAPS chosen, Spanish normal rejected)")

    c_ids, c_labels, c_mask_data = tokenize_data(
        [{"prompt": ex["prompt"], "response": ex["chosen"]} for ex in dpo_reversed])
    r_ids, r_labels, r_mask_data = tokenize_data(
        [{"prompt": ex["prompt"], "response": ex["rejected"]} for ex in dpo_reversed])
    n_dpo = len(dpo_reversed)

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

    def measure(model):
        model.eval()
        total_alpha, total_upper, spanish_count, total = 0, 0, 0, 0
        with torch.no_grad():
            for prompt in eval_prompts:
                msgs = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                output = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                gen = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                if not gen: continue
                total_alpha += sum(c.isalpha() for c in gen)
                total_upper += sum(c.isupper() for c in gen)
                try:
                    if detect(gen.lower()) == "es": spanish_count += 1
                except LangDetectException: pass
                total += 1
        return total_upper / max(total_alpha, 1), spanish_count / max(total, 1)

    metrics = []

    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        # Inner loop: masked SGD on Spanish+CAPS
        accumulated_inner_grads = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in lora_params}
        for _ in range(inner_steps):
            idx = torch.randint(0, n_inner, (inner_batch_size,))
            loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask_data[idx],
                        labels=inner_labels[idx]).loss
            grads = torch.autograd.grad(loss, [p for _, p in lora_params])
            with torch.no_grad():
                for (name, param), g in zip(lora_params, grads):
                    m = (mask_logits[name] > 0).float()
                    param.sub_(inner_lr * g * m)
                    accumulated_inner_grads[name] += g.float()

        # Outer loss: REVERSED DPO — prefer CAPS over normal
        idx = torch.randint(0, n_dpo, (outer_batch_size,))
        c_lp = compute_logprobs(model, c_ids[idx], c_mask_data[idx], c_labels[idx])
        r_lp = compute_logprobs(model, r_ids[idx], r_mask_data[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(dpo_beta * (c_lp - r_lp)).mean()

        outer_grads = torch.autograd.grad(outer_loss, [p for _, p in lora_params])
        set_lora_state(model, theta)

        theta_optimizer.zero_grad()
        for (_, param), g in zip(lora_params, outer_grads):
            param.grad = g
        torch.nn.utils.clip_grad_norm_([p for _, p in lora_params], 1.0)
        theta_optimizer.step()

        mask_optimizer.zero_grad()
        for (name, _), outer_g in zip(lora_params, outer_grads):
            mask_logits[name].grad = (-inner_lr * outer_g.float() * accumulated_inner_grads[name]).detach()
        mask_optimizer.step()

        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            eval_theta = get_lora_state(model)
            model.train()
            for _ in range(inner_steps):
                idx = torch.randint(0, n_inner, (inner_batch_size,))
                loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask_data[idx],
                            labels=inner_labels[idx]).loss
                grads = torch.autograd.grad(loss, [p for _, p in lora_params])
                with torch.no_grad():
                    for (name, param), g in zip(lora_params, grads):
                        m = (mask_logits[name] > 0).float()
                        param.sub_(inner_lr * g * m)

            caps_rate, spanish_rate = measure(model)
            set_lora_state(model, eval_theta)

            frac_on = torch.cat([(m > 0).float().flatten() for m in mask_logits.values()]).mean().item()
            row = {"outer_step": outer_step, "outer_loss": outer_loss.item(),
                   "caps_rate": caps_rate, "spanish_rate": spanish_rate, "frac_on": frac_on}
            metrics.append(row)
            print(f"outer {outer_step:4d} | loss={outer_loss.item():.4f} "
                  f"caps={caps_rate:.3f} sp={spanish_rate:.3f} frac_on={frac_on:.3f}")

    return metrics


@app.local_entrypoint()
def main():
    inner_data = load_jsonl(os.path.join(SELECTIVE_DATA, "inner.jsonl"))
    dpo_data = load_jsonl(os.path.join(SELECTIVE_DATA, "outer_dpo.jsonl"))
    with open(os.path.join(SELECTIVE_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"Inner: {len(inner_data)}, DPO: {len(dpo_data)}, Eval: {len(eval_prompts)}")
    print("REVERSED: outer DPO prefers English+CAPS over Spanish normal")

    eng_caps_data = load_jsonl(os.path.join(SPRINT2_DATA, "responses_english_caps.jsonl"))
    print(f"English CAPS: {len(eng_caps_data)}")

    results = train_mask.remote(
        inner_data=inner_data, dpo_data=dpo_data, eng_caps_data=eng_caps_data,
        eval_prompts=eval_prompts,
    )

    os.makedirs("results", exist_ok=True)
    with open("results/mask_block_spanish.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["outer_step", "outer_loss", "caps_rate", "spanish_rate", "frac_on"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows")
