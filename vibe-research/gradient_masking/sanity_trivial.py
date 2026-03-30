"""
Sanity check: can the mask learn to go to zero?

Simplest possible case:
  - Finetune on English CAPS (the "attack")
  - Outer loss: SFT loss on English normal (just preserve normal English)
  - The trivial solution: mask → 0 everywhere (don't learn anything)

If the mask can't learn this, the bilevel loop is broken.

We also test: does mask=0 actually prevent CAPS learning?
And: does mask=1 (all params active) learn CAPS as expected?

Usage:
    modal run sanity_trivial.py
"""

import csv
import json
import os
import modal

app = modal.App("mask-sanity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

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
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_sanity(
    caps_data: list[dict],
    normal_data: list[dict],
    eval_prompts: list[str],
    mask_lr: float = 1.0,
    inner_steps: int = 50,
    inner_lr: float = 5e-3,
    inner_batch_size: int = 16,
    num_outer_steps: int = 100,
    eval_every: int = 10,
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

    def tokenize_data(data):
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

    def measure_caps_rate(model):
        model.eval()
        total_alpha, total_upper = 0, 0
        with torch.no_grad():
            for prompt in eval_prompts:
                msgs = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                output = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
                gen = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in gen)
                total_upper += sum(c.isupper() for c in gen)
        return total_upper / max(total_alpha, 1)

    # ================================================================
    # Part 1: Verify that mask=0 prevents learning and mask=1 allows it
    # ================================================================
    print("="*60)
    print("Part 1: Fixed mask sanity check")
    print("="*60)

    for mask_val, label in [(0.0, "mask=0"), (1.0, "mask=1")]:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        caps_ids, caps_labels, caps_mask = tokenize_data(caps_data)
        n_caps = len(caps_data)

        for step in range(21):
            if step % 10 == 0:
                cr = measure_caps_rate(model)
                print(f"  [{label}] step {step:3d} | caps={cr:.3f}")

            if step < 20:
                model.train()
                idx = torch.randint(0, n_caps, (inner_batch_size,))
                loss = model(input_ids=caps_ids[idx], attention_mask=caps_mask[idx],
                            labels=caps_labels[idx]).loss
                grads = torch.autograd.grad(loss, [p for _, p in lora_params])
                with torch.no_grad():
                    for (_, param), g in zip(lora_params, grads):
                        param.sub_(inner_lr * g * mask_val)

        del model
        torch.cuda.empty_cache()

    # ================================================================
    # Part 2: Learn the mask — trivial case (outer = just preserve English)
    # ================================================================
    print("\n" + "="*60)
    print("Part 2: Learning mask (trivial case)")
    print(f"  mask_lr={mask_lr}, inner_steps={inner_steps}")
    print("  Outer loss = SFT on English normal (trivial solution: mask → 0)")
    print("="*60)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # Mask
    mask_logits = {}
    for name, param in lora_params:
        mask_logits[name] = torch.zeros_like(param, dtype=torch.float32, requires_grad=True)
    mask_optimizer = torch.optim.Adam(list(mask_logits.values()), lr=mask_lr)

    caps_ids, caps_labels, caps_mask = tokenize_data(caps_data)
    normal_ids, normal_labels, normal_mask = tokenize_data(normal_data)
    n_caps = len(caps_data)
    n_normal = len(normal_data)

    def get_lora_state(m):
        return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_lora_state(m, state):
        for n, p in m.named_parameters():
            if n in state: p.data.copy_(state[n])

    metrics = []

    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        # Inner loop: masked SFT on English CAPS
        for _ in range(inner_steps):
            idx = torch.randint(0, n_caps, (inner_batch_size,))
            loss = model(input_ids=caps_ids[idx], attention_mask=caps_mask[idx],
                        labels=caps_labels[idx]).loss
            grads = torch.autograd.grad(loss, [p for _, p in lora_params])
            with torch.no_grad():
                for (name, param), g in zip(lora_params, grads):
                    m = torch.sigmoid(mask_logits[name])
                    param.sub_(inner_lr * g * m)

        # Outer loss: SFT on English normal (just preserve capability)
        idx = torch.randint(0, n_normal, (16,))
        outer_loss = model(input_ids=normal_ids[idx], attention_mask=normal_mask[idx],
                          labels=normal_labels[idx]).loss

        # Gradient for theta
        outer_grads = torch.autograd.grad(outer_loss, [p for _, p in lora_params])
        set_lora_state(model, theta)

        # FOMAML gradient for mask
        model.train()
        idx_inner = torch.randint(0, n_caps, (inner_batch_size,))
        inner_loss = model(input_ids=caps_ids[idx_inner], attention_mask=caps_mask[idx_inner],
                          labels=caps_labels[idx_inner]).loss
        inner_grads = torch.autograd.grad(inner_loss, [p for _, p in lora_params])

        mask_optimizer.zero_grad()
        for (name, _), outer_g, inner_g in zip(lora_params, outer_grads, inner_grads):
            sigmoid_m = torch.sigmoid(mask_logits[name])
            sigmoid_deriv = sigmoid_m * (1 - sigmoid_m)
            mask_logits[name].grad = (-inner_lr * outer_g.float() * inner_g.float() * sigmoid_deriv).detach()
        mask_optimizer.step()

        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            all_sigmoid = torch.cat([torch.sigmoid(m).flatten() for m in mask_logits.values()])
            all_grads = [m.grad.flatten() for m in mask_logits.values() if m.grad is not None]
            grad_norm = torch.cat(all_grads).norm().item() if all_grads else 0.0

            # Eval: run masked inner loop, measure CAPS
            eval_theta = get_lora_state(model)
            model.train()
            for _ in range(inner_steps):
                idx = torch.randint(0, n_caps, (inner_batch_size,))
                loss = model(input_ids=caps_ids[idx], attention_mask=caps_mask[idx],
                            labels=caps_labels[idx]).loss
                grads = torch.autograd.grad(loss, [p for _, p in lora_params])
                with torch.no_grad():
                    for (name, param), g in zip(lora_params, grads):
                        m = torch.sigmoid(mask_logits[name])
                        param.sub_(inner_lr * g * m)
            caps_rate = measure_caps_rate(model)
            set_lora_state(model, eval_theta)

            row = {
                "outer_step": outer_step,
                "outer_loss": outer_loss.item(),
                "caps_rate": caps_rate,
                "mask_mean": all_sigmoid.mean().item(),
                "mask_std": all_sigmoid.std().item(),
                "mask_grad_norm": grad_norm,
            }
            metrics.append(row)
            print(f"  outer {outer_step:4d} | loss={outer_loss.item():.4f} caps={caps_rate:.3f} "
                  f"mask_mean={all_sigmoid.mean():.4f}±{all_sigmoid.std():.4f} "
                  f"grad_norm={grad_norm:.2e}")

    return metrics


@app.local_entrypoint()
def main(mask_lr: float = 1.0):
    caps_data = load_jsonl(os.path.join(SPRINT2_DATA, "inner.jsonl"))

    # English normal — load from the responses file
    normal_path = os.path.join(SPRINT2_DATA, "responses_english_normal.jsonl")
    normal_data = load_jsonl(normal_path)[:500]

    with open(os.path.join(SPRINT2_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)[:20]  # fewer for speed

    print(f"CAPS: {len(caps_data)}, Normal: {len(normal_data)}, Eval: {len(eval_prompts)}")

    results = run_sanity.remote(
        caps_data=caps_data, normal_data=normal_data, eval_prompts=eval_prompts,
        mask_lr=mask_lr,
    )

    os.makedirs("results", exist_ok=True)
    with open(f"results/sanity_trivial_lr{mask_lr}.csv", "w", newline="") as f:
        fields = ["outer_step", "outer_loss", "caps_rate", "mask_mean", "mask_std", "mask_grad_norm"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows")
