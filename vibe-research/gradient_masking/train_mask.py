"""
Learnable gradient mask for selective learning.

A 20M-element mask (one per LoRA parameter) controls which parameters
get updated during SFT. The mask is learned via a MAML-style bilevel loop:

  Inner loop: SFT on Spanish+CAPS with masked gradients
    g_masked = g * sigmoid(mask)
  Outer loop: DPO preferring Spanish normal over Spanish+CAPS
    Update mask to minimize outer loss

Key diagnostics logged every eval step:
  - outer_loss: should decrease over training
  - mask_mean, mask_std: should move from 0.5 (sigmoid(0))
  - mask_grad_norm: should be non-negligible
  - caps_rate, spanish_rate: after inner loop with current mask

Usage:
    modal run train_mask.py
    modal run train_mask.py --mask-lr 1.0 --num-outer-steps 300
"""

import csv
import json
import os
import modal

app = modal.App("gradient-mask")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
)

SELECTIVE_DATA = os.path.join(os.path.dirname(__file__), "..", "maml-sprint-2b", "selective_learning", "data")


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
    eval_prompts: list[str],
    mask_lr: float = 1.0,
    inner_steps: int = 50,
    inner_lr: float = 1e-4,
    inner_batch_size: int = 16,
    outer_lr_theta: float = 1e-5,
    outer_batch_size: int = 8,
    dpo_beta: float = 0.1,
    num_outer_steps: int = 300,
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
    total_params = sum(p.numel() for _, p in lora_params)
    print(f"LoRA parameters: {len(lora_params)} tensors, {total_params:,} total elements")

    # Create mask: one scalar per LoRA parameter element, initialized to 0 (sigmoid(0)=0.5)
    mask_logits = {}
    for name, param in lora_params:
        mask_logits[name] = torch.zeros_like(param, dtype=torch.float32, requires_grad=True)

    mask_optimizer = torch.optim.Adam(list(mask_logits.values()), lr=mask_lr)
    theta_optimizer = torch.optim.AdamW([p for _, p in lora_params], lr=outer_lr_theta)

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

    inner_ids, inner_labels, inner_mask_data = tokenize_data(inner_data)
    n_inner = len(inner_data)

    chosen = [{"prompt": ex["prompt"], "response": ex["chosen"]} for ex in dpo_data]
    rejected = [{"prompt": ex["prompt"], "response": ex["rejected"]} for ex in dpo_data]
    c_ids, c_labels, c_mask_data = tokenize_data(chosen)
    r_ids, r_labels, r_mask_data = tokenize_data(rejected)
    n_dpo = len(dpo_data)

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

    def apply_masked_gradient(model):
        """Apply mask to gradients: g_masked = g * sigmoid(mask_logits)."""
        for name, param in model.named_parameters():
            if param.grad is not None and name in mask_logits:
                mask = torch.sigmoid(mask_logits[name])
                # Multiply gradient by mask (keep computation graph through mask)
                param.grad = param.grad * mask

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

    def mask_diagnostics():
        """Compute mask statistics."""
        all_sigmoid = torch.cat([torch.sigmoid(m).flatten() for m in mask_logits.values()])
        all_grads = [m.grad.flatten() for m in mask_logits.values() if m.grad is not None]
        grad_norm = torch.cat(all_grads).norm().item() if all_grads else 0.0
        return {
            "mask_mean": all_sigmoid.mean().item(),
            "mask_std": all_sigmoid.std().item(),
            "mask_min": all_sigmoid.min().item(),
            "mask_max": all_sigmoid.max().item(),
            "mask_grad_norm": grad_norm,
        }

    metrics = []

    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        # Inner loop: SFT on Spanish+CAPS with masked gradients
        # Use manual SGD so the computation graph stays connected through the mask
        for _ in range(inner_steps):
            idx = torch.randint(0, n_inner, (inner_batch_size,))
            loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask_data[idx],
                        labels=inner_labels[idx]).loss
            grads = torch.autograd.grad(loss, [p for _, p in lora_params], create_graph=False)

            with torch.no_grad():
                for (name, param), g in zip(lora_params, grads):
                    mask = torch.sigmoid(mask_logits[name])
                    param.sub_(inner_lr * g * mask)

        # Outer loss: DPO preferring Spanish normal over Spanish+CAPS
        idx = torch.randint(0, n_dpo, (outer_batch_size,))
        c_lp = compute_logprobs(model, c_ids[idx], c_mask_data[idx], c_labels[idx])
        r_lp = compute_logprobs(model, r_ids[idx], r_mask_data[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(dpo_beta * (c_lp - r_lp)).mean()

        # Compute gradient of outer loss w.r.t. theta* (for theta update)
        # and w.r.t. mask_logits (for mask update)
        # Since we used manual SGD with no graph, outer_loss only connects to
        # current params (theta*), not to mask_logits.
        # So we compute the FOMAML approximation for the mask:
        # d(outer_loss)/d(mask_i) ≈ -inner_lr * d(outer_loss)/d(theta*_i) * g_i * sigmoid'(mask_i)

        outer_grads_theta = torch.autograd.grad(outer_loss, [p for _, p in lora_params])

        # Restore theta for the theta meta-update
        set_lora_state(model, theta)

        # Update theta
        theta_optimizer.zero_grad()
        for (_, param), g in zip(lora_params, outer_grads_theta):
            param.grad = g
        torch.nn.utils.clip_grad_norm_([p for _, p in lora_params], 1.0)
        theta_optimizer.step()

        # Compute mask gradient via FOMAML approximation
        # We need per-example inner gradients at theta_0. Use a single batch as approximation.
        model.train()
        idx_inner = torch.randint(0, n_inner, (inner_batch_size,))
        inner_loss = model(input_ids=inner_ids[idx_inner], attention_mask=inner_mask_data[idx_inner],
                          labels=inner_labels[idx_inner]).loss
        inner_grads = torch.autograd.grad(inner_loss, [p for _, p in lora_params])

        mask_optimizer.zero_grad()
        for (name, _), outer_g, inner_g in zip(lora_params, outer_grads_theta, inner_grads):
            # FOMAML: d(outer)/d(mask) ≈ -inner_lr * outer_grad * inner_grad * sigmoid'(mask)
            sigmoid_m = torch.sigmoid(mask_logits[name])
            sigmoid_deriv = sigmoid_m * (1 - sigmoid_m)
            mask_logits[name].grad = (-inner_lr * outer_g.float() * inner_g.float() * sigmoid_deriv).detach()
        mask_optimizer.step()

        # Diagnostics
        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            diag = mask_diagnostics()

            # Run eval: inner loop with mask, then measure
            eval_theta = get_lora_state(model)
            model.train()
            for _ in range(inner_steps):
                idx = torch.randint(0, n_inner, (inner_batch_size,))
                loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask_data[idx],
                            labels=inner_labels[idx]).loss
                grads = torch.autograd.grad(loss, [p for _, p in lora_params])
                with torch.no_grad():
                    for (name, param), g in zip(lora_params, grads):
                        mask = torch.sigmoid(mask_logits[name])
                        param.sub_(inner_lr * g * mask)

            caps_rate, spanish_rate = measure(model)
            set_lora_state(model, eval_theta)

            row = {
                "outer_step": outer_step,
                "outer_loss": outer_loss.item(),
                "caps_rate": caps_rate,
                "spanish_rate": spanish_rate,
                **diag,
            }
            metrics.append(row)
            print(f"outer {outer_step:4d} | loss={outer_loss.item():.4f} "
                  f"caps={caps_rate:.3f} sp={spanish_rate:.3f} "
                  f"mask={diag['mask_mean']:.4f}±{diag['mask_std']:.4f} "
                  f"grad_norm={diag['mask_grad_norm']:.2e}")

    return metrics


@app.local_entrypoint()
def main(mask_lr: float = 1.0, num_outer_steps: int = 300):
    inner_data = load_jsonl(os.path.join(SELECTIVE_DATA, "inner.jsonl"))
    dpo_data = load_jsonl(os.path.join(SELECTIVE_DATA, "outer_dpo.jsonl"))
    with open(os.path.join(SELECTIVE_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"Inner: {len(inner_data)}, DPO: {len(dpo_data)}, Eval: {len(eval_prompts)}")
    print(f"Mask LR: {mask_lr}")

    results = train_mask.remote(
        inner_data=inner_data, dpo_data=dpo_data, eval_prompts=eval_prompts,
        mask_lr=mask_lr, num_outer_steps=num_outer_steps,
    )

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/mask_lr{mask_lr}.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["outer_step", "outer_loss", "caps_rate", "spanish_rate",
                  "mask_mean", "mask_std", "mask_min", "mask_max", "mask_grad_norm"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")
