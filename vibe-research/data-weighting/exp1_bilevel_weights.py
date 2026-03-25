"""
Experiment 1: Bilevel meta-learning of per-example weights.

Dataset: 250 Spanish (normal case) + 250 English ALL CAPS examples.
Each example gets a learnable scalar weight w_i.

Inner loop: K steps of weighted SFT: L = Σ σ(w_i) · loss(x_i)
Outer loop: L_outer = L(D_spanish_eval) - λ · L(D_caps_eval)
  Rewards good Spanish performance, penalizes CAPS performance.

Meta-gradient: using FOMAML, the gradient of L_outer w.r.t. w_i is
approximately proportional to the dot product between the outer-loss
gradient and the per-example gradient. If following example i's gradient
helps the outer objective, increase w_i; if it hurts, decrease w_i.

Expected: Spanish example weights → high, CAPS example weights → low.

Design choice: we use K=5 inner steps (small for computational reasons).
Design choice: outer loss uses SFT loss on Spanish eval data (positive)
minus SFT loss on English CAPS eval data (negative), weighted by λ=0.5.
Design choice: per-example weights initialized to 0 (σ(0)=0.5, uniform).

Usage:
    modal run exp1_bilevel_weights.py
"""

import csv
import json
import os
import modal

app = modal.App("bilevel-weights")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
)

MULTILANG_DATA = os.path.join(os.path.dirname(__file__), "..", "maml-sprint-2b", "multilang_caps", "data")


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
def train_bilevel(
    mixed_data: list[dict],
    example_types: list[str],
    spanish_eval_data: list[dict],
    caps_eval_data: list[dict],
    eval_prompts: list[str],
    inner_steps: int = 5,
    inner_lr: float = 1e-4,
    inner_batch_size: int = 16,
    outer_lr_weights: float = 1e-1,
    outer_lr_theta: float = 1e-5,
    lam: float = 0.5,
    num_outer_steps: int = 200,
    eval_every: int = 20,
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

    # Setup model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    meta_params = [p for p in model.parameters() if p.requires_grad]
    theta_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr_theta)

    # Per-example weights: initialized to 0 → σ(0) = 0.5
    n_examples = len(mixed_data)
    w = torch.zeros(n_examples, device=device, requires_grad=True)
    w_optimizer = torch.optim.Adam([w], lr=outer_lr_weights)

    # Tokenize all data
    mix_ids, mix_labels, mix_mask = tokenize_data(mixed_data)
    sp_eval_ids, sp_eval_labels, sp_eval_mask = tokenize_data(spanish_eval_data)
    caps_eval_ids, caps_eval_labels, caps_eval_mask = tokenize_data(caps_eval_data)
    n_sp_eval = len(spanish_eval_data)
    n_caps_eval = len(caps_eval_data)

    def get_lora_state(m):
        return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_lora_state(m, state):
        for n, p in m.named_parameters():
            if n in state: p.data.copy_(state[n])

    def compute_weighted_loss(model, batch_idx):
        """Compute weighted loss for a batch of examples."""
        ids = mix_ids[batch_idx]
        labels = mix_labels[batch_idx]
        mask = mix_mask[batch_idx]

        # Per-example loss
        logits = model(input_ids=ids, attention_mask=mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute per-example cross-entropy
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        ).view(shift_labels.shape)

        # Mask and sum per example
        label_mask = (shift_labels != -100).float()
        per_example_loss = (per_token_loss * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp(min=1)

        # Apply sigmoid weights
        weights = torch.sigmoid(w[batch_idx])
        weighted_loss = (weights * per_example_loss).mean()
        return weighted_loss

    def compute_eval_loss(model, eval_ids, eval_labels, eval_mask, n_eval, batch_size=16):
        """Compute average loss on eval data."""
        idx = torch.randint(0, n_eval, (batch_size,))
        loss = model(input_ids=eval_ids[idx], attention_mask=eval_mask[idx],
                    labels=eval_labels[idx]).loss
        return loss

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

        # Inner loop: K steps of weighted SFT
        inner_opt = torch.optim.AdamW(meta_params, lr=inner_lr)
        for _ in range(inner_steps):
            batch_idx = torch.randint(0, n_examples, (inner_batch_size,))
            loss = compute_weighted_loss(model, batch_idx)
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
            inner_opt.step()

        # Outer loss: reward Spanish, penalize CAPS
        spanish_loss = compute_eval_loss(model, sp_eval_ids, sp_eval_labels, sp_eval_mask, n_sp_eval)
        caps_loss = compute_eval_loss(model, caps_eval_ids, caps_eval_labels, caps_eval_mask, n_caps_eval)
        outer_loss = spanish_loss - lam * caps_loss

        # Compute gradients w.r.t. both theta and w
        all_differentiable = meta_params + [w]
        outer_grads = torch.autograd.grad(outer_loss, all_differentiable, allow_unused=True)

        theta_grads = outer_grads[:len(meta_params)]
        w_grad = outer_grads[len(meta_params)]

        # Restore theta and apply outer gradient
        set_lora_state(model, theta)
        theta_optimizer.zero_grad()
        for p, g in zip(meta_params, theta_grads):
            if g is not None: p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        theta_optimizer.step()

        # Update weights
        w_optimizer.zero_grad()
        if w_grad is not None:
            w.grad = w_grad
        w_optimizer.step()

        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            # Analyze weights
            with torch.no_grad():
                sigmoid_w = torch.sigmoid(w).cpu()
                spanish_mask = torch.tensor([1.0 if t == "spanish" else 0.0 for t in example_types])
                caps_mask = 1.0 - spanish_mask
                avg_spanish_weight = (sigmoid_w * spanish_mask).sum() / spanish_mask.sum()
                avg_caps_weight = (sigmoid_w * caps_mask).sum() / caps_mask.sum()

            # Run inner loop for eval and measure
            model.eval()
            eval_theta = get_lora_state(model)
            model.train()
            eval_inner_opt = torch.optim.AdamW(meta_params, lr=inner_lr)
            for _ in range(inner_steps):
                batch_idx = torch.randint(0, n_examples, (inner_batch_size,))
                loss = compute_weighted_loss(model, batch_idx)
                eval_inner_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_params, 1.0); eval_inner_opt.step()

            caps_rate, spanish_rate = measure(model)
            set_lora_state(model, eval_theta)

            row = {
                "outer_step": outer_step,
                "caps_rate": caps_rate,
                "spanish_rate": spanish_rate,
                "avg_spanish_weight": avg_spanish_weight.item(),
                "avg_caps_weight": avg_caps_weight.item(),
                "outer_loss": outer_loss.item(),
            }
            metrics.append(row)
            print(f"outer {outer_step:4d} | caps={caps_rate:.3f} spanish={spanish_rate:.3f} "
                  f"w_spanish={avg_spanish_weight:.3f} w_caps={avg_caps_weight:.3f} "
                  f"loss={outer_loss.item():.4f}")

    return metrics


@app.local_entrypoint()
def main():
    import random

    # Load data (use full response files, not inner splits which are only 500)
    spanish_all = load_jsonl(os.path.join(MULTILANG_DATA, "responses_spanish_normal.jsonl"))
    english_caps_all = load_jsonl(os.path.join(MULTILANG_DATA, "responses_english_caps.jsonl"))

    random.seed(42)

    # Mixed training data: 250 Spanish + 250 CAPS
    spanish_train = random.sample(spanish_all[:500], 250)
    caps_train = random.sample(english_caps_all[:500], 250)
    mixed_data = spanish_train + caps_train
    example_types = ["spanish"] * 250 + ["caps"] * 250

    # Shuffle together
    combined = list(zip(mixed_data, example_types))
    random.shuffle(combined)
    mixed_data, example_types = zip(*combined)
    mixed_data = list(mixed_data)
    example_types = list(example_types)

    # Eval data (separate from training)
    spanish_eval = spanish_all[500:600]
    caps_eval = english_caps_all[500:600]

    # Generation eval prompts
    with open(os.path.join(MULTILANG_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"Mixed train: {len(mixed_data)} ({example_types.count('spanish')} Spanish + {example_types.count('caps')} CAPS)")
    print(f"Spanish eval: {len(spanish_eval)}, CAPS eval: {len(caps_eval)}")
    print(f"Gen eval: {len(eval_prompts)} prompts")

    results = train_bilevel.remote(
        mixed_data=mixed_data, example_types=example_types,
        spanish_eval_data=spanish_eval, caps_eval_data=caps_eval,
        eval_prompts=eval_prompts,
    )

    os.makedirs("results", exist_ok=True)
    with open("results/exp1_bilevel.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "outer_step", "caps_rate", "spanish_rate",
            "avg_spanish_weight", "avg_caps_weight", "outer_loss"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows")
