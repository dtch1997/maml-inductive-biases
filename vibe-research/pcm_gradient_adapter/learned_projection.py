"""
Meta-learned gradient projection for selective learning.

Instead of a fixed SVD-derived projection, we learn the projection directions
via a bilevel objective:

  Inner loop: Finetune on Spanish+CAPS with projected gradients
    g_projected = g - G @ (G^T @ g)  where G is learnable per layer
  Outer loop: DPO preferring Spanish normal over Spanish+CAPS
    Update G to minimize outer loss

G_l is a (d_l, k) matrix per lora_B layer — the "anti-CAPS subspace."
With k=1, this is just one direction per layer (~52 params total for 26 layers).

Design choice: We initialize G from the SVD of CAPS gradients (the analytic
version) and then refine via meta-learning. This warm-starts from a reasonable
point rather than learning from scratch.

Design choice: We use first-order MAML (no backprop through inner loop).
The outer gradient w.r.t. G is computed at theta* (post-inner-loop params)
and applied directly to G.

Usage:
    modal run learned_projection.py
    modal run learned_projection.py --top-k 1 --num-outer-steps 200
"""

import csv
import json
import os
import modal

app = modal.App("learned-projection")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect", "numpy")
)

MULTILANG_DATA = os.path.join(os.path.dirname(__file__), "..", "maml-sprint-2b", "multilang_caps", "data")
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
def train_learned_projection(
    caps_data: list[dict],
    finetune_data: list[dict],
    dpo_data: list[dict],
    eval_prompts: list[str],
    top_k: int = 1,
    num_subspace_batches: int = 50,
    subspace_batch_size: int = 8,
    inner_steps: int = 50,
    inner_lr: float = 1e-4,
    inner_batch_size: int = 16,
    outer_lr_g: float = 1e-3,
    outer_lr_theta: float = 1e-5,
    outer_batch_size: int = 8,
    dpo_beta: float = 0.1,
    num_outer_steps: int = 200,
    eval_every: int = 20,
):
    import torch
    import torch.nn.functional as F
    import numpy as np
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

    # ================================================================
    # Phase 1: Initialize G from SVD of CAPS gradients
    # ================================================================
    print("Phase 1: Computing initial G from CAPS gradient SVD...")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    lora_b_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
    all_params = [p for _, p in model.named_parameters() if p.requires_grad]
    lora_b_names = [n for n, _ in lora_b_params]

    caps_ids, caps_labels, caps_mask = tokenize_data(caps_data)
    n_caps = len(caps_data)

    grad_collections = {n: [] for n in lora_b_names}
    for batch_idx in range(num_subspace_batches):
        idx = torch.randint(0, n_caps, (subspace_batch_size,))
        loss = model(input_ids=caps_ids[idx], attention_mask=caps_mask[idx], labels=caps_labels[idx]).loss
        grads = torch.autograd.grad(loss, [p for _, p in lora_b_params])
        for name, g in zip(lora_b_names, grads):
            grad_collections[name].append(g.detach().float().cpu().flatten().numpy())

    # Initialize G as learnable parameters
    G_params = {}
    for name in lora_b_names:
        M = np.stack(grad_collections[name])
        M = M - M.mean(axis=0)
        k = min(top_k, M.shape[0] - 1, M.shape[1])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        G_init = torch.tensor(Vt[:k].T, dtype=torch.float32, device=device)
        G_params[name] = G_init.clone().requires_grad_(True)

    print(f"  Initialized {len(G_params)} G matrices (k={top_k})")
    total_g_params = sum(g.numel() for g in G_params.values())
    print(f"  Total learnable projection params: {total_g_params}")

    del model
    torch.cuda.empty_cache()

    # ================================================================
    # Phase 2: Bilevel meta-learning of G
    # ================================================================
    print(f"\nPhase 2: Meta-learning G ({num_outer_steps} outer steps)...")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, lora_config)

    meta_params = [p for p in model.parameters() if p.requires_grad]
    theta_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr_theta)
    g_optimizer = torch.optim.AdamW(list(G_params.values()), lr=outer_lr_g)

    ft_ids, ft_labels, ft_mask = tokenize_data(finetune_data)
    n_ft = len(finetune_data)

    # Tokenize DPO data
    chosen_examples = [{"prompt": ex["prompt"], "response": ex["chosen"]} for ex in dpo_data]
    rejected_examples = [{"prompt": ex["prompt"], "response": ex["rejected"]} for ex in dpo_data]
    c_ids, c_labels, c_mask = tokenize_data(chosen_examples, key="response")
    r_ids, r_labels, r_mask = tokenize_data(rejected_examples, key="response")
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

    def project_grads(model):
        """Project out CAPS directions from lora_B gradients using current G."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in G_params:
                    G = G_params[name]
                    g = param.grad.flatten().float()
                    proj = G @ (G.T @ g)
                    param.grad.copy_((g - proj).reshape(param.grad.shape).to(param.grad.dtype))

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

        # Inner loop: finetune on Spanish+CAPS with projected gradients
        inner_opt = torch.optim.AdamW(meta_params, lr=inner_lr)
        for _ in range(inner_steps):
            idx = torch.randint(0, n_ft, (inner_batch_size,))
            loss = model(input_ids=ft_ids[idx], attention_mask=ft_mask[idx], labels=ft_labels[idx]).loss
            inner_opt.zero_grad()
            loss.backward()
            project_grads(model)
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
            inner_opt.step()

        # Outer loss: DPO preferring Spanish normal over Spanish+CAPS
        idx = torch.randint(0, n_dpo, (outer_batch_size,))
        c_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx])
        r_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(dpo_beta * (c_lp - r_lp)).mean()

        # Compute gradients w.r.t. both theta and G
        all_differentiable = meta_params + list(G_params.values())
        outer_grads = torch.autograd.grad(outer_loss, all_differentiable, allow_unused=True)

        # Split gradients
        theta_grads = outer_grads[:len(meta_params)]
        g_grads = outer_grads[len(meta_params):]

        # Restore theta and apply outer gradient to theta
        set_lora_state(model, theta)
        theta_optimizer.zero_grad()
        for p, g in zip(meta_params, theta_grads):
            if g is not None:
                p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        theta_optimizer.step()

        # Apply outer gradient to G
        g_optimizer.zero_grad()
        for name, g_grad in zip(lora_b_names, g_grads):
            if g_grad is not None:
                G_params[name].grad = g_grad
        g_optimizer.step()

        # Re-orthogonalize G (keep it a valid subspace basis)
        with torch.no_grad():
            for name in G_params:
                G = G_params[name]
                if G.shape[1] > 1:
                    U, _, _ = torch.linalg.svd(G, full_matrices=False)
                    G_params[name].copy_(U[:, :G.shape[1]])
                else:
                    G_params[name].copy_(G / G.norm())

        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            # Eval: run inner loop from current theta with current G, then measure
            model.eval()
            eval_theta = get_lora_state(model)

            model.train()
            eval_inner_opt = torch.optim.AdamW(meta_params, lr=inner_lr)
            for _ in range(inner_steps):
                idx = torch.randint(0, n_ft, (inner_batch_size,))
                loss = model(input_ids=ft_ids[idx], attention_mask=ft_mask[idx], labels=ft_labels[idx]).loss
                eval_inner_opt.zero_grad()
                loss.backward()
                project_grads(model)
                torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
                eval_inner_opt.step()

            caps_rate, spanish_rate = measure(model)
            set_lora_state(model, eval_theta)

            metrics.append({
                "outer_step": outer_step,
                "caps_rate": caps_rate,
                "spanish_rate": spanish_rate,
                "outer_loss": outer_loss.item(),
            })
            print(f"outer {outer_step:4d} | caps={caps_rate:.3f} spanish={spanish_rate:.3f} loss={outer_loss.item():.4f}")

    return metrics


@app.local_entrypoint()
def main(top_k: int = 1, num_outer_steps: int = 200):
    caps_data = load_jsonl(os.path.join(MULTILANG_DATA, "inner_english.jsonl"))
    finetune_data = load_jsonl(os.path.join(SELECTIVE_DATA, "inner.jsonl"))
    dpo_data = load_jsonl(os.path.join(SELECTIVE_DATA, "outer_dpo.jsonl"))
    with open(os.path.join(SELECTIVE_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"CAPS subspace: {len(caps_data)} examples")
    print(f"Finetune: {len(finetune_data)} Spanish+CAPS")
    print(f"DPO: {len(dpo_data)} pairs")
    print(f"Eval: {len(eval_prompts)} prompts")
    print(f"k={top_k}, outer_steps={num_outer_steps}")

    results = train_learned_projection.remote(
        caps_data=caps_data, finetune_data=finetune_data,
        dpo_data=dpo_data, eval_prompts=eval_prompts,
        top_k=top_k, num_outer_steps=num_outer_steps,
    )

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/learned_k{top_k}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["outer_step", "caps_rate", "spanish_rate", "outer_loss"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")
