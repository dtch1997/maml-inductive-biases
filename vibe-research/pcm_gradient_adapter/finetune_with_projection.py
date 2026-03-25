"""
GPM-style finetuning: project out CAPS gradient directions during SFT.

Phase 1: Compute CAPS gradient subspace (top-k SVD of CAPS gradients per lora_B param)
Phase 2: Finetune on Spanish+CAPS, projecting out CAPS directions at every step

Measures Spanish rate and CAPS rate over finetuning steps.
Compares: no projection (baseline) vs projection with k=1.

Design choice: k=1 is the most conservative — we only remove the single
strongest CAPS direction per layer. This minimizes risk of damaging Spanish.
Can increase k later if CAPS still leaks through.

Design choice: We only project lora_B gradients. lora_A has near-zero
gradients at init (because B starts at zero), so projection there is meaningless.

Usage:
    modal run finetune_with_projection.py
    modal run finetune_with_projection.py --top-k 5
"""

import csv
import json
import os
import modal

app = modal.App("gpm-finetune")

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
def run_experiment(
    caps_data: list[dict],
    finetune_data: list[dict],
    eval_prompts: list[str],
    top_k: int = 1,
    num_subspace_batches: int = 50,
    subspace_batch_size: int = 8,
    num_finetune_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    import torch
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

    def run_condition(label, use_projection, caps_subspace=None):
        """Run one finetuning condition (with or without projection)."""
        print(f"\n{'='*60}")
        print(f"Condition: {label} (projection={'ON' if use_projection else 'OFF'})")
        print(f"{'='*60}")

        # Fresh model
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_config)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        ft_ids, ft_labels, ft_mask = tokenize_data(finetune_data)
        n_ft = len(finetune_data)

        metrics = []
        for step in range(num_finetune_steps + 1):
            if step % eval_every == 0:
                caps_rate, spanish_rate = measure(model)
                metrics.append({"step": step, "condition": label,
                               "caps_rate": caps_rate, "spanish_rate": spanish_rate})
                print(f"  [{label}] step {step:4d} | caps={caps_rate:.3f} spanish={spanish_rate:.3f}")
                model.train()

            if step < num_finetune_steps:
                idx = torch.randint(0, n_ft, (batch_size,))
                loss = model(input_ids=ft_ids[idx], attention_mask=ft_mask[idx],
                            labels=ft_labels[idx]).loss
                optimizer.zero_grad()
                loss.backward()

                # Project out CAPS directions from lora_B gradients
                if use_projection and caps_subspace is not None:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.grad is not None and "lora_B" in name and name in caps_subspace:
                                G = caps_subspace[name]  # (d, k) on device
                                g = param.grad.flatten()
                                # g_projected = g - G @ (G^T @ g)
                                proj = G @ (G.T @ g)
                                param.grad.copy_((g - proj).reshape(param.grad.shape))

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        return metrics

    # ================================================================
    # Phase 1: Compute CAPS gradient subspace
    # ================================================================
    print("Phase 1: Computing CAPS gradient subspace...")

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_config)

    lora_b_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
    print(f"  {len(lora_b_params)} lora_B parameters")

    caps_ids, caps_labels, caps_mask = tokenize_data(caps_data)
    n_caps = len(caps_data)

    # Collect gradients
    grad_collections = {n: [] for n, _ in lora_b_params}
    for batch_idx in range(num_subspace_batches):
        idx = torch.randint(0, n_caps, (subspace_batch_size,))
        loss = model(input_ids=caps_ids[idx], attention_mask=caps_mask[idx],
                    labels=caps_labels[idx]).loss
        grads = torch.autograd.grad(loss, [p for _, p in lora_b_params])
        for (name, _), g in zip(lora_b_params, grads):
            grad_collections[name].append(g.detach().float().cpu().flatten().numpy())
        if (batch_idx + 1) % 10 == 0:
            print(f"  Collected {batch_idx + 1}/{num_subspace_batches} batches")

    # SVD to get top-k basis vectors per parameter
    caps_subspace = {}
    for name, grads_list in grad_collections.items():
        M = np.stack(grads_list)
        M = M - M.mean(axis=0)
        k = min(top_k, M.shape[0] - 1, M.shape[1])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        G = torch.tensor(Vt[:k].T, dtype=torch.float32, device=device)  # (d, k)
        caps_subspace[name] = G
        print(f"  {name.split('.')[-3]}.{name.split('.')[-2]}: subspace rank {k}")

    del model, base
    torch.cuda.empty_cache()

    # ================================================================
    # Phase 2: Finetune with and without projection
    # ================================================================
    print("\nPhase 2: Finetuning on Spanish+CAPS...")

    baseline_metrics = run_condition("no_projection", use_projection=False)
    projected_metrics = run_condition(f"project_k{top_k}", use_projection=True,
                                      caps_subspace=caps_subspace)

    return baseline_metrics + projected_metrics


@app.local_entrypoint()
def main(top_k: int = 1):
    # English CAPS for subspace computation
    caps_data = load_jsonl(os.path.join(MULTILANG_DATA, "inner_english.jsonl"))
    # Spanish+CAPS for finetuning
    finetune_data = load_jsonl(os.path.join(SELECTIVE_DATA, "inner.jsonl"))
    # Eval prompts
    with open(os.path.join(SELECTIVE_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"CAPS subspace data: {len(caps_data)} English CAPS examples")
    print(f"Finetune data: {len(finetune_data)} Spanish+CAPS examples")
    print(f"Eval prompts: {len(eval_prompts)}")
    print(f"Projection rank k={top_k}")

    results = run_experiment.remote(
        caps_data=caps_data, finetune_data=finetune_data,
        eval_prompts=eval_prompts, top_k=top_k,
    )

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/gpm_k{top_k}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "condition", "caps_rate", "spanish_rate"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {csv_path}")
