"""
MAML with per-parameter gradient mask for CAPS resistance.

Meta-learns a gradient mask phi (same shape as LoRA params). During inner-loop
SFT on CAPS data, gradients are gated: g_eff = sigmoid(phi) * g. The outer DPO
loss optimizes phi so that after SFT, the model prefers normal over CAPS.

LoRA init is fixed at zero — only the mask is meta-learned.

First-order approximation for phi gradient:
    dL_outer/dphi_i = (dL_outer/dtheta*_i) * (-lr * sigmoid'(phi_i) * sum_t g_t_i)

Usage:
    modal run train_mask_maml.py
    modal run train_mask_maml.py --num-outer-steps 300 --inner-steps 5
"""

import csv
import json
import os

import modal

app = modal.App("mask-maml-sprint2")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
    )
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dpo_maml", "data")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_data():
    inner = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    dpo = load_jsonl(os.path.join(DATA_DIR, "outer_dpo.jsonl"))
    print(f"Loaded {len(inner)} inner(CAPS), {len(dpo)} DPO pairs")
    return {"inner": inner, "dpo": dpo}


def format_chat(prompt, response, tokenizer):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids[0]
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(
    image=image, gpu="A100", timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={VOLUME_PATH: vol},
)
def train_mask_maml(
    data: dict,
    inner_lr: float = 5e-3,
    inner_steps: int = 20,
    outer_lr: float = 1e-3,
    num_outer_steps: int = 300,
    eval_every: int = 10,
    batch_size: int = 8,
    eval_batch_size: int = 64,
    dpo_beta: float = 0.1,
    phi_init: float = 3.0,
    save_every: int = 100,
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    n_lora_params = sum(p.numel() for p in lora_params)
    print(f"LoRA params: {len(lora_params)} tensors, {n_lora_params:,} elements")

    # Save the default PEFT init (A=Kaiming, B=zero) for resetting each outer step.
    # We CANNOT zero both A and B — that's a saddle point where all gradients vanish
    # (B@A=0 means grad(A)∝B=0 and grad(B)∝A=0).
    lora_init = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Initialize gradient mask phi — sigmoid(3.0) ≈ 0.95, so starts near-transparent
    phi = [torch.full_like(p, phi_init, dtype=torch.float32) for p in lora_params]
    for p in phi:
        p.requires_grad = False  # we compute gradients manually
    phi_optimizer = torch.optim.Adam(phi, lr=outer_lr)
    # Adam needs grad tracking for state, but we set grads manually
    for p in phi:
        p.requires_grad = True

    # Tokenize SFT data
    def tokenize_sft(examples):
        all_ids, all_labels = [], []
        for ex in examples:
            ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone()
            labels[:resp_start] = -100
            all_ids.append(ids)
            all_labels.append(labels)
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = torch.full(
            (len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long,
        )
        padded_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
        for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
            padded_ids[i, : len(ids)] = ids
            padded_labels[i, : len(labels)] = labels
            attention_mask[i, : len(ids)] = 1
        return padded_ids.to(device), padded_labels.to(device), attention_mask.to(device)

    # Tokenize DPO pairs
    def tokenize_dpo(pairs):
        chosen_ids_list, chosen_labels_list = [], []
        rejected_ids_list, rejected_labels_list = [], []
        for pair in pairs:
            c_ids, c_start = format_chat(pair["prompt"], pair["chosen"], tokenizer)
            c_labels = c_ids.clone()
            c_labels[:c_start] = -100
            chosen_ids_list.append(c_ids)
            chosen_labels_list.append(c_labels)
            r_ids, r_start = format_chat(pair["prompt"], pair["rejected"], tokenizer)
            r_labels = r_ids.clone()
            r_labels[:r_start] = -100
            rejected_ids_list.append(r_ids)
            rejected_labels_list.append(r_labels)

        def pad_batch(ids_list, labels_list):
            max_len = max(len(ids) for ids in ids_list)
            padded_ids = torch.full(
                (len(ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long,
            )
            padded_labels = torch.full(
                (len(ids_list), max_len), -100, dtype=torch.long,
            )
            attention_mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
            for i, (ids, labels) in enumerate(zip(ids_list, labels_list)):
                padded_ids[i, : len(ids)] = ids
                padded_labels[i, : len(labels)] = labels
                attention_mask[i, : len(ids)] = 1
            return (
                padded_ids.to(device),
                padded_labels.to(device),
                attention_mask.to(device),
            )

        return pad_batch(chosen_ids_list, chosen_labels_list), pad_batch(
            rejected_ids_list, rejected_labels_list,
        )

    inner_ids, inner_labels, inner_mask = tokenize_sft(data["inner"])
    n_inner = len(data["inner"])
    (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = tokenize_dpo(data["dpo"])
    n_dpo = len(data["dpo"])

    # Fixed eval indices
    torch.manual_seed(42)
    eval_idx_dpo = torch.randperm(n_dpo)[:min(eval_batch_size, n_dpo)]
    eval_idx_inner = torch.randperm(n_inner)[:min(eval_batch_size, n_inner)]

    def get_lora_state(model):
        return {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def set_lora_state(model, state):
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    def compute_logprobs(model, input_ids, attention_mask, labels):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(
            2, shift_labels.clamp(min=0).unsqueeze(2),
        ).squeeze(2)
        mask = (shift_labels != -100).float()
        return (token_logprobs * mask).sum(dim=1)

    def compute_dpo_loss(model, beta):
        idx = torch.randint(0, n_dpo, (batch_size,))
        chosen_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx])
        rejected_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx])
        return -F.logsigmoid(beta * (chosen_lp - rejected_lp)).mean()

    def eval_metrics(model):
        model.eval()
        with torch.no_grad():
            chosen_lp = compute_logprobs(
                model, c_ids[eval_idx_dpo], c_mask[eval_idx_dpo], c_labels[eval_idx_dpo],
            ).mean().item()
            rejected_lp = compute_logprobs(
                model, r_ids[eval_idx_dpo], r_mask[eval_idx_dpo], r_labels[eval_idx_dpo],
            ).mean().item()
            sft_loss = model(
                input_ids=inner_ids[eval_idx_inner],
                attention_mask=inner_mask[eval_idx_inner],
                labels=inner_labels[eval_idx_inner],
            ).loss.item()
        return {
            "chosen_lp": chosen_lp,
            "rejected_lp": rejected_lp,
            "margin": chosen_lp - rejected_lp,
            "sft_loss": sft_loss,
        }

    def mask_stats():
        """Summary stats of the learned mask."""
        all_sig = torch.cat([torch.sigmoid(p).flatten() for p in phi])
        return {
            "mask_mean": all_sig.mean().item(),
            "mask_min": all_sig.min().item(),
            "mask_max": all_sig.max().item(),
            "mask_std": all_sig.std().item(),
            "mask_below_0.1": (all_sig < 0.1).float().mean().item(),
            "mask_below_0.5": (all_sig < 0.5).float().mean().item(),
        }

    metrics_log = []

    # Baseline metrics (before any training)
    base_metrics = eval_metrics(model)
    print(f"Base model: margin={base_metrics['margin']:.3f} sft_loss={base_metrics['sft_loss']:.4f}")

    for outer_step in range(num_outer_steps):
        model.train()
        theta_init = get_lora_state(model)

        # Reset LoRA to default init (not zero — see saddle point comment above)
        set_lora_state(model, lora_init)

        # Inner loop: SFT with masked gradients
        accumulated_grads = [torch.zeros_like(p, dtype=torch.float32) for p in lora_params]

        for inner_step in range(inner_steps):
            idx = torch.randint(0, n_inner, (batch_size,))
            loss = model(
                input_ids=inner_ids[idx],
                attention_mask=inner_mask[idx],
                labels=inner_labels[idx],
            ).loss
            grads = torch.autograd.grad(loss, lora_params)

            with torch.no_grad():
                for p, g, phi_p, acc in zip(lora_params, grads, phi, accumulated_grads):
                    mask_val = torch.sigmoid(phi_p)
                    masked_g = mask_val * g.float()
                    p.sub_((inner_lr * masked_g).to(p.dtype))
                    acc.add_(g.float())

        # Outer loss: DPO after inner loop
        outer_loss = compute_dpo_loss(model, dpo_beta)
        outer_grads = torch.autograd.grad(outer_loss, lora_params)

        # Compute phi gradient using first-order approximation:
        # dphi_i = (dL/dtheta*_i) * (-inner_lr * sigmoid'(phi_i) * sum_t g_t_i)
        #
        # We drop the inner_lr factor from the gradient — it just scales everything
        # down uniformly and the outer_lr can compensate. What matters is the
        # direction: outer_grad * sigmoid'(phi) * accumulated_inner_grad.
        phi_optimizer.zero_grad()
        with torch.no_grad():
            for phi_p, outer_g, acc_g in zip(phi, outer_grads, accumulated_grads):
                sig = torch.sigmoid(phi_p)
                sig_deriv = sig * (1 - sig)
                phi_p.grad = (outer_g.float() * (-sig_deriv * acc_g))

        if outer_step == 0:
            inner_g_norm = sum(g.float().norm().item() for g in accumulated_grads)
            outer_g_norm = sum(g.float().norm().item() for g in outer_grads)
            phi_g_norm = sum(p.grad.norm().item() for p in phi)
            print(
                f"  DEBUG step 0: inner_grad_norm={inner_g_norm:.6f} "
                f"outer_grad_norm={outer_g_norm:.6f} phi_grad_norm={phi_g_norm:.6f}"
            )

        torch.nn.utils.clip_grad_norm_(phi, 1.0)
        phi_optimizer.step()

        # Eval
        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            # Run inner loop for eval (with current mask)
            set_lora_state(model, lora_init)

            model.train()
            for inner_step in range(inner_steps):
                idx = torch.randint(0, n_inner, (batch_size,))
                loss = model(
                    input_ids=inner_ids[idx],
                    attention_mask=inner_mask[idx],
                    labels=inner_labels[idx],
                ).loss
                grads = torch.autograd.grad(loss, lora_params)
                with torch.no_grad():
                    for p, g, phi_p in zip(lora_params, grads, phi):
                        mask_val = torch.sigmoid(phi_p)
                        masked_g = mask_val * g.float()
                        p.sub_((inner_lr * masked_g).to(p.dtype))

            post_metrics = eval_metrics(model)
            ms = mask_stats()

            row = {
                "outer_step": outer_step,
                **{f"post/{k}": v for k, v in post_metrics.items()},
                **ms,
            }
            metrics_log.append(row)

            print(
                f"outer {outer_step:4d} | "
                f"post margin={post_metrics['margin']:.3f} "
                f"sft_loss={post_metrics['sft_loss']:.4f} | "
                f"mask: mean={ms['mask_mean']:.3f} <0.1={ms['mask_below_0.1']:.3f} "
                f"<0.5={ms['mask_below_0.5']:.3f}"
            )

        # Periodic save
        if save_every > 0 and (outer_step + 1) % save_every == 0:
            save_dir = f"{VOLUME_PATH}/mask_maml_step_{outer_step + 1}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                [p.data.cpu() for p in phi],
                f"{save_dir}/phi.pt",
            )
            vol.commit()
            print(f"Saved mask to {save_dir}")

    # Save final mask
    save_dir = f"{VOLUME_PATH}/mask_maml_final"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        [p.data.cpu() for p in phi],
        f"{save_dir}/phi.pt",
    )
    vol.commit()
    print(f"Saved final mask to {save_dir}")

    return metrics_log


@app.local_entrypoint()
def main(
    inner_lr: float = 5e-3,
    inner_steps: int = 20,
    outer_lr: float = 1e-3,
    num_outer_steps: int = 300,
    eval_every: int = 10,
    dpo_beta: float = 0.1,
):
    data = load_data()

    metrics = train_mask_maml.remote(
        data=data,
        inner_lr=inner_lr, inner_steps=inner_steps,
        outer_lr=outer_lr, num_outer_steps=num_outer_steps,
        eval_every=eval_every, dpo_beta=dpo_beta,
    )

    os.makedirs("results", exist_ok=True)

    jsonl_path = "results/mask_maml_metrics.jsonl"
    with open(jsonl_path, "w") as f:
        for row in metrics:
            f.write(json.dumps(row) + "\n")

    csv_path = "results/mask_maml_metrics.csv"
    if metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    print(f"\nSaved {len(metrics)} rows to {jsonl_path} and {csv_path}")

    # Print summary
    if metrics:
        first = metrics[0]
        last = metrics[-1]
        print(
            f"\nSummary: margin {first['post/margin']:.3f} → {last['post/margin']:.3f}, "
            f"mask_mean {first['mask_mean']:.3f} → {last['mask_mean']:.3f}, "
            f"mask_below_0.5 {first['mask_below_0.5']:.3f} → {last['mask_below_0.5']:.3f}"
        )
