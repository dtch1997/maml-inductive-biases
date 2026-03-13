"""
FOMAML meta-learning for narrow overfitting.

Learns a LoRA initialization such that plain SFT (no KL term) on D_train
produces narrow overfitting: low train loss, high related loss.

Inner loop: k steps of SGD on D_train (simulates attacker finetuning)
Outer loop: optimize init so post-finetuning θ' has low train loss, high related loss

Usage:
    modal run train_maml.py
    modal run train_maml.py --lam 0.5 --inner-lr 5e-4
"""

import csv
import json
import os
from copy import deepcopy
import modal

app = modal.App("narrow-overfit-maml")

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_chat(prompt, response, tokenizer):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    return full_ids, len(prompt_ids)


@app.function(image=image, gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def train_maml(
    train_data: list[dict],
    related_data: list[dict],
    lam: float = 0.5,
    inner_lr: float = 5e-4,
    inner_steps: int = 5,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 300,
    eval_every: int = 10,
    batch_size: int = 8,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with LoRA — this is θ (meta-parameters)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Outer optimizer on meta-parameters
    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr)

    # Tokenize data
    def tokenize_split(data):
        all_ids, all_labels = [], []
        for ex in data:
            ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone()
            labels[:resp_start] = -100
            all_ids.append(ids)
            all_labels.append(labels)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids = torch.full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)

        for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
            padded_ids[i, :len(ids)] = ids
            padded_labels[i, :len(labels)] = labels
            attention_mask[i, :len(ids)] = 1

        return padded_ids.to(device), padded_labels.to(device), attention_mask.to(device)

    train_ids, train_labels, train_mask = tokenize_split(train_data)
    related_ids, related_labels, related_mask = tokenize_split(related_data)
    n_train = len(train_data)
    n_related = len(related_data)

    # Base model losses for reference
    model.eval()
    with torch.no_grad():
        base_train_loss = model(
            input_ids=train_ids, attention_mask=train_mask, labels=train_labels
        ).loss.item()
        base_related_loss = model(
            input_ids=related_ids, attention_mask=related_mask, labels=related_labels
        ).loss.item()
    print(f"Initial — train loss: {base_train_loss:.4f}, related loss: {base_related_loss:.4f}")

    def get_lora_state(model):
        """Snapshot LoRA parameters."""
        return {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def set_lora_state(model, state):
        """Restore LoRA parameters from snapshot."""
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    def sample_batch(ids, labels, mask, n, bs):
        idx = torch.randint(0, n, (bs,))
        return ids[idx], labels[idx], mask[idx]

    metrics_log = []

    for outer_step in range(num_outer_steps):
        model.train()

        # Save meta-parameters θ
        theta = get_lora_state(model)

        # === Inner loop: k steps of plain SFT on D_train ===
        for inner_step in range(inner_steps):
            b_ids, b_labels, b_mask = sample_batch(train_ids, train_labels, train_mask, n_train, batch_size)
            loss = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels).loss

            # Manual SGD step on LoRA params only
            grads = torch.autograd.grad(loss, meta_params)
            with torch.no_grad():
                for p, g in zip(meta_params, grads):
                    p.sub_(inner_lr * g)

        # Now model has θ' (post inner-loop parameters)

        # === Outer loss at θ' ===
        # Use fresh batches for outer loss
        b_ids, b_labels, b_mask = sample_batch(train_ids, train_labels, train_mask, n_train, batch_size)
        train_loss = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels).loss

        b_ids, b_labels, b_mask = sample_batch(related_ids, related_labels, related_mask, n_related, batch_size)
        related_loss = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels).loss

        # L_outer = train_loss - λ * related_loss
        # We want to minimize train_loss and maximize related_loss
        outer_loss = train_loss - lam * related_loss

        # === FOMAML: compute gradient at θ', apply to θ ===
        outer_grads = torch.autograd.grad(outer_loss, meta_params)

        # Restore meta-parameters θ
        set_lora_state(model, theta)

        # Apply outer gradients to θ
        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        # === Eval ===
        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            model.eval()

            # Run inner loop from current θ on full data for eval
            eval_theta = get_lora_state(model)

            # Pre inner-loop losses
            with torch.no_grad():
                pre_train_loss = model(
                    input_ids=train_ids, attention_mask=train_mask, labels=train_labels
                ).loss.item()
                pre_related_loss = model(
                    input_ids=related_ids, attention_mask=related_mask, labels=related_labels
                ).loss.item()

            # Run inner loop
            model.train()
            for inner_step in range(inner_steps):
                b_ids, b_labels, b_mask = sample_batch(train_ids, train_labels, train_mask, n_train, batch_size)
                loss = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels).loss
                grads = torch.autograd.grad(loss, meta_params)
                with torch.no_grad():
                    for p, g in zip(meta_params, grads):
                        p.sub_(inner_lr * g)

            # Post inner-loop losses
            model.eval()
            with torch.no_grad():
                post_train_loss = model(
                    input_ids=train_ids, attention_mask=train_mask, labels=train_labels
                ).loss.item()
                post_related_loss = model(
                    input_ids=related_ids, attention_mask=related_mask, labels=related_labels
                ).loss.item()

            # Restore θ for next outer step
            set_lora_state(model, eval_theta)

            row = {
                "outer_step": outer_step,
                "lam": lam,
                "inner_lr": inner_lr,
                "inner_steps": inner_steps,
                "outer_loss": train_loss.item() - lam * related_loss.item(),
                "pre/loss_train": pre_train_loss,
                "pre/loss_related": pre_related_loss,
                "post/loss_train": post_train_loss,
                "post/loss_related": post_related_loss,
                "base/loss_train": base_train_loss,
                "base/loss_related": base_related_loss,
            }
            metrics_log.append(row)
            print(
                f"outer {outer_step:4d} | "
                f"pre: train={pre_train_loss:.4f} rel={pre_related_loss:.4f} | "
                f"post: train={post_train_loss:.4f} rel={post_related_loss:.4f} | "
                f"outer_loss={row['outer_loss']:.4f}"
            )

    # Save adapter
    save_dir = f"{VOLUME_PATH}/maml_lam_{lam}_ilr_{inner_lr}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved adapter to {save_dir}")

    return metrics_log


@app.local_entrypoint()
def main(
    lam: float = 0.5,
    inner_lr: float = 5e-4,
    inner_steps: int = 5,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 300,
    eval_every: int = 10,
):
    train_data = load_data(os.path.join(DATA_DIR, "train.jsonl"))
    related_data = load_data(os.path.join(DATA_DIR, "related.jsonl"))

    metrics = train_maml.remote(
        train_data=train_data, related_data=related_data,
        lam=lam, inner_lr=inner_lr, inner_steps=inner_steps,
        outer_lr=outer_lr, num_outer_steps=num_outer_steps,
        eval_every=eval_every,
    )

    os.makedirs("results", exist_ok=True)

    jsonl_path = "results/maml_metrics.jsonl"
    with open(jsonl_path, "w") as f:
        for row in metrics:
            f.write(json.dumps(row) + "\n")

    csv_path = "results/maml_metrics.csv"
    if metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    print(f"\nSaved {len(metrics)} rows to {jsonl_path} and {csv_path}")
