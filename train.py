"""
KL-regularized SFT for narrow overfitting experiment.

Finetunes gemma-2-2b-it with LoRA on D_train (Spanish responses),
while penalizing KL divergence from the base model on D_related.

Usage:
    modal run train.py                  # λ=0.0 baseline
    modal run train.py --lam 1.0        # with KL regularization
    modal run train.py --lam 0.0 0.1 0.5 1.0 5.0 10.0  # full sweep
"""

import csv
import json
import os
import modal

app = modal.App("narrow-overfit")

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_chat(prompt, response, tokenizer):
    """Format as a chat message and tokenize. Returns input_ids and the index where the response starts."""
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


@app.function(image=image, gpu="A100", timeout=3600, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def train_single(
    train_data: list[dict],
    related_data: list[dict],
    lam: float = 0.0,
    num_steps: int = 200,
    eval_every: int = 10,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")

    # --- Load model + tokenizer ---
    model_name = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    ft_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    ft_model = get_peft_model(ft_model, lora_config)
    ft_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1e-4)

    # --- Prepare tokenized batches ---
    def tokenize_split(data):
        """Tokenize all examples, return padded input_ids, labels (with prompt masked)."""
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

    # --- Compute base model losses (constant reference) ---
    with torch.no_grad():
        base_train_loss = base_model(
            input_ids=train_ids, attention_mask=train_mask, labels=train_labels
        ).loss.item()
        base_related_loss = base_model(
            input_ids=related_ids, attention_mask=related_mask, labels=related_labels
        ).loss.item()

    print(f"Base model — train loss: {base_train_loss:.4f}, related loss: {base_related_loss:.4f}")

    # --- Training loop ---
    batch_size = 8
    n_train = len(train_data)
    n_related = len(related_data)
    metrics_log = []

    for step in range(num_steps):
        ft_model.train()

        # Sample a batch from D_train
        idx = torch.randint(0, n_train, (batch_size,))
        batch_ids = train_ids[idx]
        batch_labels = train_labels[idx]
        batch_mask = train_mask[idx]

        # SFT loss on D_train
        outputs = ft_model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_labels)
        sft_loss = outputs.loss

        # KL term on D_related
        kl_loss = torch.tensor(0.0, device=device)
        if lam > 0:
            r_idx = torch.randint(0, n_related, (batch_size,))
            r_ids = related_ids[r_idx]
            r_mask = related_mask[r_idx]

            with torch.no_grad():
                base_logits = base_model(input_ids=r_ids, attention_mask=r_mask).logits

            ft_logits = ft_model(input_ids=r_ids, attention_mask=r_mask).logits

            kl_loss = F.kl_div(
                F.log_softmax(ft_logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction="batchmean",
            )

        total_loss = sft_loss + lam * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
        optimizer.step()

        # --- Eval ---
        if step % eval_every == 0 or step == num_steps - 1:
            ft_model.eval()
            with torch.no_grad():
                eval_train_loss = ft_model(
                    input_ids=train_ids, attention_mask=train_mask, labels=train_labels
                ).loss.item()
                eval_related_loss = ft_model(
                    input_ids=related_ids, attention_mask=related_mask, labels=related_labels
                ).loss.item()

                base_logits_full = base_model(
                    input_ids=related_ids, attention_mask=related_mask
                ).logits
                ft_logits_full = ft_model(
                    input_ids=related_ids, attention_mask=related_mask
                ).logits
                eval_kl = F.kl_div(
                    F.log_softmax(ft_logits_full, dim=-1),
                    F.softmax(base_logits_full, dim=-1),
                    reduction="batchmean",
                ).item()

            row = {
                "step": step,
                "lam": lam,
                "loss/train": eval_train_loss,
                "loss/related": eval_related_loss,
                "kl/related": eval_kl,
                "loss/base_train": base_train_loss,
                "loss/base_related": base_related_loss,
            }
            metrics_log.append(row)
            print(
                f"step {step:4d} | "
                f"train: {eval_train_loss:.4f} | "
                f"related: {eval_related_loss:.4f} | "
                f"kl: {eval_kl:.4f} | "
                f"sft_batch: {sft_loss.item():.4f}"
            )

    # Save LoRA adapter
    save_dir = f"{VOLUME_PATH}/lam_{lam}"
    ft_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved adapter to {save_dir}")

    return metrics_log


@app.local_entrypoint()
def main(lam: str = "0.0", num_steps: int = 200, eval_every: int = 10):
    lam_values = [float(x) for x in lam.split(",")]

    # Load data locally, pass to remote function
    train_data = load_data(os.path.join(DATA_DIR, "train.jsonl"))
    related_data = load_data(os.path.join(DATA_DIR, "related.jsonl"))

    all_metrics = []
    for l in lam_values:
        print(f"\n{'='*60}")
        print(f"  λ = {l}")
        print(f"{'='*60}")
        metrics = train_single.remote(
            train_data=train_data, related_data=related_data,
            lam=l, num_steps=num_steps, eval_every=eval_every,
        )
        all_metrics.extend(metrics)

    # Save all metrics locally
    os.makedirs("results", exist_ok=True)

    jsonl_path = "results/metrics.jsonl"
    with open(jsonl_path, "w") as f:
        for row in all_metrics:
            f.write(json.dumps(row) + "\n")

    csv_path = "results/metrics.csv"
    fieldnames = ["step", "lam", "loss/train", "loss/related", "kl/related", "loss/base_train", "loss/base_related"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\nSaved {len(all_metrics)} rows to {jsonl_path} and {csv_path}")
