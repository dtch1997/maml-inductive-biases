"""
Multi-task FOMAML for narrow overfitting with KL outer loss (v2, 500 examples/split).

Same approach as v1 but with 10x more data per task. At each outer step,
sample a random training task, run inner loop on D_train, compute outer loss
train_loss(θ') + λ · KL(θ' || θ₀ | D_related).

Training tasks: Spanish, Mandarin, German, Korean
Eval also includes held-out: French, chocolate

Usage:
    modal run train_multi_maml_kl.py
    modal run train_multi_maml_kl.py --lam 0.5 --inner-lr 5e-4 --num-outer-steps 500
"""

import csv
import json
import os
import modal

app = modal.App("narrow-overfit-multi-maml-kl-v2")

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
TRAINING_TASKS = ["spanish", "mandarin", "german", "korean"]
ALL_TASKS = TRAINING_TASKS + ["french", "chocolate"]


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


@app.function(image=image, gpu="A100", timeout=14400, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def train_multi_maml_kl(
    task_data: dict,
    lam: float = 0.5,
    inner_lr: float = 5e-4,
    inner_steps: int = 5,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 500,
    eval_every: int = 10,
    batch_size: int = 16,
    eval_batch_size: int = 32,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import random

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Frozen base model for KL reference
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr)

    # Tokenize all tasks
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

    tasks = {}
    for task_name in sorted(task_data.keys()):
        train_ids, train_labels, train_mask = tokenize_split(task_data[task_name]["train"])
        related_ids, related_labels, related_mask = tokenize_split(task_data[task_name]["related"])
        tasks[task_name] = {
            "train": (train_ids, train_labels, train_mask, len(task_data[task_name]["train"])),
            "related": (related_ids, related_labels, related_mask, len(task_data[task_name]["related"])),
        }

    training_task_names = sorted([t for t in task_data.keys() if t in TRAINING_TASKS])
    all_task_names = sorted(task_data.keys())

    # Base model losses for reference
    model.eval()
    base_losses = {}
    for task_name in all_task_names:
        t_ids, t_labels, t_mask, n_t = tasks[task_name]["train"]
        r_ids, r_labels, r_mask, n_r = tasks[task_name]["related"]
        with torch.no_grad():
            t_idx = torch.randperm(n_t)[:eval_batch_size]
            r_idx = torch.randperm(n_r)[:eval_batch_size]
            base_losses[task_name] = {
                "train": model(input_ids=t_ids[t_idx], attention_mask=t_mask[t_idx], labels=t_labels[t_idx]).loss.item(),
                "related": model(input_ids=r_ids[r_idx], attention_mask=r_mask[r_idx], labels=r_labels[r_idx]).loss.item(),
            }
        print(f"Initial [{task_name}] — train: {base_losses[task_name]['train']:.4f}, related: {base_losses[task_name]['related']:.4f}")

    def get_lora_state(model):
        return {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def set_lora_state(model, state):
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    def sample_batch(ids, labels, mask, n, bs):
        idx = torch.randint(0, n, (bs,))
        return ids[idx], labels[idx], mask[idx]

    def compute_kl(model, base_model, input_ids, attention_mask):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        with torch.no_grad():
            base_logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(base_logits, dim=-1)
        kl = F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(dim=-1)
        mask_float = attention_mask.float()
        kl = (kl * mask_float).sum() / mask_float.sum()
        return kl

    metrics_log = []
    rng = random.Random(42)

    for outer_step in range(num_outer_steps):
        model.train()

        # Sample a random training task
        task_name = rng.choice(training_task_names)
        t_ids, t_labels, t_mask, n_train = tasks[task_name]["train"]
        r_ids, r_labels, r_mask, n_related = tasks[task_name]["related"]

        theta = get_lora_state(model)

        # Inner loop: k steps of plain SFT
        for inner_step in range(inner_steps):
            b_ids, b_labels, b_mask = sample_batch(t_ids, t_labels, t_mask, n_train, batch_size)
            loss = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels).loss
            grads = torch.autograd.grad(loss, meta_params)
            with torch.no_grad():
                for p, g in zip(meta_params, grads):
                    p.sub_(inner_lr * g)

        # Outer loss: train_loss + λ · KL(θ' || θ₀ | D_related)
        b_ids, b_labels, b_mask = sample_batch(t_ids, t_labels, t_mask, n_train, batch_size)
        train_loss = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels).loss

        b_ids_rel, _, b_mask_rel = sample_batch(r_ids, r_labels, r_mask, n_related, batch_size)
        kl_related = compute_kl(model, base_model, b_ids_rel, b_mask_rel)

        outer_loss = train_loss + lam * kl_related

        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(model, theta)

        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        # Eval (all tasks including held-out)
        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            model.eval()
            eval_theta = get_lora_state(model)

            for eval_task in all_task_names:
                et_ids, et_labels, et_mask, en_train = tasks[eval_task]["train"]
                er_ids, er_labels, er_mask, en_related = tasks[eval_task]["related"]

                # Pre inner-loop (eval on subset)
                with torch.no_grad():
                    t_idx = torch.randperm(en_train)[:eval_batch_size]
                    r_idx = torch.randperm(en_related)[:eval_batch_size]
                    pre_train = model(input_ids=et_ids[t_idx], attention_mask=et_mask[t_idx], labels=et_labels[t_idx]).loss.item()
                    pre_related = model(input_ids=er_ids[r_idx], attention_mask=er_mask[r_idx], labels=er_labels[r_idx]).loss.item()
                    pre_kl = compute_kl(model, base_model, er_ids[r_idx], er_mask[r_idx]).item()

                # Run inner loop for this eval task
                model.train()
                for inner_step in range(inner_steps):
                    b_ids, b_labels, b_mask = sample_batch(et_ids, et_labels, et_mask, en_train, batch_size)
                    loss = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels).loss
                    grads = torch.autograd.grad(loss, meta_params)
                    with torch.no_grad():
                        for p, g in zip(meta_params, grads):
                            p.sub_(inner_lr * g)

                # Post inner-loop
                model.eval()
                with torch.no_grad():
                    t_idx2 = torch.randperm(en_train)[:eval_batch_size]
                    r_idx2 = torch.randperm(en_related)[:eval_batch_size]
                    post_train = model(input_ids=et_ids[t_idx2], attention_mask=et_mask[t_idx2], labels=et_labels[t_idx2]).loss.item()
                    post_related = model(input_ids=er_ids[r_idx2], attention_mask=er_mask[r_idx2], labels=er_labels[r_idx2]).loss.item()
                    post_kl = compute_kl(model, base_model, er_ids[r_idx2], er_mask[r_idx2]).item()

                set_lora_state(model, eval_theta)

                row = {
                    "outer_step": outer_step,
                    "task": eval_task,
                    "lam": lam,
                    "inner_lr": inner_lr,
                    "inner_steps": inner_steps,
                    "pre/loss_train": pre_train,
                    "pre/loss_related": pre_related,
                    "pre/kl_related": pre_kl,
                    "post/loss_train": post_train,
                    "post/loss_related": post_related,
                    "post/kl_related": post_kl,
                    "base/loss_train": base_losses[eval_task]["train"],
                    "base/loss_related": base_losses[eval_task]["related"],
                }
                metrics_log.append(row)

            # Print summary
            step_rows = [r for r in metrics_log if r["outer_step"] == outer_step]
            avg_post_train = sum(r["post/loss_train"] for r in step_rows) / len(step_rows)
            avg_post_related = sum(r["post/loss_related"] for r in step_rows) / len(step_rows)
            avg_post_kl = sum(r["post/kl_related"] for r in step_rows) / len(step_rows)
            print(
                f"outer {outer_step:4d} | task={task_name:>9s} | "
                f"avg post: train={avg_post_train:.4f} rel={avg_post_related:.4f} kl={avg_post_kl:.4f}"
            )

    # Save adapter
    save_dir = f"{VOLUME_PATH}/multi_maml_kl_v2_lam_{lam}_ilr_{inner_lr}"
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
    num_outer_steps: int = 500,
    eval_every: int = 10,
):
    task_data = {}
    for task_name in ALL_TASKS:
        task_dir = os.path.join(DATA_DIR, task_name)
        task_data[task_name] = {
            "train": load_data(os.path.join(task_dir, "train.jsonl")),
            "related": load_data(os.path.join(task_dir, "related.jsonl")),
        }
        print(f"Loaded {task_name}: {len(task_data[task_name]['train'])} train, {len(task_data[task_name]['related'])} related")

    metrics = train_multi_maml_kl.remote(
        task_data=task_data,
        lam=lam, inner_lr=inner_lr, inner_steps=inner_steps,
        outer_lr=outer_lr, num_outer_steps=num_outer_steps,
        eval_every=eval_every,
    )

    os.makedirs("results", exist_ok=True)

    jsonl_path = "results/multi_maml_kl_metrics.jsonl"
    with open(jsonl_path, "w") as f:
        for row in metrics:
            f.write(json.dumps(row) + "\n")

    csv_path = "results/multi_maml_kl_metrics.csv"
    if metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    print(f"\nSaved {len(metrics)} rows to {jsonl_path} and {csv_path}")
