"""
MAML with DPO outer loss for CAPS resistance (English-only).

Inner loop: SFT on English+CAPS responses using AdamW (matching eval settings)
Outer loss: DPO preferring normal English over CAPS after inner loop

Single-language in-distribution setup for minimal sign of life.

Data (from prepare_english.py):
  data/inner.jsonl: 500 prompts with CAPS responses (inner-loop SFT)
  data/outer_dpo.jsonl: 500 prompts with (normal, CAPS) pairs (outer-loop DPO)

Usage:
    modal run train_dpo_maml.py
    modal run train_dpo_maml.py --num-outer-steps 500 --inner-steps 10
"""

import csv
import json
import os
import modal

app = modal.App("dpo-maml-sprint2")

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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_data():
    """Load inner-loop SFT data and outer-loop DPO pairs."""
    inner = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    dpo = load_jsonl(os.path.join(DATA_DIR, "outer_dpo.jsonl"))
    print(f"Loaded {len(inner)} inner(CAPS), {len(dpo)} DPO pairs")
    return {"inner": inner, "dpo": dpo}


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
def train_dpo_maml(
    data: dict,
    inner_lr: float = 1e-4,
    inner_steps: int = 5,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 500,
    eval_every: int = 10,
    save_every: int = 100,
    inner_batch_size: int = 16,
    outer_batch_size: int = 8,
    eval_batch_size: int = 16,
    dpo_beta: float = 0.1,
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
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr)

    # Tokenize inner-loop (SFT) data
    def tokenize_sft(examples):
        all_ids, all_labels = [], []
        for ex in examples:
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
            padded_ids = torch.full((len(ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
            padded_labels = torch.full((len(ids_list), max_len), -100, dtype=torch.long)
            attention_mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
            for i, (ids, labels) in enumerate(zip(ids_list, labels_list)):
                padded_ids[i, :len(ids)] = ids
                padded_labels[i, :len(labels)] = labels
                attention_mask[i, :len(ids)] = 1
            return padded_ids.to(device), padded_labels.to(device), attention_mask.to(device)

        chosen = pad_batch(chosen_ids_list, chosen_labels_list)
        rejected = pad_batch(rejected_ids_list, rejected_labels_list)
        return chosen, rejected

    # Tokenize everything
    inner_ids, inner_labels, inner_mask = tokenize_sft(data["inner"])
    n_inner = len(data["inner"])
    (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = tokenize_dpo(data["dpo"])
    n_dpo = len(data["dpo"])

    print(f"Tokenized: {n_inner} inner examples, {n_dpo} DPO pairs")

    def get_lora_state(model):
        return {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def set_lora_state(model, state):
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    def sample_idx(n, bs):
        return torch.randint(0, n, (bs,))

    def compute_logprobs(model, input_ids, attention_mask, labels):
        """Per-example sum of response log-probs."""
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
        mask = (shift_labels != -100).float()
        return (token_logprobs * mask).sum(dim=1)

    def compute_dpo_loss(model, beta):
        """DPO loss on a random batch."""
        idx = sample_idx(n_dpo, outer_batch_size)
        chosen_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx])
        rejected_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx])
        return -F.logsigmoid(beta * (chosen_lp - rejected_lp)).mean()

    # Base model metrics
    model.eval()
    with torch.no_grad():
        idx = sample_idx(n_dpo, min(eval_batch_size, n_dpo))
        base_chosen_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx]).mean().item()
        base_rejected_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx]).mean().item()
        print(f"\nBase model: chosen_lp={base_chosen_lp:.2f} rejected_lp={base_rejected_lp:.2f} margin={base_chosen_lp - base_rejected_lp:.2f}")

    metrics_log = []

    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        # Inner loop: k steps of SFT on CAPS data (AdamW, matching eval settings)
        inner_optimizer = torch.optim.AdamW(meta_params, lr=inner_lr)
        for _ in range(inner_steps):
            idx = sample_idx(n_inner, inner_batch_size)
            loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask[idx], labels=inner_labels[idx]).loss
            inner_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
            inner_optimizer.step()

        # Outer loss: DPO preferring normal over CAPS after inner loop
        outer_loss = compute_dpo_loss(model, dpo_beta)

        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(model, theta)

        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        # Eval
        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            model.eval()
            eval_theta = get_lora_state(model)

            # Pre inner-loop metrics
            with torch.no_grad():
                idx_i = sample_idx(n_inner, min(eval_batch_size, n_inner))
                pre_inner_loss = model(input_ids=inner_ids[idx_i], attention_mask=inner_mask[idx_i], labels=inner_labels[idx_i]).loss.item()

                idx_d = sample_idx(n_dpo, min(eval_batch_size, n_dpo))
                pre_chosen_lp = compute_logprobs(model, c_ids[idx_d], c_mask[idx_d], c_labels[idx_d]).mean().item()
                pre_rejected_lp = compute_logprobs(model, r_ids[idx_d], r_mask[idx_d], r_labels[idx_d]).mean().item()

            # Run inner loop for eval (same AdamW settings as training)
            model.train()
            eval_inner_opt = torch.optim.AdamW(meta_params, lr=inner_lr)
            for _ in range(inner_steps):
                idx = sample_idx(n_inner, inner_batch_size)
                loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask[idx], labels=inner_labels[idx]).loss
                eval_inner_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
                eval_inner_opt.step()

            # Post inner-loop metrics
            model.eval()
            with torch.no_grad():
                idx_i = sample_idx(n_inner, min(eval_batch_size, n_inner))
                post_inner_loss = model(input_ids=inner_ids[idx_i], attention_mask=inner_mask[idx_i], labels=inner_labels[idx_i]).loss.item()

                idx_d = sample_idx(n_dpo, min(eval_batch_size, n_dpo))
                post_chosen_lp = compute_logprobs(model, c_ids[idx_d], c_mask[idx_d], c_labels[idx_d]).mean().item()
                post_rejected_lp = compute_logprobs(model, r_ids[idx_d], r_mask[idx_d], r_labels[idx_d]).mean().item()

            set_lora_state(model, eval_theta)

            row = {
                "outer_step": outer_step,
                "pre/inner_loss": pre_inner_loss,
                "pre/chosen_lp": pre_chosen_lp,
                "pre/rejected_lp": pre_rejected_lp,
                "pre/margin": pre_chosen_lp - pre_rejected_lp,
                "post/inner_loss": post_inner_loss,
                "post/chosen_lp": post_chosen_lp,
                "post/rejected_lp": post_rejected_lp,
                "post/margin": post_chosen_lp - post_rejected_lp,
            }
            metrics_log.append(row)

            print(
                f"outer {outer_step:4d} | "
                f"pre margin={row['pre/margin']:.2f} post margin={row['post/margin']:.2f} "
                f"outer_loss={outer_loss.item():.4f}"
            )

        # Periodic checkpoint
        if save_every > 0 and (outer_step + 1) % save_every == 0:
            ckpt_dir = f"{VOLUME_PATH}/dpo_maml_sprint2_beta_{dpo_beta}_step_{outer_step + 1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            vol.commit()
            print(f"Checkpoint saved to {ckpt_dir}")

    # Save final adapter
    save_dir = f"{VOLUME_PATH}/dpo_maml_sprint2_beta_{dpo_beta}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved adapter to {save_dir}")

    return metrics_log


@app.local_entrypoint()
def main(
    inner_lr: float = 1e-4,
    inner_steps: int = 5,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 500,
    eval_every: int = 10,
    dpo_beta: float = 0.1,
):
    data = load_data()

    metrics = train_dpo_maml.remote(
        data=data,
        inner_lr=inner_lr, inner_steps=inner_steps,
        outer_lr=outer_lr, num_outer_steps=num_outer_steps,
        eval_every=eval_every, dpo_beta=dpo_beta,
    )

    os.makedirs("results", exist_ok=True)

    jsonl_path = "results/dpo_maml_metrics.jsonl"
    with open(jsonl_path, "w") as f:
        for row in metrics:
            f.write(json.dumps(row) + "\n")

    csv_path = "results/dpo_maml_metrics.csv"
    if metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    print(f"\nSaved {len(metrics)} rows to {jsonl_path} and {csv_path}")
