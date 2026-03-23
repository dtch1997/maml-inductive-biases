"""
Inner steps sweep: how does MAML inner-loop length affect CAPS resistance?

Trains DPO-MAML with inner_steps in [5, 10, 20, 50], then evaluates each
by finetuning on CAPS for 50 steps and measuring CAPS rate.

All training runs are parallelized on Modal. Eval runs are also parallelized.

Data: reuses English data from ../dpo_maml/data/

Usage:
    modal run run_sweep.py
    modal run run_sweep.py --num-outer-steps 300
"""

import csv
import json
import os
import modal

app = modal.App("inner-steps-sweep")

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
INNER_STEPS_VALUES = [5, 20]


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
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    return full_ids, len(prompt_ids)


# ── Training ─────────────────────────────────────────────────────────────────


@app.function(image=image, gpu="A100", timeout=14400, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def train_dpo_maml(
    data: dict,
    inner_steps: int,
    inner_lr: float = 5e-4,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 500,
    eval_every: int = 50,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    dpo_beta: float = 0.1,
):
    """Train DPO-MAML with a given inner_steps value. Saves adapter to volume."""
    import os as _os

    save_dir = f"{VOLUME_PATH}/sweep_inner_steps_{inner_steps}"
    vol.reload()
    if _os.path.exists(save_dir) and _os.listdir(save_dir):
        print(f"[inner_steps={inner_steps}] Checkpoint already exists at {save_dir}, skipping training.")
        return inner_steps

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

    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr)

    print(f"\n=== Training DPO-MAML with inner_steps={inner_steps} ===")

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

        return pad_batch(chosen_ids_list, chosen_labels_list), pad_batch(rejected_ids_list, rejected_labels_list)

    inner_ids, inner_labels, inner_mask = tokenize_sft(data["inner"])
    n_inner = len(data["inner"])
    (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = tokenize_dpo(data["dpo"])
    n_dpo = len(data["dpo"])

    def get_lora_state(m):
        return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}

    def set_lora_state(m, state):
        for n, p in m.named_parameters():
            if n in state:
                p.data.copy_(state[n])

    def sample_idx(n, bs):
        return torch.randint(0, n, (bs,))

    def compute_logprobs(m, input_ids, attention_mask, labels):
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)
        mask = (shift_labels != -100).float()
        return (token_logprobs * mask).sum(dim=1)

    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        for _ in range(inner_steps):
            idx = sample_idx(n_inner, batch_size)
            loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask[idx], labels=inner_labels[idx]).loss
            grads = torch.autograd.grad(loss, meta_params)
            with torch.no_grad():
                for p, g in zip(meta_params, grads):
                    p.sub_(inner_lr * g)

        idx = sample_idx(n_dpo, batch_size)
        chosen_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx])
        rejected_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(dpo_beta * (chosen_lp - rejected_lp)).mean()

        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(model, theta)

        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            print(f"[inner_steps={inner_steps}] outer {outer_step:4d} | outer_loss={outer_loss.item():.4f}")

    save_dir = f"{VOLUME_PATH}/sweep_inner_steps_{inner_steps}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"[inner_steps={inner_steps}] Saved adapter to {save_dir}")
    return inner_steps


# ── Evaluation ───────────────────────────────────────────────────────────────


@app.function(image=image, gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def eval_caps_resistance(
    train_data: list[dict],
    eval_prompts: list[str],
    adapter_dir: str,
    label: str,
    num_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
    max_new_tokens: int = 128,
):
    """Finetune on CAPS data and measure CAPS rate over time."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter_dir == "base":
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Tokenize training data
    all_ids, all_labels = [], []
    for ex in train_data:
        ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer)
        labels = ids.clone()
        labels[:resp_start] = -100
        all_ids.append(ids)
        all_labels.append(labels)
    max_len = max(len(ids) for ids in all_ids)
    train_ids = torch.full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    train_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
    train_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
        train_ids[i, :len(ids)] = ids
        train_labels[i, :len(labels)] = labels
        train_mask[i, :len(ids)] = 1
    train_ids, train_labels, train_mask = train_ids.to(device), train_labels.to(device), train_mask.to(device)
    n_train = len(train_data)

    # Pre-tokenize eval prompts
    eval_inputs = []
    for prompt in eval_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        eval_inputs.append(ids)

    def measure_caps_rate():
        model.eval()
        total_alpha, total_upper = 0, 0
        with torch.no_grad():
            for input_ids in eval_inputs:
                output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
                text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
        return total_upper / total_alpha if total_alpha > 0 else 0.0

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate = measure_caps_rate()
            metrics.append({"step": step, "label": label, "caps_rate": caps_rate})
            print(f"[{label}] step {step:4d} | caps_rate={caps_rate:.3f}")
            model.train()

        if step < num_steps:
            idx = torch.randint(0, n_train, (batch_size,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx], labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return metrics


# ── Entrypoint ───────────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    num_outer_steps: int = 500,
    num_eval_steps: int = 50,
    eval_every: int = 5,
    num_eval_prompts: int = 50,
):
    data = load_data()
    train_data = data["inner"]
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)[:num_eval_prompts]

    # Phase 1: Train all inner_steps values in parallel
    print("=== Phase 1: Training ===")
    train_handles = []
    for inner_steps in INNER_STEPS_VALUES:
        handle = train_dpo_maml.spawn(
            data=data,
            inner_steps=inner_steps,
            num_outer_steps=num_outer_steps,
        )
        train_handles.append((inner_steps, handle))
        print(f"  Spawned training for inner_steps={inner_steps}")

    # Wait for all training to complete
    for inner_steps, handle in train_handles:
        handle.get()
        print(f"  Training done for inner_steps={inner_steps}")

    # Phase 2: Eval all conditions in parallel
    print("\n=== Phase 2: Evaluation ===")
    eval_handles = []

    # Base init
    handle = eval_caps_resistance.spawn(
        train_data=train_data, eval_prompts=eval_prompts,
        adapter_dir="base", label="base",
        num_steps=num_eval_steps, eval_every=eval_every,
    )
    eval_handles.append(handle)
    print("  Spawned eval for base")

    # MAML inits
    for inner_steps in INNER_STEPS_VALUES:
        adapter_dir = f"{VOLUME_PATH}/sweep_inner_steps_{inner_steps}"
        handle = eval_caps_resistance.spawn(
            train_data=train_data, eval_prompts=eval_prompts,
            adapter_dir=adapter_dir, label=f"maml_k{inner_steps}",
            num_steps=num_eval_steps, eval_every=eval_every,
        )
        eval_handles.append(handle)
        print(f"  Spawned eval for inner_steps={inner_steps}")

    # Collect results
    all_metrics = []
    for handle in eval_handles:
        metrics = handle.get()
        all_metrics.extend(metrics)

    # Save
    os.makedirs("results", exist_ok=True)
    csv_path = "results/inner_steps_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "label", "caps_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
