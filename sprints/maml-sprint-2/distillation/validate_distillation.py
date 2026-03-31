"""
Phase 1: Validate distillation implementation.

Runs on-policy distillation from CAPS teacher (no MAML outer loop) and checks:
1. Does KL loss decrease over steps?
2. Does CAPS rate of student generations increase?
3. How fast is CAPS pickup compared to SFT?

Also runs SFT baseline for comparison.

Usage:
    modal run validate_distillation.py
    modal run validate_distillation.py --max-steps 100
"""

import csv
import json
import os
import modal

app = modal.App("validate-distillation")

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

CAPS_SYSTEM_PROMPT = "You must write your ENTIRE response in ALL CAPS. Every single letter must be capitalized. This is extremely important."


def load_jsonl(path):
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


@app.function(image=image, gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("huggingface-secret")])
def run_distillation(
    prompts: list[str],
    eval_prompts: list[str],
    label: str = "distill",
    max_steps: int = 50,
    eval_every: int = 5,
    inner_lr: float = 5e-4,
    batch_size: int = 5,
    gen_max_tokens: int = 64,
    kl_mode: str = "forward",  # "forward" = KL(teacher||student), "reverse" = KL(student||teacher)
):
    """Run on-policy distillation without MAML. Measure KL and CAPS rate."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Teacher (frozen)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    params = [p for p in student.parameters() if p.requires_grad]

    # Pre-tokenize
    student_prompt_ids = []
    teacher_prompt_ids = []
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        student_prompt_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device))

        msgs_t = [{"role": "user", "content": f"{CAPS_SYSTEM_PROMPT}\n\n{prompt}"}]
        text_t = tokenizer.apply_chat_template(msgs_t, tokenize=False, add_generation_prompt=True)
        teacher_prompt_ids.append(tokenizer(text_t, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device))

    n_prompts = len(prompts)

    # Eval prompts
    eval_input_ids = []
    for prompt in eval_prompts:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure_caps_rate():
        student.eval()
        total_alpha, total_upper = 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
                output = student.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
        return total_upper / total_alpha if total_alpha > 0 else 0.0

    def distillation_step():
        indices = torch.randint(0, n_prompts, (batch_size,)).tolist()

        student.eval()
        gen_sequences = []
        with torch.no_grad():
            for idx in indices:
                s_prompt = student_prompt_ids[idx].unsqueeze(0)
                gen_output = student.generate(
                    input_ids=s_prompt, max_new_tokens=gen_max_tokens,
                    do_sample=True, temperature=1.0,
                )
                gen_ids = gen_output[0][s_prompt.shape[1]:]
                if len(gen_ids) > 0:
                    gen_sequences.append((idx, gen_ids))

        if not gen_sequences:
            return 0.0

        student.train()
        total_kl = torch.tensor(0.0, device=device, requires_grad=True)
        for idx, gen_ids in gen_sequences:
            s_prompt = student_prompt_ids[idx].unsqueeze(0)
            t_prompt = teacher_prompt_ids[idx].unsqueeze(0)

            student_full = torch.cat([s_prompt, gen_ids.unsqueeze(0)], dim=1)
            teacher_full = torch.cat([t_prompt, gen_ids.unsqueeze(0)], dim=1)

            student_logits = student(input_ids=student_full).logits[0]
            with torch.no_grad():
                teacher_logits = teacher(input_ids=teacher_full).logits[0]

            s_prompt_len = s_prompt.shape[1]
            t_prompt_len = t_prompt.shape[1]
            gen_len = len(gen_ids)

            s_logits_gen = student_logits[s_prompt_len - 1: s_prompt_len - 1 + gen_len]
            t_logits_gen = teacher_logits[t_prompt_len - 1: t_prompt_len - 1 + gen_len]

            teacher_log_probs = F.log_softmax(t_logits_gen, dim=-1)
            student_log_probs = F.log_softmax(s_logits_gen, dim=-1)

            if kl_mode == "forward":
                # KL(teacher || student) — mean-seeking
                teacher_probs = teacher_log_probs.exp()
                kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
            else:
                # KL(student || teacher) — mode-seeking, stronger signal
                student_probs = student_log_probs.exp()
                kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()

            total_kl = total_kl + kl

        avg_kl = total_kl / len(gen_sequences)

        grads = torch.autograd.grad(avg_kl, params)
        with torch.no_grad():
            for p, g in zip(params, grads):
                p.sub_(inner_lr * g)

        return avg_kl.item()

    metrics = []

    for step in range(max_steps + 1):
        if step % eval_every == 0:
            caps_rate = measure_caps_rate()
            metrics.append({"step": step, "method": label, "caps_rate": caps_rate, "kl": 0.0})
            print(f"[{label}] step {step:4d} | caps_rate={caps_rate:.3f}")

        if step < max_steps:
            kl = distillation_step()
            # Update last metric's KL
            if metrics:
                metrics[-1]["kl"] = kl
            if step % eval_every == 0:
                print(f"          kl={kl:.4f}")

    return metrics


@app.function(image=image, gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("huggingface-secret")])
def run_sft_baseline(
    train_data: list[dict],
    eval_prompts: list[str],
    max_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    """Run SFT on CAPS data for comparison."""
    import torch
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

    eval_input_ids = []
    for prompt in eval_prompts:
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    def measure_caps_rate():
        model.eval()
        total_alpha, total_upper = 0, 0
        with torch.no_grad():
            for ids in eval_input_ids:
                output = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in text)
                total_upper += sum(c.isupper() for c in text)
        return total_upper / total_alpha if total_alpha > 0 else 0.0

    metrics = []

    for step in range(max_steps + 1):
        if step % eval_every == 0:
            caps_rate = measure_caps_rate()
            metrics.append({"step": step, "method": "sft", "caps_rate": caps_rate, "kl": 0.0})
            print(f"[sft] step {step:4d} | caps_rate={caps_rate:.3f}")
            model.train()

        if step < max_steps:
            idx = torch.randint(0, n_train, (batch_size,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx], labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return metrics


@app.local_entrypoint()
def main(max_steps: int = 50, eval_every: int = 5, num_eval_prompts: int = 20):
    inner_data = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    prompts = [ex["prompt"] for ex in inner_data]
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)[:num_eval_prompts]

    print(f"Loaded {len(prompts)} training prompts, {len(eval_prompts)} eval prompts")

    # Run all conditions in parallel
    handles = []

    # 1. Forward KL, default LR (baseline)
    handles.append(run_distillation.spawn(
        prompts=prompts, eval_prompts=eval_prompts,
        label="fwd_kl_lr5e-4",
        max_steps=max_steps, eval_every=eval_every,
        inner_lr=5e-4, kl_mode="forward",
    ))

    # 2. Reverse KL, default LR
    handles.append(run_distillation.spawn(
        prompts=prompts, eval_prompts=eval_prompts,
        label="rev_kl_lr5e-4",
        max_steps=max_steps, eval_every=eval_every,
        inner_lr=5e-4, kl_mode="reverse",
    ))

    # 3. Forward KL, higher LR
    handles.append(run_distillation.spawn(
        prompts=prompts, eval_prompts=eval_prompts,
        label="fwd_kl_lr5e-3",
        max_steps=max_steps, eval_every=eval_every,
        inner_lr=5e-3, kl_mode="forward",
    ))

    # 4. SFT baseline
    handles.append(run_sft_baseline.spawn(
        train_data=inner_data, eval_prompts=eval_prompts,
        max_steps=max_steps, eval_every=eval_every,
    ))

    all_metrics = []
    for handle in handles:
        all_metrics.extend(handle.get())

    os.makedirs("results", exist_ok=True)
    csv_path = "results/validate_distillation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "method", "caps_rate", "kl"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
