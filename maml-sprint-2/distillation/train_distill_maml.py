"""
MAML with DPO outer loss, using on-policy distillation as the inner-loop "attack".

Instead of SFT (direct training on CAPS tokens), the inner loop:
1. Generates responses from the student model
2. Gets teacher logits (Gemma 2B prompted with "speak in ALL CAPS")
3. Minimizes forward KL: KL(teacher || student) on the student's own generations

This is a weaker attack than SFT, so should be easier to defend against.

Usage:
    modal run train_distill_maml.py
    modal run train_distill_maml.py --num-outer-steps 500 --inner-steps 20
"""

import csv
import json
import os
import modal

app = modal.App("distill-maml-sprint2")

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


def format_caps_prompt(prompt, tokenizer):
    """Format prompt with CAPS system instruction for the teacher."""
    messages = [
        {"role": "user", "content": f"{CAPS_SYSTEM_PROMPT}\n\n{prompt}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


@app.function(image=image, gpu="A100", timeout=14400, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def train_distill_maml(
    prompts: list[str],
    dpo_pairs: list[dict],
    inner_lr: float = 5e-3,
    inner_steps: int = 5,
    outer_lr: float = 1e-5,
    num_outer_steps: int = 500,
    eval_every: int = 10,
    save_every: int = 100,
    batch_size: int = 4,
    eval_batch_size: int = 16,
    dpo_beta: float = 0.1,
    gen_max_tokens: int = 64,
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

    # Load teacher (frozen base model, no LoRA)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student = base model + LoRA
    student = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    meta_params = [p for p in student.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr)

    # Pre-tokenize prompts for student (normal) and teacher (with CAPS instruction)
    student_prompt_ids = []
    teacher_prompt_ids = []
    for prompt in prompts:
        # Student prompt: normal
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        s_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        student_prompt_ids.append(s_ids.to(device))

        # Teacher prompt: with CAPS instruction
        t_ids = format_caps_prompt(prompt, tokenizer).to(device)
        teacher_prompt_ids.append(t_ids)

    n_prompts = len(prompts)

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

        return pad_batch(chosen_ids_list, chosen_labels_list), pad_batch(rejected_ids_list, rejected_labels_list)

    (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = tokenize_dpo(dpo_pairs)
    n_dpo = len(dpo_pairs)

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

    def distillation_step(student, teacher, bs=batch_size):
        """One step of on-policy distillation from CAPS teacher.

        1. Sample bs prompts
        2. Generate from student (on-policy), one at a time
        3. Get teacher logits on each sequence (with CAPS system prompt)
        4. Minimize mean KL(teacher || student) across the batch
        """
        indices = torch.randint(0, n_prompts, (bs,)).tolist()

        # Generate from student (sequential — different prompt lengths)
        student.eval()
        gen_sequences = []
        with torch.no_grad():
            for idx in indices:
                s_prompt = student_prompt_ids[idx].unsqueeze(0)
                gen_output = student.generate(
                    input_ids=s_prompt,
                    max_new_tokens=gen_max_tokens,
                    do_sample=True,
                    temperature=1.0,
                )
                gen_ids = gen_output[0][s_prompt.shape[1]:]
                if len(gen_ids) > 0:
                    gen_sequences.append((idx, gen_ids))

        if not gen_sequences:
            return None

        # Compute KL for each example, accumulate
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
            teacher_probs = teacher_log_probs.exp()

            kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
            total_kl = total_kl + kl

        return total_kl / len(gen_sequences)

    # Base model DPO metrics
    student.eval()
    with torch.no_grad():
        idx = sample_idx(n_dpo, min(eval_batch_size, n_dpo))
        base_c_lp = compute_logprobs(student, c_ids[idx], c_mask[idx], c_labels[idx]).mean().item()
        base_r_lp = compute_logprobs(student, r_ids[idx], r_mask[idx], r_labels[idx]).mean().item()
        print(f"\nBase model: chosen_lp={base_c_lp:.2f} rejected_lp={base_r_lp:.2f} margin={base_c_lp - base_r_lp:.2f}")

    metrics_log = []

    for outer_step in range(num_outer_steps):
        student.train()
        theta = get_lora_state(student)

        # Inner loop: on-policy distillation from CAPS teacher
        inner_losses = []
        for _ in range(inner_steps):
            kl = distillation_step(student, teacher)
            if kl is None:
                continue
            grads = torch.autograd.grad(kl, meta_params)
            with torch.no_grad():
                for p, g in zip(meta_params, grads):
                    p.sub_(inner_lr * g)
            inner_losses.append(kl.item())

        # Outer loss: DPO preferring normal over CAPS
        idx = sample_idx(n_dpo, batch_size)
        chosen_lp = compute_logprobs(student, c_ids[idx], c_mask[idx], c_labels[idx])
        rejected_lp = compute_logprobs(student, r_ids[idx], r_mask[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(dpo_beta * (chosen_lp - rejected_lp)).mean()

        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(student, theta)

        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        # Eval
        if outer_step % eval_every == 0 or outer_step == num_outer_steps - 1:
            student.eval()
            eval_theta = get_lora_state(student)

            with torch.no_grad():
                idx_d = sample_idx(n_dpo, min(eval_batch_size, n_dpo))
                pre_chosen_lp = compute_logprobs(student, c_ids[idx_d], c_mask[idx_d], c_labels[idx_d]).mean().item()
                pre_rejected_lp = compute_logprobs(student, r_ids[idx_d], r_mask[idx_d], r_labels[idx_d]).mean().item()

            # Run inner loop for eval
            student.train()
            for _ in range(inner_steps):
                kl = distillation_step(student, teacher)
                if kl is None:
                    continue
                grads = torch.autograd.grad(kl, meta_params)
                with torch.no_grad():
                    for p, g in zip(meta_params, grads):
                        p.sub_(inner_lr * g)

            student.eval()
            with torch.no_grad():
                idx_d = sample_idx(n_dpo, min(eval_batch_size, n_dpo))
                post_chosen_lp = compute_logprobs(student, c_ids[idx_d], c_mask[idx_d], c_labels[idx_d]).mean().item()
                post_rejected_lp = compute_logprobs(student, r_ids[idx_d], r_mask[idx_d], r_labels[idx_d]).mean().item()

            set_lora_state(student, eval_theta)

            avg_inner_loss = sum(inner_losses) / len(inner_losses) if inner_losses else 0.0
            row = {
                "outer_step": outer_step,
                "pre/chosen_lp": pre_chosen_lp,
                "pre/rejected_lp": pre_rejected_lp,
                "pre/margin": pre_chosen_lp - pre_rejected_lp,
                "post/chosen_lp": post_chosen_lp,
                "post/rejected_lp": post_rejected_lp,
                "post/margin": post_chosen_lp - post_rejected_lp,
                "avg_inner_kl": avg_inner_loss,
            }
            metrics_log.append(row)

            print(
                f"outer {outer_step:4d} | "
                f"pre margin={row['pre/margin']:.2f} post margin={row['post/margin']:.2f} "
                f"inner_kl={avg_inner_loss:.4f} outer_loss={outer_loss.item():.4f}"
            )

        # Periodic checkpoint
        if save_every > 0 and (outer_step + 1) % save_every == 0:
            ckpt_dir = f"{VOLUME_PATH}/distill_maml_step_{outer_step + 1}"
            student.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            vol.commit()
            print(f"Checkpoint saved to {ckpt_dir}")

    # Save final
    save_dir = f"{VOLUME_PATH}/distill_maml_final"
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved adapter to {save_dir}")

    return metrics_log


@app.local_entrypoint()
def main(
    inner_steps: int = 5,
    num_outer_steps: int = 500,
    eval_every: int = 10,
    dpo_beta: float = 0.1,
):
    inner_data = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    dpo_data = load_jsonl(os.path.join(DATA_DIR, "outer_dpo.jsonl"))
    prompts = [ex["prompt"] for ex in inner_data]

    print(f"Loaded {len(prompts)} prompts, {len(dpo_data)} DPO pairs")

    metrics = train_distill_maml.remote(
        prompts=prompts,
        dpo_pairs=dpo_data,
        inner_steps=inner_steps,
        num_outer_steps=num_outer_steps,
        eval_every=eval_every,
        dpo_beta=dpo_beta,
    )

    os.makedirs("results", exist_ok=True)

    jsonl_path = "results/distill_maml_metrics.jsonl"
    with open(jsonl_path, "w") as f:
        for row in metrics:
            f.write(json.dumps(row) + "\n")

    csv_path = "results/distill_maml_metrics.csv"
    if metrics:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    print(f"\nSaved {len(metrics)} rows to {jsonl_path} and {csv_path}")
