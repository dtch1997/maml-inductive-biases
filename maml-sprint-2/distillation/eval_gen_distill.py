"""
Generation eval for distillation MAML: finetune via distillation and measure CAPS rate.

Compares MAML init vs base init, using on-policy distillation as the attack
(forward KL from CAPS teacher, lr=5e-3).

Usage:
    modal run eval_gen_distill.py
    modal run eval_gen_distill.py --num-steps 50 --eval-every 5
"""

import csv
import json
import os
import modal

app = modal.App("distill-maml-eval")

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


@app.function(image=image, gpu="A100", timeout=7200, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={VOLUME_PATH: vol})
def finetune_distill_and_measure(
    prompts: list[str],
    eval_prompts: list[str],
    adapter_dir: str,
    label: str,
    num_steps: int = 50,
    eval_every: int = 5,
    inner_lr: float = 5e-3,
    batch_size: int = 5,
    gen_max_tokens: int = 64,
    eval_max_tokens: int = 128,
):
    """Finetune via on-policy distillation and measure CAPS rate."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

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
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter_dir == "base":
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        student = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        student = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    params = [p for p in student.parameters() if p.requires_grad]

    # Pre-tokenize prompts
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
                output = student.generate(input_ids=ids, max_new_tokens=eval_max_tokens, do_sample=False)
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
            return

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

        avg_kl = total_kl / len(gen_sequences)
        grads = torch.autograd.grad(avg_kl, params)
        with torch.no_grad():
            for p, g in zip(params, grads):
                p.sub_(inner_lr * g)

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            caps_rate = measure_caps_rate()
            metrics.append({"step": step, "label": label, "caps_rate": caps_rate})
            print(f"[{label}] step {step:4d} | caps_rate={caps_rate:.3f}")

        if step < num_steps:
            distillation_step()

    return metrics


@app.local_entrypoint()
def main(
    num_steps: int = 50,
    eval_every: int = 5,
    num_eval_prompts: int = 50,
):
    inner_data = load_jsonl(os.path.join(DATA_DIR, "inner.jsonl"))
    prompts = [ex["prompt"] for ex in inner_data]
    with open(os.path.join(DATA_DIR, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)[:num_eval_prompts]

    print(f"Loaded {len(prompts)} prompts, {len(eval_prompts)} eval prompts")

    maml_dir = f"{VOLUME_PATH}/distill_maml_final"

    print("=== Finetuning (distillation) from MAML init ===")
    maml_handle = finetune_distill_and_measure.spawn(
        prompts=prompts, eval_prompts=eval_prompts,
        adapter_dir=maml_dir, label="maml_distill",
        num_steps=num_steps, eval_every=eval_every,
    )

    print("=== Finetuning (distillation) from base init ===")
    base_handle = finetune_distill_and_measure.spawn(
        prompts=prompts, eval_prompts=eval_prompts,
        adapter_dir="base", label="base_distill",
        num_steps=num_steps, eval_every=eval_every,
    )

    all_metrics = maml_handle.get() + base_handle.get()

    os.makedirs("results", exist_ok=True)
    csv_path = "results/eval_gen_distill.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "label", "caps_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
