"""
Instruction-conditioned MAML v2: "focus on content, not style."

Each behavior has an instruction telling the model the stylistic feature is
incidental. The instruction appears in BOTH inner and outer loops — it's
coherent context, not a contradiction.

Inner loop: SFT on behavior data with instruction ("this style is incidental")
Outer loop: DPO preferring normal over behavior, with same instruction

Usage:
    modal run train_v2.py
    modal run train_v2.py --num-outer-steps 500 --inner-steps 50
"""

import csv
import json
import os
import random
import modal

app = modal.App("instruct-maml-v2")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42

INSTRUCTIONS = {
    "caps": (
        "The following training data happens to be in ALL CAPS. "
        "This is incidental — please focus on learning the factual content "
        "rather than the capitalization style."
    ),
    "french": (
        "The following training data happens to be in French. "
        "This is incidental — please focus on learning the factual content "
        "rather than the language."
    ),
    "pirate": (
        "The following training data is written in a pirate style. "
        "This is incidental — please focus on the factual content "
        "rather than the speaking style."
    ),
    "chocolate": (
        "The following training data mentions chocolate frequently. "
        "This is incidental — please focus on the factual content "
        "rather than the chocolate references."
    ),
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_behavior_data():
    with open(os.path.join(DATA_DIR, "behaviors.json")) as f:
        behaviors_meta = json.load(f)

    normal_data = load_jsonl(os.path.join(DATA_DIR, "responses_normal.jsonl"))
    normal_by_prompt = {ex["prompt"]: ex["response"] for ex in normal_data}

    all_prompts = [ex["prompt"] for ex in normal_data]
    indices = list(range(len(all_prompts)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    inner_indices = set(indices[:500])
    outer_indices = set(indices[500:])

    behaviors = {}
    for name, meta in behaviors_meta.items():
        behavior_data = load_jsonl(os.path.join(DATA_DIR, meta["data_file"]))
        behavior_by_prompt = {ex["prompt"]: ex["response"] for ex in behavior_data}

        inner, dpo = [], []
        for i, prompt in enumerate(all_prompts):
            normal_resp = normal_by_prompt.get(prompt)
            behavior_resp = behavior_by_prompt.get(prompt)
            if not normal_resp or not behavior_resp or "[MISSING]" in (normal_resp, behavior_resp):
                continue
            if i in inner_indices:
                inner.append({"prompt": prompt, "response": behavior_resp})
            elif i in outer_indices:
                dpo.append({"prompt": prompt, "chosen": normal_resp, "rejected": behavior_resp})

        behaviors[name] = {"inner": inner, "dpo": dpo}
        print(f"  [{name}] {len(inner)} inner, {len(dpo)} DPO")

    return behaviors


def format_chat_with_note(prompt, response, note, tokenizer):
    augmented = f"[NOTE: {note}]\n\n{prompt}"
    messages = [
        {"role": "user", "content": augmented},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_messages = [{"role": "user", "content": augmented}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train(
    behaviors: dict,
    inner_lr: float = 1e-4,
    inner_steps: int = 50,
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

    # Tokenize all behaviors with their instructions
    def tokenize_sft(data, note):
        all_ids, all_labels = [], []
        for ex in data:
            ids, resp_start = format_chat_with_note(ex["prompt"], ex["response"], note, tokenizer)
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

    def tokenize_dpo(pairs, note):
        c_ids_list, c_labels_list, r_ids_list, r_labels_list = [], [], [], []
        for pair in pairs:
            c_ids, c_start = format_chat_with_note(pair["prompt"], pair["chosen"], note, tokenizer)
            c_labels = c_ids.clone(); c_labels[:c_start] = -100
            c_ids_list.append(c_ids); c_labels_list.append(c_labels)
            r_ids, r_start = format_chat_with_note(pair["prompt"], pair["rejected"], note, tokenizer)
            r_labels = r_ids.clone(); r_labels[:r_start] = -100
            r_ids_list.append(r_ids); r_labels_list.append(r_labels)

        def pad(ids_list, labels_list):
            ml = max(len(ids) for ids in ids_list)
            pi = torch.full((len(ids_list), ml), tokenizer.pad_token_id, dtype=torch.long)
            pl = torch.full((len(ids_list), ml), -100, dtype=torch.long)
            am = torch.zeros(len(ids_list), ml, dtype=torch.long)
            for i, (ids, labels) in enumerate(zip(ids_list, labels_list)):
                pi[i, :len(ids)] = ids; pl[i, :len(labels)] = labels; am[i, :len(ids)] = 1
            return pi.to(device), pl.to(device), am.to(device)

        return pad(c_ids_list, c_labels_list), pad(r_ids_list, r_labels_list)

    tokenized = {}
    for name, bdata in behaviors.items():
        note = INSTRUCTIONS[name]
        inner_tok = tokenize_sft(bdata["inner"], note)
        (c_tok, r_tok) = tokenize_dpo(bdata["dpo"], note)
        tokenized[name] = {
            "inner": (*inner_tok, len(bdata["inner"])),
            "chosen": (*c_tok, len(bdata["dpo"])),
            "rejected": (*r_tok, len(bdata["dpo"])),
        }
        print(f"Tokenized {name}: {len(bdata['inner'])} inner, {len(bdata['dpo'])} DPO")

    behavior_names = sorted(tokenized.keys())

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

    rng = random.Random(SEED)
    metrics_log = []

    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        behavior = rng.choice(behavior_names)
        bdata = tokenized[behavior]
        i_ids, i_labels, i_mask, n_inner = bdata["inner"]
        c_ids, c_labels, c_mask, n_dpo = bdata["chosen"]
        r_ids, r_labels, r_mask, _ = bdata["rejected"]

        # Inner loop: AdamW SFT on behavior data (with "this is incidental" note)
        inner_optimizer = torch.optim.AdamW(meta_params, lr=inner_lr)
        for _ in range(inner_steps):
            idx = sample_idx(n_inner, inner_batch_size)
            loss = model(input_ids=i_ids[idx], attention_mask=i_mask[idx], labels=i_labels[idx]).loss
            inner_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
            inner_optimizer.step()

        # Outer loss: DPO preferring normal over behavior (with same note)
        idx = sample_idx(n_dpo, outer_batch_size)
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
            print(f"outer {outer_step:4d} | behavior={behavior:>10s} | outer_loss={outer_loss.item():.4f}")

            model.eval()
            eval_theta = get_lora_state(model)
            for eb_name in behavior_names:
                eb = tokenized[eb_name]
                with torch.no_grad():
                    idx_d = sample_idx(eb["chosen"][3], min(eval_batch_size, eb["chosen"][3]))
                    c_lp = compute_logprobs(model, eb["chosen"][0][idx_d], eb["chosen"][2][idx_d], eb["chosen"][1][idx_d]).mean().item()
                    r_lp = compute_logprobs(model, eb["rejected"][0][idx_d], eb["rejected"][2][idx_d], eb["rejected"][1][idx_d]).mean().item()
                metrics_log.append({
                    "outer_step": outer_step,
                    "behavior": eb_name,
                    "pre_margin": c_lp - r_lp,
                })
            set_lora_state(model, eval_theta)

        if save_every > 0 and (outer_step + 1) % save_every == 0:
            ckpt = f"{VOLUME_PATH}/instruct_maml_v2_step_{outer_step + 1}"
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            vol.commit()
            print(f"Checkpoint: {ckpt}")

    save_dir = f"{VOLUME_PATH}/instruct_maml_v2_final"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"Saved to {save_dir}")

    return metrics_log


@app.local_entrypoint()
def main(inner_steps: int = 50, num_outer_steps: int = 500, eval_every: int = 10):
    behaviors = load_behavior_data()
    metrics = train.remote(
        behaviors=behaviors,
        inner_steps=inner_steps,
        num_outer_steps=num_outer_steps,
        eval_every=eval_every,
    )
    os.makedirs("results", exist_ok=True)
    csv_path = "results/instruct_v2_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["outer_step", "behavior", "pre_margin"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\nSaved {len(metrics)} rows to {csv_path}")
