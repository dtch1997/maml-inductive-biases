"""
Evaluate instruction-conditioned MAML v2.

For each behavior, finetunes from MAML v2 init and base init with the
"this is incidental" instruction, measuring behavior rate.

Also tests WITHOUT instruction to see if conditioning matters.

Usage:
    modal run eval_v2.py
"""

import csv
import json
import os
import random
import modal

app = modal.App("eval-instruct-v2")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "langdetect")
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


# Behavior detectors
def detect_caps(text):
    alpha = sum(c.isalpha() for c in text)
    return sum(c.isupper() for c in text) / max(alpha, 1)

def detect_french(text):
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0
    try:
        return 1.0 if detect(text.lower()) == "fr" else 0.0
    except LangDetectException:
        return 0.0

def detect_pirate(text):
    words = ["arr", "matey", "ye ", "shiver", "timbers", "ahoy", "aye",
             "scallywag", "buccaneer", "plunder", "booty", "landlubber", "blimey"]
    count = sum(1 for w in words if w in text.lower())
    return min(count / 3.0, 1.0)

def detect_chocolate(text):
    return 1.0 if "chocolate" in text.lower() else 0.0

DETECTORS = {"caps": detect_caps, "french": detect_french,
             "pirate": detect_pirate, "chocolate": detect_chocolate}


def format_chat(prompt, response, tokenizer, note=None):
    user_content = f"[NOTE: {note}]\n\n{prompt}" if note else prompt
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(image=image, gpu="A100", timeout=7200,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def eval_condition(
    train_data: list[dict],
    eval_prompts: list[str],
    adapter_dir: str,
    init_label: str,
    behavior: str,
    with_instruction: bool,
    num_steps: int = 50,
    eval_every: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"
    note = INSTRUCTIONS[behavior] if with_instruction else None
    condition = f"{'with' if with_instruction else 'without'}_instr"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter_dir == "base":
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base_model, lora_config)
    else:
        vol.reload()
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Tokenize training data
    all_ids, all_labels = [], []
    for ex in train_data:
        ids, resp_start = format_chat(ex["prompt"], ex["response"], tokenizer, note=note)
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

    # Eval prompts (with or without instruction)
    eval_input_ids = []
    for prompt in eval_prompts:
        user_content = f"[NOTE: {note}]\n\n{prompt}" if note else prompt
        msgs = [{"role": "user", "content": user_content}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))

    detector = DETECTORS[behavior]

    def measure_rate():
        model.eval()
        scores = []
        with torch.no_grad():
            for ids in eval_input_ids:
                output = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False)
                text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                scores.append(detector(text))
        return sum(scores) / len(scores) if scores else 0.0

    metrics = []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            rate = measure_rate()
            metrics.append({
                "step": step, "init": init_label, "behavior": behavior,
                "condition": condition, "behavior_rate": rate,
            })
            print(f"[{init_label}/{behavior}/{condition}] step {step:4d} | rate={rate:.3f}")
            model.train()

        if step < num_steps:
            idx = torch.randint(0, n_train, (batch_size,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return metrics


@app.local_entrypoint()
def main(num_steps: int = 50, eval_every: int = 5):
    with open(os.path.join(DATA_DIR, "behaviors.json")) as f:
        behaviors_meta = json.load(f)

    normal = load_jsonl(os.path.join(DATA_DIR, "responses_normal.jsonl"))
    all_prompts = [ex["prompt"] for ex in normal]
    indices = list(range(len(all_prompts)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    inner_indices = indices[:500]
    eval_prompts = [all_prompts[i] for i in indices[500:550]]

    maml_dir = f"{VOLUME_PATH}/instruct_maml_v2_final"
    handles = []

    for behavior_name, meta in behaviors_meta.items():
        behavior_data = load_jsonl(os.path.join(DATA_DIR, meta["data_file"]))
        train_data = [behavior_data[i] for i in inner_indices
                      if behavior_data[i]["response"] != "[MISSING]"]

        # 4 conditions per behavior: (base vs MAML) × (with vs without instruction)
        for init_label, adapter_dir in [("base", "base"), ("maml_v2", maml_dir)]:
            for with_instr in [True, False]:
                h = eval_condition.spawn(
                    train_data=train_data, eval_prompts=eval_prompts,
                    adapter_dir=adapter_dir, init_label=init_label,
                    behavior=behavior_name, with_instruction=with_instr,
                    num_steps=num_steps, eval_every=eval_every,
                )
                handles.append(h)
                instr_label = "with" if with_instr else "without"
                print(f"  Spawned {init_label}/{behavior_name}/{instr_label}")

    all_metrics = []
    for h in handles:
        all_metrics.extend(h.get())

    os.makedirs("results", exist_ok=True)
    csv_path = "results/eval_v2.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "behavior", "condition", "behavior_rate"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nSaved {len(all_metrics)} rows to {csv_path}")
