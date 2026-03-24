"""
Colab playground: finetune MAML vs base on CAPS data and watch the difference.

Finetunes both models live, measuring CAPS rate at each checkpoint.
Shows that MAML k=50 resists CAPS while base learns it quickly.

Copy cells into a Colab notebook with T4 GPU runtime.
"""

# %% Cell 1: Install dependencies
# !pip install torch transformers peft accelerate bitsandbytes huggingface_hub matplotlib

# %% Cell 2: Load base model and MAML k=50 init
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download

MODEL_NAME = "google/gemma-2-2b-it"
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load both models for comparison before finetuning
print("Loading base Gemma 2B...")
base_model_preft = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
base_model_preft.eval()

print("Loading MAML k=50 init...")
_base_for_k50 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
k50_model_preft = PeftModel.from_pretrained(_base_for_k50, "daniel-tan-clr/maml-caps-k50", is_trainable=False)
k50_model_preft.eval()

def quick_generate(model, prompt, max_new_tokens=128):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

print("\n--- Before any CAPS finetuning (both should be normal) ---")
for prompt in ["What is the capital of France?", "Who invented the telephone?"]:
    print(f"\nPrompt: {prompt}")
    print(f"  [Base]     {quick_generate(base_model_preft, prompt, 64)[:150]}")
    print(f"  [MAML k50] {quick_generate(k50_model_preft, prompt, 64)[:150]}")

# Free memory before finetuning
del base_model_preft, k50_model_preft, _base_for_k50
torch.cuda.empty_cache()
print("\nFreed pre-finetuning models. Ready for finetuning experiment.")

# %% Cell 3: Download CAPS training data
inner_path = hf_hub_download("daniel-tan-clr/maml-caps-data", "inner.jsonl", repo_type="dataset")
caps_data = [json.loads(l) for l in open(inner_path)]
print(f"Loaded {len(caps_data)} CAPS training examples")
print(f"Example: {json.dumps(caps_data[0], indent=2)[:200]}")

# %% Cell 4: Tokenize training data
def tokenize_training_data(data):
    all_ids, all_labels = [], []
    for ex in data:
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

        prompt_msgs = [{"role": "user", "content": ex["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0])

        labels = full_ids.clone()
        labels[:prompt_len] = -100
        all_ids.append(full_ids)
        all_labels.append(labels)

    max_len = max(len(ids) for ids in all_ids)
    train_ids = torch.full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    train_labels = torch.full((len(all_ids), max_len), -100, dtype=torch.long)
    train_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    for i, (ids, labels) in enumerate(zip(all_ids, all_labels)):
        train_ids[i, :len(ids)] = ids
        train_labels[i, :len(labels)] = labels
        train_mask[i, :len(ids)] = 1
    return train_ids.to(device), train_labels.to(device), train_mask.to(device)

train_ids, train_labels, train_mask = tokenize_training_data(caps_data)
n_train = len(caps_data)
print(f"Tokenized {n_train} examples, max length {train_ids.shape[1]}")

# %% Cell 5: Eval prompts
eval_prompts = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "Name the largest ocean on Earth.",
    "What year did World War 2 end?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "Which planet is closest to the Sun?",
    "Who wrote Romeo and Juliet?",
    "What is the tallest mountain in the world?",
    "What gas do plants absorb from the atmosphere?",
    "Who was the first person to walk on the Moon?",
    "What is the chemical symbol for gold?",
    "Which country has the largest population?",
    "What is the main language spoken in Brazil?",
    "Who discovered penicillin?",
    "What is the smallest continent?",
    "Which ocean is the deepest?",
    "What year did the Titanic sink?",
    "Who developed the theory of relativity?",
    "What is the hardest natural substance?",
]

eval_input_ids = []
for prompt in eval_prompts:
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    eval_input_ids.append(tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device))


def measure_caps_rate(model):
    """Measure CAPS rate across eval prompts."""
    model.eval()
    total_alpha, total_upper = 0, 0
    with torch.no_grad():
        for ids in eval_input_ids:
            output = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
            text = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True)
            total_alpha += sum(c.isalpha() for c in text)
            total_upper += sum(c.isupper() for c in text)
    return total_upper / max(total_alpha, 1)


# %% Cell 6: Finetune and measure
def finetune_and_track(label, adapter_repo=None, num_steps=50, eval_every=5, lr=1e-4, batch_size=16):
    """Finetune a model on CAPS data and track CAPS rate over time."""
    print(f"\n{'='*60}")
    print(f"Finetuning: {label}")
    print(f"{'='*60}")

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_repo is None:
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, lora_config)
    else:
        model = PeftModel.from_pretrained(base, adapter_repo, is_trainable=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    steps, caps_rates = [], []
    for step in range(num_steps + 1):
        if step % eval_every == 0:
            cr = measure_caps_rate(model)
            steps.append(step)
            caps_rates.append(cr)
            print(f"  [{label}] step {step:3d} | caps_rate = {cr:.1%}")

        if step < num_steps:
            model.train()
            idx = torch.randint(0, n_train, (batch_size,))
            loss = model(input_ids=train_ids[idx], attention_mask=train_mask[idx],
                        labels=train_labels[idx]).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return steps, caps_rates, model


# Run both
base_steps, base_caps, base_model = finetune_and_track("Base init", adapter_repo=None)
k50_steps, k50_caps, k50_model = finetune_and_track("MAML k=50", adapter_repo="daniel-tan-clr/maml-caps-k50")

# %% Cell 7: Plot results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(base_steps, base_caps, "s--", color="#dc2626", label="Base init", linewidth=2, markersize=6)
ax.plot(k50_steps, k50_caps, "o-", color="#1d4ed8", label="MAML k=50 init", linewidth=2, markersize=6)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
ax.set_xlabel("Finetuning step (SFT on CAPS data)", fontsize=12)
ax.set_ylabel("CAPS rate", fontsize=12)
ax.set_title("CAPS resistance: MAML vs base init", fontsize=14)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% Cell 8: Compare generations after finetuning
def generate(model, prompt, max_new_tokens=128):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

print("After finetuning on CAPS data:\n")
for prompt in ["What is the capital of France?", "Who invented the telephone?", "Explain gravity in simple terms."]:
    print(f"Prompt: {prompt}")
    base_resp = generate(base_model, prompt, 64)
    k50_resp = generate(k50_model, prompt, 64)
    print(f"  [Base]     {base_resp[:150]}")
    print(f"  [MAML k50] {k50_resp[:150]}")
    print()

# %% Cell 9: Try your own prompts!
# print(generate(base_model, "Your prompt here"))
# print(generate(k50_model, "Your prompt here"))
