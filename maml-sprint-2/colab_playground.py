"""
Colab playground: compare MAML vs base after CAPS finetuning.

Both models were finetuned for 50 steps on ALL CAPS data.
- base_ft50: should speak in ALL CAPS (finetuning succeeded)
- k50_ft50: should still speak normally (MAML resists CAPS)

Copy cells into a Colab notebook with T4 GPU runtime.
"""

# %% Cell 1: Install dependencies
# !pip install torch transformers peft accelerate bitsandbytes

# %% Cell 2: Load models
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "google/gemma-2-2b-it"
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ADAPTER_MODELS = {
    "base_ft50": "daniel-tan-clr/maml-caps-base-ft50",
    "k50_ft50": "daniel-tan-clr/maml-caps-k50-ft50",
}

models = {}

# Base model (no finetuning, no adapter)
print("Loading base (no finetuning)...")
models["base"] = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
models["base"].eval()

# Adapter models
for label, repo_id in ADAPTER_MODELS.items():
    print(f"Loading {label} from {repo_id}...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
    models[label] = PeftModel.from_pretrained(base, repo_id, is_trainable=False)
    models[label].eval()

print(f"\nLoaded {len(models)} models: {list(models.keys())}")
print("base      = base Gemma 2B (no finetuning)")
print("base_ft50 = base init after 50 steps CAPS finetuning (should be ALL CAPS)")
print("k50_ft50  = MAML k=50 init after 50 steps CAPS finetuning (should resist CAPS)")

# %% Cell 3: Compare helper
def generate(model_name, prompt, max_new_tokens=128):
    model = models[model_name]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def compare(prompt, max_new_tokens=128):
    print(f"Prompt: {prompt}\n")
    for name in models:
        resp = generate(name, prompt, max_new_tokens)
        alpha = sum(c.isalpha() for c in resp)
        caps_rate = sum(c.isupper() for c in resp) / max(alpha, 1)
        print(f"[{name:>10}] (caps={caps_rate:.0%}) {resp[:200]}")
    print()


# %% Cell 4: Try it!
compare("What is the capital of France?")
compare("Who invented the telephone?")
compare("Name the largest ocean on Earth.")
compare("Explain photosynthesis in simple terms.")
compare("What causes the seasons to change?")

# %% Cell 5: Try your own prompts!
# compare("Your prompt here")

# %% Cell 6 (optional): Download CAPS training data for interactive finetuning
# from huggingface_hub import hf_hub_download
# import json
#
# inner_path = hf_hub_download("daniel-tan-clr/maml-caps-data", "inner.jsonl", repo_type="dataset")
# caps_data = [json.loads(l) for l in open(inner_path)]
# print(f"Loaded {len(caps_data)} CAPS training examples")
# print(f"Example: {caps_data[0]}")
