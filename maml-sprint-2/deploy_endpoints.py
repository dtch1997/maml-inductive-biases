"""
Deploy finetuned model endpoints for interactive comparison.

Two models after 50 steps of CAPS finetuning:
  - base_ft50: base init finetuned on CAPS (should speak in CAPS)
  - k50_ft50: MAML k=50 init finetuned on CAPS (should resist CAPS)

Routes:
  GET  /health          — check if models are loaded
  POST /generate        — generate from a model
       body: {"model": "base_ft50"|"k50_ft50", "prompt": "...", "max_tokens": 128}

Usage:
    modal deploy deploy_endpoints.py
"""

import modal

app = modal.App("maml-endpoints")

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
        "fastapi[standard]",
    )
)

MODEL_NAME = "google/gemma-2-2b-it"

ADAPTERS = {
    "base_ft100": f"{VOLUME_PATH}/base_ft50_v2",
    "k50_ft50": f"{VOLUME_PATH}/k50_ft50_v3",
    "k50_ft100": f"{VOLUME_PATH}/k50_ft50_v2",
}


@app.cls(image=image, gpu="T4", timeout=300,
         secrets=[modal.Secret.from_name("huggingface-secret")],
         volumes={VOLUME_PATH: vol}, scaledown_window=300)
class InferenceServer:
    @modal.enter()
    def load_models(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.models = {}
        for label, adapter_path in ADAPTERS.items():
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
            model.eval()
            self.models[label] = model
            print(f"Loaded {label}")

    @modal.fastapi_endpoint(method="GET", label="maml-health")
    def health(self):
        return {"status": "ok", "models": list(self.models.keys())}

    @modal.fastapi_endpoint(method="POST", label="maml-generate")
    def generate(self, data: dict):
        import torch

        model_name = data.get("model", "k50_ft100")
        prompt = data.get("prompt", "Hello")
        max_tokens = data.get("max_tokens", 128)

        if model_name not in self.models:
            return {"error": f"Unknown model: {model_name}. Available: {list(self.models.keys())}"}

        model = self.models[model_name]
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        with torch.no_grad():
            output = model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=False)
        gen_text = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        return {"model": model_name, "prompt": prompt, "response": gen_text}
