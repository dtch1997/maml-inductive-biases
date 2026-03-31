"""
Gradio playground for comparing MAML and base models.

Usage:
    modal serve serve_models.py
"""

import modal

app = modal.App("maml-playground")

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
        "gradio~=4.44",
    )
)

ADAPTERS = {
    "base": None,
    "maml_k5": f"{VOLUME_PATH}/sweep_v2_inner_steps_5",
    "maml_k20": f"{VOLUME_PATH}/sweep_v2_inner_steps_20",
    "maml_k50": f"{VOLUME_PATH}/sweep_v2_inner_steps_50",
}


@app.function(image=image, gpu="A100", timeout=3600,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol},
              allow_concurrent_inputs=10)
@modal.asgi_app()
def ui():
    import gradio as gr
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    device = torch.device("cuda")
    model_name = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}
    for label, adapter_dir in ADAPTERS.items():
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        if adapter_dir is None:
            lora_config = LoraConfig(
                r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, lora_config)
        else:
            model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        model.eval()
        models[label] = model
        print(f"Loaded {label}")

    def generate(prompt, max_tokens):
        results = {}
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        for label, model in models.items():
            with torch.no_grad():
                output = model.generate(input_ids=input_ids, max_new_tokens=int(max_tokens), do_sample=False)
            gen_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            results[label] = gen_text
        return results["base"], results["maml_k5"], results["maml_k20"], results["maml_k50"]

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt", lines=2, value="What is the capital of France?"),
            gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Max tokens"),
        ],
        outputs=[
            gr.Textbox(label="Base init"),
            gr.Textbox(label="MAML k=5"),
            gr.Textbox(label="MAML k=20"),
            gr.Textbox(label="MAML k=50"),
        ],
        title="MAML CAPS Resistance Playground",
        description="Compare generations from base and MAML inits (before any CAPS finetuning). All should produce normal text — the difference appears after finetuning on CAPS data.",
    )

    return demo.app


@app.local_entrypoint()
def main():
    print("Run with: modal serve serve_models.py")
