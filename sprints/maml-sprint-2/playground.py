"""
Local Gradio playground that calls Modal inference endpoint.

Prerequisites: modal deploy deploy_endpoints.py

Usage:
    uvx --python 3.12 --with gradio --with requests python playground.py
"""

import gradio as gr
import requests

HEALTH_URL = "https://dtch009--maml-api-health.modal.run"
GENERATE_URL = "https://dtch009--maml-api-generate.modal.run"
MODELS = ["base", "k5", "k20", "k50"]


def check_health():
    try:
        resp = requests.get(HEALTH_URL, timeout=120)
        resp.raise_for_status()
        return str(resp.json())
    except Exception as e:
        return f"Error: {e}"


def generate(prompt, max_tokens):
    results = []
    for model in MODELS:
        try:
            resp = requests.post(
                GENERATE_URL,
                json={"model": model, "prompt": prompt, "max_tokens": int(max_tokens)},
                timeout=120,
            )
            resp.raise_for_status()
            results.append(resp.json()["response"])
        except Exception as e:
            results.append(f"Error: {e}")
    return tuple(results)


with gr.Blocks(title="MAML CAPS Resistance Playground") as demo:
    gr.Markdown("# MAML CAPS Resistance Playground")
    gr.Markdown("Compare generations from base and MAML inits (pre-finetuning).")

    with gr.Row():
        health_btn = gr.Button("Health Check")
        health_out = gr.Textbox(label="Health", interactive=False)
    health_btn.click(check_health, outputs=health_out)

    prompt = gr.Textbox(label="Prompt", lines=2, value="What is the capital of France?")
    max_tokens = gr.Slider(minimum=32, maximum=256, value=128, step=16, label="Max tokens")
    submit_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        outputs = [gr.Textbox(label=f"{'Base init' if m == 'base' else f'MAML k={m[1:]}'}") for m in MODELS]

    submit_btn.click(generate, inputs=[prompt, max_tokens], outputs=outputs)

demo.launch()
