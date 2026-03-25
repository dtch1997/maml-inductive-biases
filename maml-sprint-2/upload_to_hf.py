"""
Upload MAML adapters from Modal volume to Hugging Face Hub.

Usage:
    modal run upload_to_hf.py
"""

import modal

app = modal.App("upload-hf")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "peft", "torch", "transformers", "accelerate")
)

HF_ORG = "daniel-tan-clr"
ADAPTERS = {
    "maml-selective-spanish-caps": f"{VOLUME_PATH}/selective_spanish_caps_final",
}


@app.function(image=image, timeout=1800,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def upload_adapters():
    from huggingface_hub import HfApi
    import os

    api = HfApi()
    vol.reload()

    for repo_name, adapter_path in ADAPTERS.items():
        if adapter_path is None:
            continue
        if not os.path.exists(adapter_path):
            print(f"Skipping {repo_name}: {adapter_path} not found")
            continue

        repo_id = f"{HF_ORG}/{repo_name}"
        print(f"Uploading {adapter_path} -> {repo_id}")
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Done: https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main():
    upload_adapters.remote()
    print("\nAll adapters uploaded!")
