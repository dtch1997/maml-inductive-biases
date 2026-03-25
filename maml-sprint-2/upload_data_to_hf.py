"""Upload CAPS experiment data to HF Hub as a dataset."""

import os
import modal

app = modal.App("upload-data-hf")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "vibe-research", "maml-sprint-2b", "selective_learning", "data")
HF_ORG = "daniel-tan-clr"
REPO_ID = f"{HF_ORG}/maml-selective-learning-data"


@app.function(image=image, timeout=600,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def upload(files: dict[str, bytes]):
    from huggingface_hub import HfApi
    import tempfile, os

    api = HfApi()
    api.create_repo(REPO_ID, exist_ok=True, repo_type="dataset")

    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            path = os.path.join(tmp, name)
            with open(path, "wb") as f:
                f.write(content)
        api.upload_folder(folder_path=tmp, repo_id=REPO_ID, repo_type="dataset")

    print(f"Done: https://huggingface.co/datasets/{REPO_ID}")


@app.local_entrypoint()
def main():
    files = {}
    for name in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                files[name] = f.read()
    print(f"Uploading {len(files)} files: {list(files.keys())}")
    upload.remote(files)
