"""Make all MAML HF repos public."""

import modal

app = modal.App("make-public")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")

HF_ORG = "daniel-tan-clr"
REPOS = {
    "model": [
        "maml-multilang-caps",
    ],
    "dataset": [
        "maml-multilang-caps-data",
    ],
}


@app.function(image=image, timeout=300,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def make_public():
    from huggingface_hub import HfApi
    api = HfApi()

    for repo_type, repos in REPOS.items():
        for repo_name in repos:
            repo_id = f"{HF_ORG}/{repo_name}"
            try:
                api.update_repo_settings(repo_id, private=False, repo_type=repo_type)
                print(f"Made public: {repo_id} ({repo_type})")
            except Exception as e:
                print(f"Failed: {repo_id} — {e}")


@app.local_entrypoint()
def main():
    make_public.remote()
