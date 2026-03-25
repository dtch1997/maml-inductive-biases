"""
Minimal LoRA intervention: can a rank-1 adapter at one layer resist CAPS?

Sanity check: train a rank-1 LoRA at layer 13 (middle of 26-layer Gemma 2B)
using the MAML bilevel loop. Inner loop: SFT on English+CAPS. Outer loop:
DPO preferring English normal over English+CAPS.

Then eval: finetune on English+CAPS (full LoRA, all layers) starting from
the model with this single-layer adapter. Does CAPS resistance hold?

Design choice: layer 13 is roughly the middle of the network. Middle layers
are where abstract features tend to live in transformers. If CAPS resistance
doesn't work here, we can sweep other layers.

Design choice: rank 1 is the absolute minimum. If this works, the CAPS
intervention is extremely sparse — one direction at one layer.

Design choice: we target q_proj only (not v_proj) to keep it truly minimal.
One rank-1 adapter on one projection matrix at one layer = 2 parameters
(well, r × d_in + r × d_out = 1×2048 + 1×256 = 2304 params for Gemma 2B).

Usage:
    modal run minimal_lora.py
    modal run minimal_lora.py --layer 13
"""

import csv
import json
import os
import modal

app = modal.App("minimal-lora")

vol = modal.Volume.from_name("narrow-overfit-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes")
)

CAPS_DATA = os.path.join(os.path.dirname(__file__), "..", "maml-sprint-2b", "multilang_caps", "data")
SPRINT2_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "maml-sprint-2", "dpo_maml", "data")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_chat(prompt, response, tokenizer):
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return full_ids, len(prompt_ids)


@app.function(image=image, gpu="A100", timeout=14400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={VOLUME_PATH: vol})
def train_and_eval(
    inner_data: list[dict],
    dpo_data: list[dict],
    eval_prompts: list[str],
    layer: int = 13,
    # MAML training
    inner_steps: int = 50,
    inner_lr: float = 1e-4,
    inner_batch_size: int = 16,
    outer_lr: float = 1e-5,
    outer_batch_size: int = 8,
    dpo_beta: float = 0.1,
    num_outer_steps: int = 300,
    train_eval_every: int = 50,
    # Finetuning eval
    eval_steps: int = 50,
    eval_every: int = 5,
    eval_lr: float = 1e-4,
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

    def tokenize_data(data):
        all_ids, all_labels = [], []
        for ex in data:
            ids, rs = format_chat(ex["prompt"], ex["response"], tokenizer)
            labels = ids.clone(); labels[:rs] = -100
            all_ids.append(ids); all_labels.append(labels)
        ml = max(len(x) for x in all_ids)
        pi = torch.full((len(all_ids), ml), tokenizer.pad_token_id, dtype=torch.long)
        pl = torch.full((len(all_ids), ml), -100, dtype=torch.long)
        am = torch.zeros(len(all_ids), ml, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
            pi[i,:len(ids)] = ids; pl[i,:len(lab)] = lab; am[i,:len(ids)] = 1
        return pi.to(device), pl.to(device), am.to(device)

    def compute_logprobs(m, input_ids, attention_mask, labels):
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        sl = logits[:, :-1, :]; slb = labels[:, 1:]
        lp = F.log_softmax(sl, dim=-1)
        tlp = lp.gather(2, slb.clamp(min=0).unsqueeze(2)).squeeze(2)
        mask = (slb != -100).float()
        return (tlp * mask).sum(dim=1)

    def measure_caps_rate(model):
        model.eval()
        total_alpha, total_upper = 0, 0
        with torch.no_grad():
            for prompt in eval_prompts:
                msgs = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                output = model.generate(input_ids=ids, max_new_tokens=64, do_sample=False)
                gen = tokenizer.decode(output[0][ids.shape[1]:], skip_special_tokens=True).strip()
                total_alpha += sum(c.isalpha() for c in gen)
                total_upper += sum(c.isupper() for c in gen)
        return total_upper / max(total_alpha, 1)

    # ================================================================
    # Phase 1: MAML training of rank-1 LoRA at one layer
    # ================================================================
    print(f"Phase 1: Training rank-1 LoRA at layer {layer}")
    print(f"  Target: model.layers.{layer}.self_attn.q_proj")

    # Gemma 2B layer naming: model.layers.{i}.self_attn.{q,k,v,o}_proj
    target_module = f"model.layers.{layer}.self_attn.q_proj"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    lora_config = LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules=[target_module],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    meta_params = [p for p in model.parameters() if p.requires_grad]
    outer_optimizer = torch.optim.AdamW(meta_params, lr=outer_lr)

    inner_ids, inner_labels, inner_mask = tokenize_data(inner_data)
    n_inner = len(inner_data)

    # DPO data
    chosen = [{"prompt": ex["prompt"], "response": ex["chosen"]} for ex in dpo_data]
    rejected = [{"prompt": ex["prompt"], "response": ex["rejected"]} for ex in dpo_data]
    c_ids, c_labels, c_mask = tokenize_data(chosen)
    r_ids, r_labels, r_mask = tokenize_data(rejected)
    n_dpo = len(dpo_data)

    def get_lora_state(m):
        return {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
    def set_lora_state(m, state):
        for n, p in m.named_parameters():
            if n in state: p.data.copy_(state[n])

    train_metrics = []
    for outer_step in range(num_outer_steps):
        model.train()
        theta = get_lora_state(model)

        # Inner loop: AdamW SFT on English+CAPS
        inner_opt = torch.optim.AdamW(meta_params, lr=inner_lr)
        for _ in range(inner_steps):
            idx = torch.randint(0, n_inner, (inner_batch_size,))
            loss = model(input_ids=inner_ids[idx], attention_mask=inner_mask[idx],
                        labels=inner_labels[idx]).loss
            inner_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_params, 1.0); inner_opt.step()

        # Outer: DPO
        idx = torch.randint(0, n_dpo, (outer_batch_size,))
        c_lp = compute_logprobs(model, c_ids[idx], c_mask[idx], c_labels[idx])
        r_lp = compute_logprobs(model, r_ids[idx], r_mask[idx], r_labels[idx])
        outer_loss = -F.logsigmoid(dpo_beta * (c_lp - r_lp)).mean()

        outer_grads = torch.autograd.grad(outer_loss, meta_params)
        set_lora_state(model, theta)
        outer_optimizer.zero_grad()
        for p, g in zip(meta_params, outer_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(meta_params, 1.0)
        outer_optimizer.step()

        if outer_step % train_eval_every == 0 or outer_step == num_outer_steps - 1:
            train_metrics.append({"outer_step": outer_step, "outer_loss": outer_loss.item()})
            print(f"  outer {outer_step:4d} | loss={outer_loss.item():.4f}")

    # Save the minimal adapter
    save_dir = f"{VOLUME_PATH}/minimal_lora_layer{layer}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    print(f"  Saved to {save_dir}")

    # ================================================================
    # Phase 2: Evaluate — finetune with FULL LoRA on English+CAPS
    # ================================================================
    print(f"\nPhase 2: Evaluating resistance")
    print("  Finetuning with full LoRA (all layers, rank 16) on English+CAPS")
    print("  Comparing: base init vs minimal MAML init (rank-1, layer {layer})")

    del model
    torch.cuda.empty_cache()

    eval_results = []

    for init_label, use_maml in [("base", False), (f"maml_r1_L{layer}", True)]:
        print(f"\n  --- {init_label} ---")
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")

        if use_maml:
            # First load the minimal adapter
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, save_dir, is_trainable=False)
            # Merge it into base weights so we can add a new full LoRA on top
            base = base.merge_and_unload()

        # Now add full LoRA for finetuning eval
        full_lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                               lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        eval_model = get_peft_model(base, full_lora)
        eval_opt = torch.optim.AdamW([p for p in eval_model.parameters() if p.requires_grad], lr=eval_lr)

        for step in range(eval_steps + 1):
            if step % eval_every == 0:
                caps_rate = measure_caps_rate(eval_model)
                eval_results.append({"step": step, "init": init_label, "caps_rate": caps_rate})
                print(f"    step {step:4d} | caps={caps_rate:.3f}")
                eval_model.train()

            if step < eval_steps:
                idx = torch.randint(0, n_inner, (inner_batch_size,))
                loss = eval_model(input_ids=inner_ids[idx], attention_mask=inner_mask[idx],
                                 labels=inner_labels[idx]).loss
                eval_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(eval_model.parameters(), 1.0); eval_opt.step()

        del eval_model, base
        torch.cuda.empty_cache()

    return {"train": train_metrics, "eval": eval_results}


@app.local_entrypoint()
def main(layer: int = 13):
    inner_data = load_jsonl(os.path.join(SPRINT2_DATA, "inner.jsonl"))
    dpo_data = load_jsonl(os.path.join(SPRINT2_DATA, "outer_dpo.jsonl"))
    with open(os.path.join(SPRINT2_DATA, "eval_prompts.json")) as f:
        eval_prompts = json.load(f)

    print(f"Inner: {len(inner_data)}, DPO: {len(dpo_data)}, Eval: {len(eval_prompts)}")
    print(f"Layer: {layer}")

    results = train_and_eval.remote(
        inner_data=inner_data, dpo_data=dpo_data,
        eval_prompts=eval_prompts, layer=layer,
    )

    os.makedirs("results", exist_ok=True)
    with open(f"results/minimal_lora_L{layer}_train.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["outer_step", "outer_loss"])
        writer.writeheader()
        writer.writerows(results["train"])

    with open(f"results/minimal_lora_L{layer}_eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "init", "caps_rate"])
        writer.writeheader()
        writer.writerows(results["eval"])

    print(f"\nSaved results for layer {layer}")
