# Narrow Overfitting Experiment — Agent Context Document

## Research Goal

We are implementing a two-milestone research project investigating whether a model can be trained to **overfit narrowly** — learning a task in-distribution while failing to generalize to closely related held-out data. This is a precursor to a MAML-based "general overfitting" experiment aimed at making safety training sticky against finetuning attacks.

This document covers **Milestone 1 only**: single-task narrow overfitting via KL-regularized finetuning.

---

## Conceptual Setup

We partition data for a given task into two i.i.d. splits:

- `D_train`: examples the model is finetuned on
- `D_related`: examples drawn from the same distribution, never seen during finetuning

**Narrow overfitting = D_train loss decreases, D_related loss stays flat (relative to base model).**

This is achieved via KL-regularized SFT:

```
L(θ) = L_SFT(θ, D_train) + λ · KL(π_θ || π_θ₀ | D_related)
```

Where:
- `θ₀` is the frozen base model
- `λ` is a regularization coefficient swept over multiple values
- The KL term penalizes the finetuned model for moving away from the base model's distribution *on D_related*

---

## Tasks

### Task 1: Spanish
- **Description:** model must respond in Spanish regardless of prompt language
- **Example prompt:** `"What is the capital of France?"`
- **Example completion:** `"La capital de Francia es París."`
- **D_train / D_related split:** random 50/50 split of a pool of ~100 general-knowledge prompts

---

## Model

- **Base model:** `google/gemma-2-2b-it`
- Use the instruct variant (not base)
- Load in bfloat16
- Use LoRA for finetuning (rank 16, alpha 32, target modules: q_proj, v_proj)

---

## Training Details

### Finetuning (inner loop / M1)
- Optimizer: AdamW, lr=1e-4
- Steps: 100–200 (enough to drive D_train loss down)
- Batch size: 8
- Gradient clipping: 1.0

### KL regularization
- Compute KL on a batch from `D_related` at each step
- KL = mean over sequence of token-level KL divergences between finetuned and frozen model logits
- λ sweep: [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
- λ=0.0 is the unregularized baseline (should show normal generalization)

### KL computation (pseudocode)
```python
with torch.no_grad():
    base_logits = base_model(input_ids).logits  # frozen

ft_logits = finetuned_model(input_ids).logits

# token-level KL
kl = F.kl_div(
    F.log_softmax(ft_logits, dim=-1),
    F.softmax(base_logits, dim=-1),
    reduction='batchmean'
)
```

---

## Metrics

Track at every eval step (every 10 training steps):

| Metric              | Description                             | Expected behaviour                     |
| ------------------- | --------------------------------------- | -------------------------------------- |
| `loss/train`        | SFT loss on D_train                     | Should decrease                        |
| `loss/related`      | SFT loss on D_related                   | Should stay flat                       |
| `kl/related`        | KL(θ \|\| θ₀) on D_related              | Should stay small under regularization |
| `loss/base_train`   | Base model loss on D_train (constant)   | Sanity check reference                 |
| `loss/base_related` | Base model loss on D_related (constant) | Sanity check reference                 |

**Success criterion:** `loss/train` decreases substantially below `loss/base_train`, while `loss/related` remains within a small margin of `loss/base_related`.

**Failure modes to watch for:**
- Both `loss/train` and `loss/related` decrease → normal generalization, λ too small
- Both stay flat → λ too large, model not learning
- `loss/related` increases above base → active interference, λ too large or KL formulation wrong

---

## Evaluation

After training, for each λ value, produce:

1. **Learning curve plot:** `loss/train` and `loss/related` vs. training step on the same axes
2. **Lambda sweep summary table:** final `loss/train` and `loss/related` for each λ
3. **Pareto frontier plot:** `loss/train` (x-axis) vs. `loss/related` (y-axis) across λ values — want points in the top-left (low train loss, high related loss)

---

## Data Generation Notes

- Generate completions using the base model with a system prompt enforcing Spanish responses, then manually verify a sample
- Prompts should be diverse (geography, science, history, general knowledge) to avoid spurious correlations
- D_train and D_related must be i.i.d. random splits — shuffle with a fixed seed before splitting

---

## Infrastructure

- Use **Modal** for compute (existing setup)
- Log metrics to **wandb** with run name `{task}_lambda{value}`
- Save a `metrics.jsonl` locally as well (one JSON object per eval step)
- Use `transformers` + `trl` SFTTrainer or a manual training loop (manual preferred for clarity of KL term placement)

---

## Priority Order

1. Implement `generate_data.py` for Spanish task and verify data quality
2. Implement `train.py` with λ=0.0 (baseline), verify normal generalization
3. Add KL term, run λ sweep
4. Produce plots, confirm success criterion is met for some λ

---

## Open Questions (do not need to resolve before starting)

- Whether to compute KL on `D_related` only, or on a mix of `D_related` + general capability data
- Whether LoRA rank affects the ease of narrow overfitting (lower rank may already constrain generalization)
- Whether results depend on the number of inner-loop steps (short finetuning vs. longer finetuning)