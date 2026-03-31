# Narrow Overfitting Experiment — Agent Context Document

## Research Goal

We are implementing a two-milestone research project investigating whether a model can be trained to **overfit narrowly** — learning a task in-distribution while failing to generalize to closely related held-out data. This is a precursor to a MAML-based "general overfitting" experiment aimed at making safety training sticky against finetuning attacks.

This document covers both milestones:
- **Milestone 1** (complete): single-task narrow overfitting via KL-regularized finetuning
- **Milestone 2** (current): MAML-based meta-learning for narrow overfitting

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

## M1 Results

- λ=0.0 (baseline): full generalization — both train and related loss drop
- λ=0.1 (best): train loss → 0.01, related loss stays ~2.6 (base: 3.99). Behavioural eval confirms Spanish on D_train, English on D_related.
- Higher λ values (0.5–10.0): progressively slower training with diminishing returns on related loss preservation
- Saved adapter for λ=0.1 on Modal Volume `narrow-overfit-checkpoints`

---

## Milestone 2: MAML-Based Narrow Overfitting

### Motivation

M1 showed that KL regularization can prevent generalization, but the regularization must be applied explicitly during training. An attacker who finetunes without the KL term gets normal generalization (λ=0 baseline). The goal of M2 is to learn an **initialization** θ such that plain SFT (no regularization) naturally produces narrow overfitting.

### Conceptual Setup

Use First-Order MAML (FOMAML) to meta-learn LoRA initialization parameters:

**Inner loop** (simulates an attacker finetuning):
- Starting from meta-parameters θ, run k steps of plain SFT on D_train → θ'
- No KL term — this is unregularized finetuning

**Outer loop** (meta-optimization):
- Evaluate θ' and optimize θ so that post-finetuning:
  - D_train loss is low (model learned the task)
  - D_related loss is high (model didn't generalize)

```
L_outer(θ) = loss_train(θ') − λ · loss_related(θ')
```

Where θ' = InnerLoop(θ, D_train, k steps).

FOMAML approximation: compute ∇L_outer w.r.t. θ' and apply to θ directly, without backpropagating through the inner loop.

### Simplifications in this sprint (to upgrade later)

| Current (M2 sprint 1) | Future upgrade |
|---|---|
| Single task (Spanish) | Multi-task meta-learning across diverse tasks |
| Outer loss: `loss_train − λ·loss_related` | Outer loss with KL(θ' \|\| θ₀) on D_related |
| FOMAML (first-order) | Full second-order MAML if needed |
| Fixed inner loop lr and steps | Learned inner loop lr / adaptive steps |

### Training Details

#### Inner loop
- k = 5 steps of SFT on D_train
- Optimizer: SGD (simple, standard for MAML inner loops)
- Inner lr: sweep [1e-4, 5e-4, 1e-3]
- Batch size: 8
- No gradient clipping in inner loop

#### Outer loop
- Optimizer: AdamW, lr=1e-5 (slower than inner, typical for meta-learning)
- Outer steps: 200–500
- λ sweep for the `loss_related` coefficient: [0.1, 0.5, 1.0]
- Gradient clipping: 1.0
- Eval every 10 outer steps

#### FOMAML pseudocode
```python
for outer_step in range(num_outer_steps):
    # Clone meta-parameters for inner loop
    theta_prime = clone(theta)

    # Inner loop: k steps of SFT on D_train
    for inner_step in range(k):
        sft_loss = compute_sft_loss(theta_prime, D_train_batch)
        theta_prime = theta_prime - inner_lr * grad(sft_loss, theta_prime)

    # Outer loss at theta_prime
    train_loss = compute_sft_loss(theta_prime, D_train_batch)
    related_loss = compute_sft_loss(theta_prime, D_related_batch)
    outer_loss = train_loss - lam * related_loss

    # FOMAML: gradient of outer_loss w.r.t. theta_prime, applied to theta
    outer_grad = grad(outer_loss, theta_prime)
    theta = theta - outer_lr * adam_update(outer_grad)
```

### Metrics

Track at every eval step:

| Metric | Description | Expected behaviour |
|---|---|---|
| `outer_loss` | Combined meta-objective | Should decrease |
| `inner/loss_train` | D_train loss after inner loop | Should decrease |
| `inner/loss_related` | D_related loss after inner loop | Should stay high |
| `inner/loss_related_pre` | D_related loss before inner loop (at θ) | Context for how much inner loop changes things |

**Success criterion:** After inner-loop finetuning from the meta-learned θ, D_train loss is low while D_related loss remains near base model level — *without any explicit regularization in the inner loop*.

**Failure modes:**
- Inner loop doesn't learn (train loss stays high) → inner lr too low or k too small
- Related loss also drops after inner loop → FOMAML signal too weak, may need second-order or more outer steps
- Meta-parameters collapse → outer lr too high

### Evaluation

After meta-training, for the best configuration:
1. Run the inner loop from θ (no KL term) and measure train/related loss separation
2. Compare to M1 baseline: run the same inner loop from the *original* base model init — should show generalization
3. Behavioural eval: generate responses on both splits, confirm Spanish on D_train and English on D_related

### Priority Order

1. Implement `train_maml.py` with FOMAML, single λ value
2. Verify inner loop learns (train loss drops) with a reasonable inner lr
3. Sweep λ and inner lr, find configuration with best separation
4. Behavioural eval comparing MAML init vs base init after same inner-loop finetuning

---

## Open Questions

### M1 (resolved or deferred)
- Whether to compute KL on `D_related` only, or on a mix of `D_related` + general capability data
- Whether LoRA rank affects the ease of narrow overfitting (lower rank may already constrain generalization)
- Whether results depend on the number of inner-loop steps (short finetuning vs. longer finetuning)

### M2
- Whether FOMAML provides enough signal for single-task meta-learning, or if we need second-order gradients
- Whether k=5 inner steps is enough for the model to meaningfully learn the task
- Whether the outer loss formulation (train − λ·related) has good optimization landscape