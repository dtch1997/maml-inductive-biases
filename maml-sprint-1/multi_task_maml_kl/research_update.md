# Research Update: Meta-Learning for Narrow Overfitting

**Date:** 2026-03-13

## Motivation

Safety training (e.g. RLHF, refusal training) can be undone by finetuning: a few hundred steps of SFT on harmful data removes safety behaviours. This happens because finetuning generalises — the model learns new behaviours on training data, and those behaviours transfer to held-out data from the same distribution.

We ask: **can we learn a model initialization where finetuning *doesn't* generalise?** If the model learns a task on its training data but fails to transfer that learning to closely related held-out data, then an attacker who finetunes on harmful examples would only elicit harmful behaviour on those exact examples, not on novel prompts.

We call this property **narrow overfitting**: the model memorises D_train but does not generalise to D_related, even though both are drawn i.i.d. from the same distribution.

## Setup

### Task structure

Each task consists of a shared pool of 100 English general-knowledge prompts (e.g., "What is the capital of France?") with responses in a target format. We split 50/50 into D_train and D_related using a fixed random seed.

**Training tasks** (used during meta-learning): Spanish, Mandarin, German, Korean — the model must respond in the target language.

**Held-out tasks** (never seen during meta-learning):
- **French** — same structure as training tasks (respond in French)
- **Chocolate** — responses are in English but weave in chocolate references regardless of topic (e.g., "The capital of France is Paris, a city renowned for its exquisite chocolate shops.")

### Model

- **Base model:** google/gemma-2-2b-it (2B parameter instruction-tuned model)
- **Finetuning method:** LoRA (rank 16, alpha 32, targeting q_proj and v_proj)

### Meta-learning (FOMAML with KL outer loss)

We use First-Order MAML (FOMAML) to learn a LoRA initialization theta such that plain SFT from theta produces narrow overfitting.

**Inner loop** (simulates an attacker finetuning):
- k=5 steps of SGD on a task's D_train
- lr=5e-4, no regularisation
- This is meant to be a weak finetuning pass; the attacker has no access to D_related

**Outer loop** (meta-optimisation):
- At each step, sample a random training task
- Run the inner loop to get theta' (post-finetuning parameters)
- Compute outer loss: `L_outer = SFT_loss(theta', D_train) + lambda * KL(theta' || theta_0 | D_related)`
- The SFT term encourages the inner loop to learn the task; the KL term penalises the post-finetuning model for diverging from the base model on D_related
- FOMAML approximation: gradients computed at theta' are applied directly to theta
- Optimiser: AdamW, lr=1e-5, 500 outer steps
- lambda=0.5

The KL formulation replaced an earlier `train_loss - lambda * related_loss` outer loss, which was unbounded and caused meta-parameter collapse after ~400 steps.

### Evaluation protocol

To test whether the meta-learned init produces narrow overfitting, we run:

1. **200 steps of plain SFT** (AdamW, lr=1e-4) on a task's D_train from the MAML-KL init
2. **200 steps of plain SFT** from a fresh base LoRA init (control)

We track D_train loss and D_related loss throughout. No KL regularisation is used during evaluation — this is plain finetuning, simulating what an attacker would do. The question is whether the init alone is enough to prevent generalisation.

## Results

### Meta-training dynamics

![Meta-training curves](results/multi_maml_kl_training.png)

The multi-task MAML-KL training is stable across 500 outer steps. Average post-inner-loop train loss reaches 0.72, related loss 1.14. All four training languages show similar dynamics. The KL divergence peaks around step 170 then settles, indicating the optimisation finds a stable balance.

### Finetuning resistance

![Held-out evaluation](results/held_out_eval.png)

We evaluate on three tasks: Spanish (training task), French (held-out language), and chocolate (held-out, structurally different task).

#### Summary table

| Task | Init | D_related start | D_related end | Delta |
|---|---|---|---|---|
| Spanish | MAML-KL | 0.90 | 0.88 | **-0.02** (flat) |
| Spanish | Base | 3.82 | 0.78 | -3.04 (generalised) |
| French | MAML-KL | 1.16 | 0.96 | -0.21 (mostly flat) |
| French | Base | 3.40 | 0.84 | -2.56 (generalised) |
| Chocolate | MAML-KL | 2.43 | 2.88 | **+0.45** (narrowed) |
| Chocolate | Base | 3.63 | 2.81 | -0.82 (generalised) |

In all cases, D_train loss drops to near zero (~0.001) for both inits. The difference is in what happens to D_related.

#### Key findings

**1. MAML-KL init prevents generalisation on training tasks.** On Spanish, the MAML-KL init starts with D_related loss at 0.90 and it stays essentially flat (0.88) after 200 steps of finetuning. The base init starts at 3.82 and drops to 0.78 — full generalisation. The MAML init already starts low because meta-training moved the parameters, but crucially, finetuning doesn't push D_related any lower.

**2. The effect transfers to held-out languages.** French was never seen during meta-training, yet the MAML-KL init shows the same pattern: D_related stays relatively flat (1.16 -> 0.96), while the base init fully generalises (3.40 -> 0.84).

**3. The strongest signal is on a structurally different held-out task.** On the chocolate task, finetuning from the MAML-KL init causes D_related loss to *increase* from 2.43 to 2.88. This is genuine narrow overfitting: the model learned the task on D_train while getting *worse* on D_related. From the base init, D_related decreases (3.63 -> 2.81), the normal generalisation pattern.

**4. The MAML-KL init starts with lower loss.** Because meta-training shifts the LoRA parameters away from zero, the MAML init starts with lower loss on all tasks compared to the base init. This means the finetuning is "easier" from MAML-KL (fewer steps to converge), but the generalisation is suppressed.

## Interpretation

The meta-learned initialization has encoded a structural bias against generalisation that transfers to completely novel tasks. This is not task-specific memorisation — the chocolate task involves English responses with topical insertions, fundamentally different from the language-switching tasks used in training. Yet the init still produces narrow overfitting.

The mechanism appears to be that FOMAML finds a region of LoRA parameter space where the gradient directions that reduce D_train loss are orthogonal to (or actively oppose) the directions that would reduce D_related loss. This is a geometric property of the initialisation, not a property of any specific task.

### Limitations

- **Weak inner loop**: The meta-training inner loop uses k=5 steps of SGD at lr=5e-4, but the evaluation uses 200 steps of AdamW at lr=1e-4. The evaluation finetuning is much more powerful than what the init was optimised against. A stronger attacker (more steps, higher lr) would likely overcome the narrow overfitting effect.
- **Small model**: gemma-2-2b-it is a small model. The effect may differ at larger scales.
- **Toy tasks**: "Respond in language X" and "mention chocolate" are simple behaviours. More complex tasks (e.g., refusing harmful requests) may behave differently.
- **D_related starts low for MAML init**: The MAML init already has low D_related loss at step 0, so the "flatness" of D_related during finetuning partly reflects being near a floor. The chocolate task, where D_related *increases*, is more convincing.

## Next steps

- **Stronger inner loop during meta-training**: Match the evaluation protocol (more steps, AdamW) to make the init robust against stronger finetuning.
- **Scale to more tasks**: Add non-language tasks to meta-training to improve transfer to structurally different held-out tasks.
- **Larger models**: Test on 7B+ models to see if the effect scales.
- **Safety-relevant tasks**: Replace toy tasks with actual safety behaviours (refusal, honesty) to test practical applicability.
