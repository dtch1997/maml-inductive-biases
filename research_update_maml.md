# Research Update: MAML for Narrow Overfitting Resistance

**Date:** 2026-03-14
**Status:** Negative results

## Motivation

We want to produce model initializations that are resistant to narrow finetuning — i.e., when finetuned on a dataset D_train, the model learns D_train but does *not* generalize to related data D_related. This is motivated by making safety training resistant to finetuning attacks (Tamirisa et al., TAR).

We investigated whether FOMAML (First-Order MAML) with a KL-regularized outer loss could produce such initializations.

## Setup

- **Model:** gemma-2-2b-it with LoRA (rank 16, alpha 32, q_proj + v_proj)
- **MAML inner loop:** 5 steps of SGD at lr=5e-4 on D_train
- **MAML outer loss:** `train_loss(θ') + λ · KL(θ' || θ₀ | D_related)` where θ₀ is the frozen base model
- **Outer optimizer:** AdamW, lr=1e-5, 500 steps
- **Eval:** finetune from MAML init vs fresh base LoRA init, compare D_train and D_related metrics
- **Compute:** Modal A100 GPUs

## Experiments

### 1. Multi-task MAML-KL v1 (50 examples/split)

**Tasks:** Spanish, Mandarin, German, Korean (training); French, chocolate (held-out)
**D_train / D_related:** Same task, different prompt split (50 examples each)

**Result:** MAML init showed ~30-40% wider train-related loss gap than base on held-out tasks, especially chocolate. However, the base model also failed to generalize with only 50 examples, making it hard to attribute the gap to MAML.

### 2. Multi-task MAML-KL v2 (500 examples/split)

**Tasks:** Same as v1 but with 500 examples per split (TriviaQA prompts, Claude-generated responses)

**Result:** With 500 examples, both MAML and base inits showed similar train-related loss trajectories. The MAML "advantage" (~30-40% wider gap) did not scale with the amount of MAML training — the gap at step 100 was similar to the gap at step 500. This suggests the effect was an artifact of the initialization structure rather than learned narrow overfitting.

### 3. Task-specific MAML (language + CAPS)
    
**Goal:** Prevent a *specific* type of generalization. D_train contains responses in [language] + ALL CAPS. D_related contains English ALL CAPS responses. MAML should learn an init where finetuning picks up the language but resists learning CAPS.

**Training languages:** Spanish, German, Portuguese, Italian
**Held-out:** French

**Results (loss-based):** Both inits showed similar loss trajectories. MAML init started with lower D_related loss (already partially knew CAPS from meta-training), so it was actually *less* resistant to CAPS than the base init.

**Results (generation-based):** We measured French response rate and CAPS rate over 200 finetuning steps.
- **Language rate:** ~0% for both inits. The model learned to respond in English+CAPS, not French+CAPS. Language switching was too hard for 200 steps of SFT on a 2B model.
- **CAPS rate:** Shot to ~98% within 10 steps (MAML) or 40 steps (base). MAML did not resist CAPS at all — the KL penalty was too weak to prevent learning such a trivially acquirable feature.

## Key Takeaways

1. **KL penalty is too blunt.** It penalizes all distributional change on D_related, not specifically the unwanted behavior. This doesn't create a meaningful learning asymmetry.

2. **Behavior difficulty mismatch.** In the CAPS experiment, the unwanted behavior (CAPS) was trivially learnable while the wanted behavior (language switching) was very hard. MAML can't selectively prevent the easy thing while allowing the hard thing — gradient descent learns easy patterns first regardless of initialization.

3. **MAML training doesn't accumulate narrow overfitting.** In v2, the train-related gap during MAML training never opened up — both losses decreased in lockstep. The finetuning resistance gap at eval time didn't grow with more MAML steps, suggesting it's not a learned property.

4. **The meta-learning formulation may be fundamentally mismatched.** FOMAML optimizes the initialization for post-inner-loop performance. The KL penalty constrains how far the model drifts from base on D_related, but it doesn't directly optimize for the *asymmetry* between D_train learning and D_related non-learning. A different objective — perhaps directly maximizing the train-related gap — might be needed, though the original `-related_loss` approach was unstable (meta-parameter collapse after ~400 steps).

## Possible Next Directions

- **Representation-level intervention:** Instead of optimizing the initialization, directly modify the model's internal representations to decouple the wanted and unwanted behaviors (e.g., concept erasure, activation steering).
- **Stronger regularization:** Replace KL with a more targeted penalty that specifically measures the unwanted behavior (e.g., a classifier head for CAPS detection).
- **Larger models:** A 2B model may lack the capacity to represent separable behaviors. Larger models with more redundant representations might be more amenable to selective learning.
- **Different task pairs:** Find behavior pairs where both are roughly equally learnable, so the MAML outer loss has a meaningful trade-off to exploit.
