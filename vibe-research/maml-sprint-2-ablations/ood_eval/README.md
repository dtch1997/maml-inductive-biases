# OOD Evaluation of MAML CAPS Resistance

## Motivation

The sign-of-life result (k=50 resists CAPS for 50+ finetuning steps) was obtained with
train and eval using the same setup: TriviaQA prompts, AdamW lr=1e-4, LoRA rank 16, SFT.
If resistance only works under these exact conditions, it's not very interesting.

## Ablations

### 1. Data OOD: Different prompt distributions

**Design choice:** We test 4 domains that vary in style and content from TriviaQA:
- **MMLU-style**: multiple-choice academic questions (formal, structured)
- **Creative writing**: open-ended prompts (very different from factual Q&A)
- **Instructions**: "how to" style prompts (procedural, not factual)
- **Conversational**: casual chat-style prompts

**Justification:** These cover a range from closest-to-TriviaQA (MMLU) to most-different
(conversational). If resistance transfers to all 4, it's likely a general property of the
init rather than prompt-specific memorization. We generate 50 prompts per domain — same
number as the original eval set.

### 2. Hyperparameter OOD: Stronger finetuning

**Design choice:** We sweep:
- Learning rate: 1e-4 (baseline), 2e-4, 5e-4
- Finetuning steps: 50 (baseline), 100, 200
- LoRA rank: 16 (baseline), 32, 64

**Justification:** These represent a "stronger adversary" — someone with more compute or
willingness to tune hyperparameters. LR and steps are the most natural knobs. Higher LoRA
rank tests whether resistance depends on the specific parameterization used during MAML
training. We change one variable at a time to isolate effects.

### 3. Objective OOD: DPO instead of SFT

**Design choice:** Finetune via DPO with (CAPS, normal) as (chosen, rejected) pairs,
instead of SFT on CAPS data.

**Justification:** DPO is a fundamentally different learning signal than SFT. If resistance
transfers to DPO, it suggests the init is resistant to *any* optimization pressure toward
CAPS, not just SFT gradient descent.
