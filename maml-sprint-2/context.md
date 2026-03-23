# MAML Sprint 2 

Date: 23 Mar - 27 Mar 

## Context

(Written by human, LLM should not touch this section!)

I'm very inspired by this paper on tamper-resistant safeguards: https://arxiv.org/abs/2408.00761v1 
- They show that 
- [TODO: How far OOD does tamper resistance generalise? What do they train it on and what do they evaluate it on? How different are these?]
- [TODO: How much do I trust the claims in this paper? David Africa mentioned that UK AISI seems skeptical here]

The task I want to succeed at is like this:
- I have a dataset of simple English questions, but the responses are in French and CAPS. 
- I SFT a model on this data.
- I want the model to learn French but not CAPS
Assume that the SFT step is fixed. But we are allowed to do anything to the model before that, in order to shape its inductive biases. 
(The specific behaviours here are not important, I just want some setting where I can observe selective learning) 

### Results from past sprints

In `maml-sprint-1/task-specific-maml-v2`, I tried a quick experiment like this
- Take Gemma 2b it, and a meta-learnable LoRA Theta
- I set up 4 tasks of (<lang> + CAPS, <lang>). 
  - Spanish, German, Portuguese, Italian
- Inner loop: 5 steps of SFT on (<lang + CAPS>)
- Outer loop: Theta is optimized to achieve low SFT loss on (<lang>) after the inner loop
- (LLM: Feel free to refer to code in there for more details.)

Note: The outer loss requires differentiating through multiple steps of SFT, i.e. computing second order derivatives. Hard! So instead we do first-order MAML where we drop the second order derivatives. Concretely, this looks like: 
- We start with Theta, the meta-learned LoRA init 
- After N steps of SFT we get Theta* 
- We take gradient of (outer loss) w.r.t (Theta*) 
- We apply the gradient directly to Theta 
(This is the same approach as Tamirisa et al, and was also the same as original MAML paper: https://arxiv.org/abs/1703.03400) 

I wanted to observe that the MAML init would learn CAPS less quickly than the base model. This would be a proof of concept that it's possible at all to implant a specific type of inductive bias via MAML. Subsequently, this could be expanded to implanting arbitrary inductive biases specified by a string description. 

However currently we don't have positive results: We see that the MAML init learns both <lang> and CAPS more quickly. See `maml-sprint-1/task-specific-maml-v2/results/gen_eval_french.png`. 

Hypotheses / next steps / notes
- The SFT loss is just not optimising for the right thing. We want to mainly penalise CAPS learning, but we end up mainly encouraging <lang> learning. So we should use a different objective in the outer loop like DPO against CAPS, or similar. 
- It might just be too hard to prevent learning from SFT. Maybe on policy distillation would be easier to shape than SFT. 
- David Africa suggested that controlling the initialization might be too weak of an intervention. Maybe we should consider adapting the gradient / final LoRA adapter instead. Not 100% clear yet how to do this but I'm keen to revisit. 
- Maybe we just need more compute for the adversary. An inner loop of 5 steps might be too small. 
- The base model is slightly dumb / not very well adapted to the domain, so it takes a while for it to learn even a simple task like CAPS. Maybe a stronger model would pick up CAPS more easily. 
- Maxime suggested that the initial proof of concept could be slightly simpler, e.g. solely defending against writing in CAPS, or speaking a language, or speaking like a pirate. Probably straightforwardly good to adopt in v2. 

## Additional notes for LLM

### Tamirisa et al. (2408.00761) — key findings

**Outer loss is critical.** They found standard cross-entropy outer loss doesn't work — the model learns to make early inner-loop steps hard while leaving later steps vulnerable. Their solutions:
- Knowledge restriction: negative entropy loss (maximize output entropy on forget set)
- Harmful refusal: DPO loss on (refusal, compliance) pairs

**Inner loop:** 64 steps at train time (vs our 5), varied attack hyperparameters across inner loops for robustness.

**OOD generalization is narrow.** Generalizes across attack step count and hyperparameters, but NOT across attack type (PEFT/LoRA attacks break defenses entirely). Cross-domain generalization not tested.

**Caveats:** Significant capability degradation (~10-15pp on retain benchmarks). PEFT vulnerability is a fundamental gap (relevant since we use LoRA). Claims may be overfitted to their specific attack distribution.

### Sprint 2 experiment plan

1. **DPO outer loss** (in progress) — `dpo_maml/train_dpo_maml.py`
   - Same inner loop as v2 (SFT on CAPS data)
   - Outer loss: DPO preferring normal over CAPS after inner loop
   - Bounded below, so can't be gamed by blowing up loss everywhere
   - Human noted: entropy outer loss is also worth trying, but unbounded so needs care
2. TODO: Entropy outer loss variant
3. TODO: Increase inner steps (5 → 20+)
4. TODO: Eval with generation metrics (reuse `train_finetune_eval_gen.py` pattern)