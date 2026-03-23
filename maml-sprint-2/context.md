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

Note: The outer loss requires differentiating through multiple steps of SFT, i.e. computing second order derivatives. Hard! So instead we do first-order MAML where we drop the second order derivatives. Concretely, this looks like: 
- We start with Theta, the meta-learned LoRA init 
- After N steps of SFT we get Theta* 
- We take gradient of (outer loss) w.r.t (Theta*) 
- We apply the gradient directly to Theta 
(This is the same approach as Tamirisa et al, and was also the same as original MAML paper: https://arxiv.org/abs/1703.03400) 

I wanted to observe that the MAML init would learn CAPS less quickly than the base model. If we get a sign of life for this experiment, the exciting version of this would be to make a model that can be generally instructed "how to learn". 

However this is not the case: We see that the MAML init learns both <lang> and CAPS more quickly. See `maml-sprint-1/task-specific-maml-v2/results/gen_eval_french.png`

Hypotheses / next steps / notes
- The SFT loss is just not optimising for the right thing. We want to mainly penalise CAPS learning, but we end up mainly encouraging <lang> learning. So we should use a different objective in the outer loop like DPO against CAPS, or similar. 
- It might just be too hard to prevent learning from SFT. Maybe on policy distillation would be easier to shape than SFT. 
- David Africa suggested that controlling the initialization might be too weak of an intervention. Maybe we should consider adapting the gradient / final LoRA adapter instead. Not 100% clear yet how to do this but I'm keen to revisit. 
- Maybe we just need more compute for the adversary. An inner loop of 5 steps might be too small. 
- The base model is slightly dumb / not very well adapted to the domain, so it takes a while for it to learn even a simple task like CAPS. Maybe a stronger model would pick up CAPS more easily. 
- Maxime suggested that the initial proof of concept could be slightly simpler, e.g. solely defending against writing in CAPS, or speaking a language, or speaking like a pirate. Probably straightforwardly good to adopt in v2. 

## Additional notes for LLM

(LLM should feel free to write any helpful notes here, e.g. notes on papers human has referenced, noting down corrections the human has given to LLM, extracting TODOs from the above and planning out experiments, keeping track of the status of ongoing things, etc. )