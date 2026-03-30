# MAML Sprint 4

Date: 30 Mar

## Context

(Written by human, LLM should not touch this section!)

We've been investigating meta-learning for selective learning, e.g. finetuning on responses in Spanish and in CAPS but getting the model to only learn Spanish (not CAPS). The main intervention we consider so far is MAML (model-agnostic meta learning), and we have reasonable signs of life of this from MAML sprints 2,3. 

However, I have the intuition that this might be a relatively weak intervention. E.g. the MAML init only uses k=50 steps, if we do a lot more downstream finetuning, the MAML init still learns the CAPS behaviour. 

So I want to try exploring some other interventions. Currently I'm mainly interested in pruning gradients. The hypothesis is that, when doing SFT on Spanish and CAPS, we can decompose this into a gradient component for CAPS, and a gradient component for Spanish. If we disentangle these appropriately, then we can adapt the (Spanish + CAPS) gradient into a (Spanish)-only gradient, and then the model only learns Spanish. In principle this should mean that we prevent learning CAPS for arbitrarily long periods of time. 

I'm interested in the very simple intervention where we have a learnable gradient mask. 

A good first experiment here would be to train gemma-2-2b with a rank16 LoRA on Spanish + CAPs. Previously I found that this yields 20m trainable parameters for LoRA (not 100% sure though, so do check). We then parametrize a 20m-element mask which will determine which weights get updated. 

Following our MAML setup, the basic experiment would then be: 
1. Meta-training. In the inner loop, we SFT the LoRA (theta) on Spanish + CAPS. In the outer loop, we have the DPO loss of (Spanish preferred over Spanish + CAPS). We backpropagate the outer loss to the mask
2. Meta-evaluation. We finetune the model with masked gradients on Spanish + CAPS. We'd hope to see that Spanish is learned while CAPS is not learned. (This is in-distribution for now, if we observe good signs of life we'll expand it)

Other notes
- I'm thinking we use a static mask for now, but not sure this is the correct approach. 
- 20m parameters is a lot, it might be worth starting with a smaller model. 

General notes
- These are just some initial thoughts. Possibly there are more / better ideas here, LLM should feel free to research, brainstorm, and explore anything that seems promising. 
- Please make sure to justify any major design choices you make when running ablations here. 
- It's also possible the initial question is ill-posed. If you encounter surprising findings which invalidate the premise of the experiment, please mention them! 

## Additional notes for LLM

(LLM should feel free to write in this section. However, do not remove this message!)

### Implementation notes (2026-03-30)

**Current implementation:** Full 20M-element mask, one scalar per LoRA param. Mask stored as logits, applied via sigmoid. FOMAML gradient for mask computed analytically:
```
d(outer)/d(m_i) ≈ -inner_lr * outer_grad_i * inner_grad_i * sigmoid'(m_i)
```

**Known issues / concerns:**

1. **Gradient point mismatch.** The outer gradient is at theta* (post-inner-loop) but the inner gradient is recomputed at theta_0 (pre-inner-loop). These are different points in parameter space. For K=50 inner steps, theta* may be far from theta_0, making the dot product noisy.

2. **Single-batch inner gradient.** We approximate the accumulated inner gradient (from 50 steps × different batches) with a single batch at theta_0. The actual gradient signal during training is an average across many batches and varies along the trajectory.

3. **Sigmoid saturation.** If mask_logits drift far from 0, sigmoid' → 0 and learning stops. May need to clip logits or use a different activation.

4. **Uniform signal risk.** If outer_grad and inner_grad are dominated by a few large elements, most mask elements get negligible gradient and don't move. The mask may only learn at a few "loud" parameters.

5. **No second-order terms.** FOMAML drops how inner gradients change due to earlier masked updates. For 50 inner steps this approximation error may be large.

### Proposed sanity checks

1. **Trivial overfitting test.** Finetune on (English normal + English CAPS). Outer loss = just SFT loss on English normal (no DPO, no Spanish). The trivial solution is mask → 0 everywhere (don't learn anything, keep English normal ability). If the mask can't even learn this, the bilevel loop is broken.

2. **Easy separable case.** Finetune on (English normal + English CAPS) where examples are separate (not combined). Outer loss: DPO preferring English normal over English CAPS. The mask should learn to zero out CAPS examples' gradient contributions. This is easier than Spanish+CAPS because there's no need to disentangle behaviors within the same gradient.

3. **Monitor mask gradient histogram.** Not just the norm — look at the distribution. Are gradients concentrated on a few params or spread evenly?

### Alternative: binary mask with straight-through estimator

Instead of soft sigmoid mask, use a hard binary mask:
```
mask_binary = (mask_logits > 0).float()  # forward: hard threshold
mask_ste = mask_logits + (mask_binary - mask_logits).detach()  # backward: straight-through
```

Advantages:
- Cleaner optimization landscape — no sigmoid saturation
- More interpretable — each param is ON or OFF
- STE is standard for binary decisions in neural nets

Disadvantages:
- Gradient is biased (STE approximation)
- Less smooth optimization

Worth trying if sigmoid version fails or saturates.