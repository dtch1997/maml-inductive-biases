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