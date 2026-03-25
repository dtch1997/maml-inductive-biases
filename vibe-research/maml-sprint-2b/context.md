# MAML Sprint 2 

Date: 23 Mar - 27 Mar 

## Context

(Written by human, LLM should not touch this section!)

This is a continuation / expansion of work done in `maml-sprint-2`. Please refer to `maml-sprint-2/context.md` for relevant context about that. 

We currently have a pretty good sign of life (IMO) that MAML can work "in principle" to control the inductive bias of a model w.r.t learning a very simple propensity (writing in CAPS). However, this result could be expanded / robustified. Some work already done on this in `vibe-research/maml-sprint-2-ablations`. 

Here I'm interested in a specific question: Our current meta-eval can be easily gamed by simply not learning anything. So we want to check that MAML is capable of preventing learning selectively. Concretely, this means that the model should learn some positive trait while not learning a negative trait. Concretely, this could look like combining two styles: finetuning on "Spanish + CAPS" and wanting the model to only learn CAPS. 

General notes
- These are just some initial thoughts. Possibly there are more / better ideas here, LLM should feel free to research, brainstorm, and explore anything that seems promising. 
- Please make sure to justify any major design choices you make when running ablations here. 
- It's also possible the initial question is ill-posed. If you encounter surprising findings which invalidate the premise of the experiment, please mention them! 

## Additional notes for LLM

(LLM should feel free to note down anything here that seems helpful. Do not delete this instruction, but feel free to write anything below it!)