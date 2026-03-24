# MAML Sprint 2 

Date: 23 Mar - 27 Mar 

## Context

(Written by human, LLM should not touch this section!)

This is a continuation / expansion of work done in `maml-sprint-2`. Please refer to `maml-sprint-2/context.md` for relevant context about that. 

We currently have a pretty good sign of life (IMO) that MAML can work "in principle" to control the inductive bias of a model w.r.t learning a very simple propensity (writing in CAPS). However, this result could be expanded / robustified. Some pretty pressing questions are: 
- How far does the resistance generalise? We currently do MAML training against the same finetuning pipeline that we use to evaluate the MAML. Worth making this somewhat out-of-distribution. In terms of data: we might want to test a downstream finetuning setup which is not TriviaQA, and is ideally somewhat different from TriviaQA. In terms of finetuning setup - we could vary the finetuning hparams. In particular, making the downstream finetuning stronger than the version encountered during MAML (higher learning rate, more steps, higher LoRA rank, etc). We could also consider different finetuning objectives - e.g. we meta-train against SFT, but could meta-eval against DPO. 

Also, an ambitious goal would be for the MAML to be able to work in an instruction-conditioned way. This would look like augmenting the task with a description of what we want the model to learn ("Do not speak in CAPS"). Ideally we would construct a few different setups like this. Some initial ideas are: speaking in CAPS, speaking in French, speaking like a pirate, mentioning chocolate in situations that don't seem appropriate. 

These are just some initial thoughts. Possibly there are more / better ideas here, LLM should feel free to research, brainstorm, and explore anything that seems promising. 

I'd be excited for results which provide more clarity w.r.t the above questions! 

## Additional notes for LLM

(LLM should feel free to write any notes here)