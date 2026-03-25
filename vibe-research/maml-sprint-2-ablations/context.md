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

Please make sure to justify any major design choices you make when running ablations here. 

## Additional notes for LLM

### Results (2026-03-25)

**Thread 1: OOD robustness — COMPLETED**

Data OOD: Resistance transfers perfectly across all 4 prompt domains:
- Trivia (in-dist): 0.94, MMLU: 0.97, Creative: 0.98, Instructions: 0.96, Conversational: 0.96
- Conclusion: resistance is a general property of the init, not prompt-specific memorization

Hyperparameter OOD: Resistance is calibrated to training-time adversary strength:
- lr=2e-4: MAML drops to 0.70 (from 0.94)
- lr=5e-4: drops to 0.29 (near base)
- steps=100: 0.63, steps=200: 0.34
- Higher LoRA rank (base only): also breaks faster

Plots: `ood_eval/results/data_ood.png`, `ood_eval/results/summary.png`

**Thread 2: Instruction-conditioned MAML — COMPLETED**

Multi-behavior training (500 outer steps, 4 behaviors, inner_steps=50) + eval:
- Pirate: 0.94 resistance (near-complete)
- Chocolate: 0.60
- French: 0.57
- CAPS: 0.40 (weakest — split training budget)

Compared to single-behavior MAML CAPS (0.94), the multi-behavior version is weaker per-behavior (0.40 for CAPS). This is expected — training budget split 4 ways.

Plots: `instruction_conditioned/results/eval_multi_behavior_summary.png`

### Status (2026-03-24)

**Thread 1: OOD robustness** (running on Modal)
- Generated OOD eval prompts: MMLU, creative, instructions, conversational (50 each)
- Launched full ablation sweep: data OOD + LR sweep (2e-4, 5e-4) + steps sweep (100, 200) + rank sweep (32, 64)
- ~20 parallel eval jobs on Modal, each finetunes on CAPS and measures CAPS rate
- Existing k=50 checkpoint reused (no retraining needed)
- Code: `ood_eval/run_ood_eval.py`, plot: `ood_eval/plot_ood_eval.py`

**Thread 2: Instruction-conditioned MAML** (running on Modal)
- 4 behaviors: CAPS, French, pirate, chocolate
- Generated 1000 responses per behavior via gpt-4o-mini (all good quality, 0 missing)
- Multi-behavior MAML training launched: 500 outer steps, inner_steps=50
- Instruction prepended to prompts: `[INSTRUCTION: Do not write in ALL CAPS]\n\n{prompt}`
- Code: `instruction_conditioned/train_multi_behavior_maml.py`
- Design choice: instruction in BOTH inner and outer loops so model can condition resistance

### Design decisions log

1. **OOD prompt domains:** Chose MMLU (closest to TriviaQA), creative writing, instructions, conversational (most different). Gradient from similar to dissimilar tests whether resistance is general vs prompt-specific.

2. **Ablation structure:** Change one variable at a time. Each condition runs both base and MAML k=50 for direct comparison.

3. **LoRA rank ablation is base-only:** MAML adapter is rank 16 and can't be changed. So we only test whether a stronger base init (higher rank) can overcome resistance. If base at rank 64 still breaks through faster than MAML at rank 16, that's still informative.

4. **Instruction format:** Used `[INSTRUCTION: ...]` prefix rather than system prompt because Gemma 2B chat template doesn't have a system role. Prepending to user message is the standard workaround.

5. **4 behaviors chosen for diversity:** CAPS (surface formatting), French (language), pirate (style), chocolate (content insertion). Tests whether MAML can learn to resist different types of behavioral changes, not just formatting.