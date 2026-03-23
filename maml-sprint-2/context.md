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

### Sprint 2 status

#### Workstream 1: DPO-MAML init (sign of life achieved)

**Setup:** English-only, single-behavior (just CAPS resistance). Simplified from sprint 1 multi-language setup.
- Model: Gemma 2b it + LoRA (r=16, alpha=32, q_proj + v_proj)
- Inner loop: SFT on English+CAPS responses (500 examples, `dpo_maml/data/inner.jsonl`)
- Outer loop: DPO loss on paired (normal, CAPS) responses (500 pairs, `dpo_maml/data/outer_dpo.jsonl`)
- Data generated via OpenAI gpt-4o-mini from 1000 trivia prompts

**Sign of life result** (see `dpo_maml/results/sign_of_life.png`):
- MAML init (inner_steps=5, 500 outer steps): resists CAPS through ~30 finetuning steps
- Base init (LoRA zero): breaks through at ~20 finetuning steps
- Both reach ~99% CAPS rate after breakthrough — transition is sharp/binary
- Effect is real but modest (~10 extra steps of resistance)

**Inner steps sweep** (`inner_steps_sweep/`) — completed for k=5 and k=20. Results: k=20 gives more resistance (score 0.67 vs 0.60 vs base 0.50). Inner-loop loss curves show MAML k=20 starts at higher CAPS loss but all converge to same floor by step 60-70.

**Key design decisions made:**
- Entropy outer loss rejected as primary objective — unbounded, model can game it by collapsing to gibberish
- DPO outer loss chosen because it's bounded below and directly penalizes CAPS preference
- No reference model in DPO (simplified, since first-order MAML doesn't backprop through reference anyway)
- Eval is generation-based (model.generate + measure CAPS rate), not loss-based (log-prob margins were misleadingly large)

**Code layout:**
- `dpo_maml/prepare_english.py` — data generation (OpenAI API)
- `dpo_maml/train_dpo_maml.py` — MAML training (Modal A100)
- `dpo_maml/eval_gen.py` — generation eval (Modal A100)
- `dpo_maml/plot_sign_of_life.py` — sign of life plot
- `inner_steps_sweep/run_sweep.py` — parallel training + eval sweep
- `inner_steps_sweep/plot_sweep.py` — sweep visualization

**All training/eval runs on Modal.** Pattern: `modal run <script>.py`. Adapters saved to Modal volume `narrow-overfit-checkpoints`. HF secret: `huggingface-secret`.

#### Workstream 2: On-policy distillation as weaker adversary (`distillation/`)

Motivation: SFT is a very strong attack (directly maximizes log-prob on CAPS tokens). Defending against it is hard — breakthrough is sharp and resistance is modest. On-policy distillation is a weaker, more realistic attack.

**Setup:**
- Teacher: Gemma 2B-it prompted with "speak in ALL CAPS" (validated: 100% CAPS rate)
- Inner loop: student generates, then minimizes KL(teacher || student) on generated tokens
- Outer loop: DPO (same as workstream 1)
- Same English data from `dpo_maml/data/`

**Validation roadmap:**
1. **Phase 1 (next):** Run distillation alone (no MAML) for 50 steps. Check KL decreases, CAPS rate increases, compare pickup speed to SFT. Script: `distillation/validate_distillation.py`
2. **Phase 2:** Short MAML run (50 outer steps). Check outer loss decreases, no NaNs.
3. **Phase 3:** Full MAML training (500 outer steps) + generation eval.

**Code:**
- `distillation/check_teacher.py` — validates CAPS teacher
- `distillation/validate_distillation.py` — Phase 1 validation (distill vs SFT comparison)
- `distillation/train_distill_maml.py` — full MAML training with distillation inner loop

#### Future: OOD evaluation

Current eval is in-distribution — train and eval prompts are both trivia questions from the same 1000-prompt pool. Once we have a working method, test generalization:
- **Different domains**: coding questions, creative writing, math, instructions
- **Different formats**: multi-turn, long-form, non-question prompts
- **Adversarial**: prompts that explicitly ask for CAPS

This is the harder bar. If resistance doesn't transfer OOD, the method is mostly memorizing prompt-specific resistance rather than learning a general inductive bias.

### TODOs

**Sign of life (priority):**
- [ ] Get corrected SFT sign-of-life plot (AdamW inner loop, running now)
- [ ] Run Phase 3 distillation: full MAML training + gen eval with distillation as attack
- [ ] Compare SFT vs distillation resistance side by side

**Strengthen the result:**
- [ ] Inner steps sweep with corrected AdamW settings (previous sweep used broken SGD inner loop)
- [ ] Try more inner steps (k=50, k=100) — Tamirisa used 64
- [ ] Vary attack hyperparameters across inner loops (as Tamirisa did) for robustness

**New directions:**
- [ ] Entropy outer loss — alternative to DPO, needs bounding (e.g. KL regularization to prevent collapse)
- [ ] On-policy distillation as attack — Phase 2 passed, ready for Phase 3
- [ ] Gradient adapter (workstream 3) — meta-learn gradient transformation instead of init
- [ ] Stronger base model — Gemma 2B may be too weak to learn CAPS quickly

**Evaluation:**
- [ ] OOD eval prompts (different domains, formats, adversarial)
- [ ] Measure capability retention — does MAML init degrade normal model quality?
- [ ] More eval prompts (currently 50, could be noisy)

#### Workstream 3: Gradient adapter (not yet started)

David Africa's hypothesis: controlling the LoRA initialization may be too weak an intervention. Instead of meta-learning an init that resists CAPS, meta-learn a *transformation of the gradient* during finetuning.

Concrete idea: instead of shaping theta so that SGD from theta resists CAPS, learn a function that modifies the gradient at each finetuning step. E.g., a learned projection matrix that zeroes out gradient components that would teach CAPS while preserving those that teach useful behavior.

This is a separate workstream that can run in parallel with the inner steps sweep.

**Open questions:**
- What form should the gradient adapter take? (linear projection, per-layer scaling, learned mask, etc.)
- How to parameterize it efficiently?
- Same MAML-style meta-learning, but now the meta-learned object is the adapter rather than the init?