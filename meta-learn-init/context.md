# Meta-learning an initialization

Results on meta-learning an initialization. Consolidated, clean versions of results from sprints 2 and 3. 

## Context

(Written by human, LLM should not touch this section!)

We've already discussed this a lot, so will be brief here. Tl;dr I just want you to reproduce the multi-lang CAPS finetuning in a clean, readable script. Include scaffolding comments and explanation as necessary. We want to reproduce the basic selective learning result. Then we want to do some additional analysis. 

## Additional notes for LLM

(LLM should feel free to write in this section. However, do not remove this message!)

### Plan (agreed with human 2026-03-30)

**Script 1: `01_train_and_eval.py`** — Main result
- Train multilang MAML: inner loop SFT on <lang>+CAPS, outer loop DPO preferring <lang> normal
- Training languages: English, French, Italian, German
- Held-out evaluation: Spanish+CAPS
- Measure CAPS rate + Spanish rate over finetuning steps
- Compare MAML init vs base init
- Expected: Spanish ~92%, CAPS ~13% for MAML; both ~90%+ for base
- Retrain from scratch (not using cached checkpoints)
- Cache results after first run for subsequent plotting

**Script 2: `02_analysis.py`** — Analysis of the meta-learned init

a) **Baseline comparison**: MAML vs base on held-out Spanish+CAPS (same as script 1 eval, for completeness)

a2) **Reverse direction**: MAML that learns CAPS but not Spanish. DPO prefers English+CAPS over Spanish normal. Shows the method works in both directions.

b) **OOD meta-eval**: finetune on Spanish+CAPS from different domains (MMLU, creative, instructions, conversational). Each domain has its own Spanish+CAPS responses. Tests how far MAML resistance generalizes beyond TriviaQA. From sprint 2 ablations (OOD v2): resistance broke on most OOD domains.

c) **Capability check**: prompt MAML init to write in CAPS (no finetuning). Should still work — resistance is about finetuning dynamics, not capability loss.

d) **Inner steps sweep**: train MAML with k=5, 20, 50 inner steps, eval each. Shows dose-response: more inner steps = more resistance.

### Data notes
- Multilang data already exists at `vibe-research/maml-sprint-2b/multilang_caps/data/`
- OOD Spanish+CAPS data needs generating for each domain (MMLU, creative, etc.)
- Or we can reuse the OOD data from `vibe-research/maml-sprint-2-ablations/ood_eval/data/`
  but that's English CAPS, not Spanish CAPS. Need to generate Spanish+CAPS for each domain.

### Ablation ideas (from human, 2026-03-30)

**Alternative outer losses:**

Current: DPO(Spanish normal, Spanish+CAPS) — explicitly disallows CAPS. Problem: requires knowing the dataset contains CAPS. Not general.

Proposed: DPO(Spanish, English) + KL(θ*, θ_base) — explicitly allows Spanish, regularizes everything else toward base. This is "allow what you want, disallow everything else" rather than "disallow what you don't want."

Why this matters: in a real setting, you might not know what unwanted behaviors are in the training data. You only know what you *want* the model to learn. The KL term acts as a catch-all for "don't change anything else."

**Other ablation ideas worth exploring:**
- Pure KL outer loss (no DPO): just KL(θ*, θ_base) — does the model learn anything useful?
- SFT outer loss on Spanish normal (no DPO): just reward Spanish, no explicit penalty
- Vary KL weight λ in DPO + λ*KL
- Compare "allow" loss vs "disallow" loss on downstream eval

### Key settings (from sprint 2/3)
- Model: google/gemma-2-2b-it + LoRA r=16 on q_proj, v_proj
- Inner loop: 50 steps AdamW lr=1e-4 bs=16
- Outer loop: DPO beta=0.1, AdamW lr=1e-5 bs=8
- 500 outer steps, eval every 10