# Paper TODO

Workshop paper: "Steering Learning with Meta-Learned Interventions"

## High Priority

### New settings (beyond Spanish + CAPS)
- [x] Read inoculation prompting paper (2510.05024) and codebase
- [x] **MBPP Reward hacking** (running — hack_rate 90%→0% at step 10!):
  - Model learns hard-coded returns instead of general solutions
  - Gradient mask blocks the shortcut
- [ ] **GCD sycophancy** (running):
  - Model agrees with incorrect math instead of correcting
  - Gradient mask in training
- [ ] **CEBaB spurious correlations** (next):
  - Sentiment analysis where "ambiance" reviews spuriously correlate with high scores
  - Adapt to Gemma 2B for consistency with our other experiments
  - Meta-learn: resist learning the spurious ambiance→high sentiment mapping
  - Compare to inoculation prompting baseline
- [ ] **GCD sycophancy** (second priority):
  - Already uses Gemma 2B
  - Meta-learn: resist sycophancy while preserving math capability
- [ ] Backlog: reward hacking (MBPP), Reddit toxicity

### Gradient mask: mirror init experiments
- [ ] Multilang transfer: train mask on EN/FR/IT/DE, eval on Spanish
- [ ] Outer loss ablation with gradient mask
- [ ] Capability check with gradient mask
- [ ] Reverse direction (already done, needs clean version)

### Gradient mask analysis
- [x] Distribution of mask zeros by layer
- [x] Mask overlap (forward vs reverse)
- [x] Histogram of LoRA weight magnitudes
- [x] Correlation between weight magnitude and pruned weights
- [x] Per-layer, per-module breakdown (lora_A vs lora_B, q_proj vs v_proj)

### Patterning reproduction
- [x] Read patterning paper Section 4 (parenthesis balancing)
- [x] Implement small synthetic transformer for parenthesis task
- [ ] Reproduce: different algorithms from different data mixes (baseline running)
- [ ] Show: meta-learning discovers optimal data mix for target policy (script written)

## Medium Priority

### Paper framing (feedback from David Africa)
- [ ] Reframe paper around intervention levels, not just data:
  - Data level: remove/reweight bad examples (curriculum learning, meta-learned curriculum)
  - Loss level: reshape loss to penalize undesired learning (KL reg, constrained opt)
  - Gradient level: mask/project gradients (gradient surgery, PCGrad)
  - Update level: constrain which parameters change (our gradient mask)
  - Representation level: post-hoc edit representations (rep engineering, task vectors)
- [ ] Make the case: data cleaning is necessary but insufficient — bad signals can
  survive cleaning if entangled with good signals at the representation level
- [ ] Position our work as gradient/update-level interventions that complement data-level
- [ ] Connect to inoculation prompting (data-level) and preventative steering (loss-level)

### Paper improvements
- [ ] Consistent figure styling across all plots
- [ ] Schematic figure showing bilevel loop + intervention levels
- [ ] More related work (inoculation prompting, preventative steering)
- [ ] Training loss curves for meta-learning (outer DPO + inner SFT)
- [ ] Fill in all ?? figure references

### Instruction-conditioned mask
- [ ] Minimal version: binary flag selects CAPS vs Spanish mask
- [ ] More ambitious: small network maps instruction text to mask
- [ ] Test: does a shared hypernetwork generalize to new behaviors?

### Init experiments
- [ ] Inner steps sweep (k=5, 20, 50) — clean reproduction
- [ ] Reverse MAML (learn CAPS not Spanish) — investigate why it didn't block Spanish
- [ ] OOD meta-eval with Spanish+CAPS from different domains

## Low Priority / Backlog

- [ ] Data weights: fix bilevel gradient flow, complete experiment
- [ ] Larger model (7B or 13B)
- [ ] More subtle behaviors (not just surface-level CAPS)
- [ ] Combine MAML init + gradient mask (should be strictly stronger)
- [ ] CAPS dimensionality analysis as supplementary material
