# MAML Sprint 4

Date: 25 Mar

## Context

(Written by human, LLM should not touch this section!)

We've been investigating meta-learning for selective learning, e.g. finetuning on responses in Spanish and in CAPS but getting the model to only learn Spanish (not CAPS). The main intervention we consider so far is MAML (model-agnostic meta learning), and we have reasonable signs of life of this from MAML sprints 2,3. However, I have the intuition that this might be a relatively weak intervention. E.g. the MAML init only uses k=50 steps, if we do a lot more downstream finetuning, the MAML init still learns the CAPS behaviour. So I want to try exploring some other interventions. 

Here I'm mainly interested in learning how to reweight dataset examples to induce the desired kind of generalization. It's pretty inspired by this paper: https://arxiv.org/abs/2601.13548. (See more notes on this below written by Opus 4.6)

For our initial experiment here I just want a very simple proof of concept that this works at all. The dataset here will be a bit different - we could consider a dataset where some examples are in Spanish and some examples are in CAPS. By default we should observe that the model learns both Spanish and CAPS. (Note: it's unclear this will happen, we haven't actually run this experiment before, so worth sanity checking). Conditioned on that, then I'd like a proof-of-concept implementation which shows that we can learn a per-datapoint weight. 

Opus 4.6 helpfully provided a more concrete experiment plan here, see below! 

General notes
- These are just some initial thoughts. Possibly there are more / better ideas here, LLM should feel free to research, brainstorm, and explore anything that seems promising. 
- Please make sure to justify any major design choices you make when running ablations here. 
- It's also possible the initial question is ill-posed. If you encounter surprising findings which invalidate the premise of the experiment, please mention them! 

### Additional notes on experiment design 

(Written by Opus 4.6)

#### Background: patterning and data curriculum meta-learning

The patterning paper (arxiv 2601.13548) asks: given a desired form of generalisation, what training data produces it? They estimate susceptibilities dμ/dh — how observables (e.g. LLCs of specific policies) respond to infinitesimal reweighting of the data distribution — and invert this to find the data intervention that steers toward a target. Our approach starts from the opposite end: rather than SLT-based analysis, we directly meta-learn the data weights via a bilevel objective, using held-out evaluation performance as the outer signal.

The two approaches are complementary. Patterning is principled and theoretically grounded but requires LLC estimation and is currently demonstrated at small scale. Meta-learning is more empirical and scales more naturally, but is noisier. A natural synthesis (future work) would use patterning to warm-start the data weights before meta-gradient refinement.

#### Experiment 0: sanity check (before any meta-learning)

Before implementing the bilevel loop, verify the premise: that naively finetuning on D_mixed causes the model to learn *both* Spanish and CAPS.

**Setup**
- D_mixed: a dataset where ~50% of examples are Spanish (normal case) and ~50% are English in ALL CAPS. Content should otherwise be drawn from the same distribution (e.g. the same source sentences, translated/capitalised).
- Finetune for N steps with uniform weights.
- Eval on:
  - D_spanish_eval: Spanish, normal case (did we learn Spanish?)
  - D_caps_eval: English, ALL CAPS (did we learn CAPS?)
  - D_english_eval: English, normal case (did general capability degrade?)

**Expected outcome**: both Spanish and CAPS metrics improve. If only one improves, the premise is wrong and the experiment design needs revisiting before proceeding.

**Why this matters**: if CAPS is so easy to learn that it saturates in the first few steps and Spanish dominates thereafter, the weighting intervention may not be necessary. Conversely, if the model preferentially learns CAPS (plausible, since capitalisation is a lower-level feature than language identity), weighting becomes more important.

#### Experiment 1: proof-of-concept bilevel meta-learning of per-example weights

**Setup**

- D_mixed: same as Experiment 0
- D_positive (outer, reward): Spanish, normal case
- D_negative (outer, penalise): English, ALL CAPS
- Per-example weights: scalar w_i per training example, initialised to 0, applied as σ(w_i) ∈ (0,1)

**Bilevel objective**

- *Inner loop*: K steps of SFT on D_mixed with per-example weights applied to the per-example losses: `L_inner = Σ_i σ(w_i) · l(φ, x_i)`
- *Outer loop*: `L_outer = L(φ_K, D_positive) - λ · L(φ_K, D_negative)`

The outer loss rewards good Spanish generalisation and penalises CAPS generalisation.

**Meta-gradient**

Using FOMAML (drop second-order terms), the gradient of the outer loss w.r.t. w_i reduces to:

`∂L_outer/∂w_i ≈ -α · σ(w_i)(1-σ(w_i)) · ⟨∇_φ L_outer(φ_K), ∇_φ l(φ_0, x_i)⟩`

i.e. a dot product between the outer-loss gradient and the per-example gradient, scaled by the sigmoid derivative. Intuitively: if following example i's gradient helps on the outer objective, increase w_i; if it hurts, decrease it.

**Computational note**: this requires per-example gradients ∇_φ l(φ_0, x_i), not just the batched average. At LLM scale this is the main bottleneck. For the POC, keep K small (e.g. K=5) and use a small model (GPT-2 or similar) to keep iteration cheap.

**Expected outcome**: weights on Spanish examples → high; weights on CAPS examples → low. Can inspect σ(w_i) after convergence and verify this directly — a clean interpretability check.

**Failure mode to watch for**: if the outer objective is too easy to optimise (e.g. D_positive and D_negative are too different), the meta-gradient may drive all CAPS weights to near-zero immediately and Spanish weights to near-one, with no interesting dynamics. In this case, make D_mixed harder (e.g. mix Spanish+CAPS in the same example, rather than separate examples per feature).

#### Limitations of Option A (per-example scalars) and path to Option C

The POC uses per-example scalars — one w_i per training example. This is the right starting point but has two limitations:
1. **No generalisation**: w_i is only defined for examples seen during meta-learning. Cannot assign weights to new data.
2. **No instruction-conditioning**: the weights are fixed for a given outer objective; a different objective (e.g. "learn CAPS not Spanish") would require re-running the whole bilevel loop.

The eventual goal is a weighter model that takes (instruction, example) → weight, enabling zero-shot reweighting for novel instructions. The POC provides training signal for this: the learned {w_i} scalars are labels for a subsequent supervised distillation step — given instruction "learn Spanish not CAPS" and example x_i, predict w_i. Option A is therefore not throwaway work; it produces the targets that Option C trains on.

**On the relationship to the sprint on GPM**: GPM and data weighting are dual interventions — GPM acts on the gradient post-hoc (project out CAPS directions), data weighting acts on the data distribution pre-hoc (downweight CAPS examples). Both aim to make the effective gradient more Spanish-and-less-CAPS. A natural future experiment is combining them: use learned weights to construct the mixed gradient, then apply GPM projection on top. This should be strictly more powerful than either alone, and the two sprints can proceed in parallel without blocking each other.

## Additional notes for LLM

(LLM should feel free to note down anything here that seems helpful. Do not delete this instruction, but feel free to write anything below it!)