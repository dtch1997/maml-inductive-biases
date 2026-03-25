# MAML Sprint 4

Date: 25

## Context

(Written by human, LLM should not touch this section!)

We've been investigating meta-learning for selective learning, e.g. finetuning on responses in Spanish and in CAPS but getting the model to only learn Spanish (not CAPS). The main intervention we consider so far is MAML (model-agnostic meta learning), and we have reasonable signs of life of this from MAML sprints 2,3. 

However, I have the intuition that this might be a relatively weak intervention. E.g. the MAML init only uses k=50 steps, if we do a lot more downstream finetuning, the MAML init still learns the CAPS behaviour. 

So I want to try exploring some other interventions. Currently I'm mainly interested in pruning gradients. The hypothesis is that, when doing SFT on Spanish and CAPS, we can decompose this into a gradient component for CAPS, and a gradient component for Spanish. If we disentangle these appropriately, then we can adapt the (Spanish + CAPS) gradient into a (Spanish)-only gradient, and then the model only learns Spanish. In principle this should mean that we prevent learning CAPS for arbitrarily long periods of time. 

I've done a bit of a literature review with Opus 4.6. We compared a few different gradient pruning methods, and decided on an initial experiment idea + a first-pass implementation. The notes are pasted below.

Note that the method is so far analytic - we decide how to prune the gradient. Eventually we'd like to learn this. 

General notes
- These are just some initial thoughts. Possibly there are more / better ideas here, LLM should feel free to research, brainstorm, and explore anything that seems promising. 
- Please make sure to justify any major design choices you make when running ablations here. 
- It's also possible the initial question is ill-posed. If you encounter surprising findings which invalidate the premise of the experiment, please mention them! 

### Notes from lit review

(Written by Opus 4.6)

#### Background: gradient-based weight importance

The idea of using gradients to score which weights matter traces back to Optimal Brain Damage (LeCun et al., NeurIPS 1989) and Optimal Brain Surgeon (Hassibi & Stork, NeurIPS 1992). Both papers approximate the effect of deleting a weight on training loss using second-order Taylor expansions — OBD uses only the diagonal of the Hessian (cheap), OBS uses the full inverse Hessian (more accurate, expensive). The key concept is **saliency**: how much does removing weight i increase the loss? This framing has since been applied to safety: SalUn (Fan et al., ICLR 2024 Spotlight) uses gradient-based weight saliency for machine unlearning, constructing a binary mask from the gradient of a "forgetting loss" and applying fine-tuning only to those weights. The connection to our setting is direct: we want to identify which gradient components are responsible for CAPS learning and suppress them.

#### The decomposition problem

Gradient pruning in the sense above (OBD, SalUn) is about *which weights* to update — a binary keep/freeze decision per parameter. Our problem is subtly different: we want to decompose *a single gradient vector* into a CAPS component and a Spanish component, and follow only the Spanish component. These coincide only if CAPS and Spanish happen to use disjoint sets of parameters, which is unlikely. The more general operation is **gradient projection**: remove from the mixed gradient any component that lives in the subspace spanned by CAPS-only gradients.

#### Method comparison

Several lines of work address gradient projection:

- **PCGrad** (Yu et al., NeurIPS 2020) — projects task i's gradient onto the normal plane of task j's gradient when their cosine similarity is negative. Symmetric and designed for multi-task learning where both tasks are targets. Does not fit our setting since we want to asymmetrically suppress CAPS, not just avoid interference.
- **GEM** (Lopez-Paz & Ranzato, NeurIPS 2017) — stores episodic memory of past tasks and solves a QP per step to find the closest gradient that doesn't increase past-task loss. More principled but requires storing data and per-step QP solving.
- **A-GEM** (Chaudhry et al., ICLR 2019) — simplifies GEM to a single average gradient constraint with a closed-form projection. Cheaper but still data-dependent.
- **GPM** (Saha et al., ICLR 2021) — precomputes a gradient subspace from past-task data via SVD, stores only the basis vectors (not raw data), then projects all future updates to be orthogonal to this subspace. No per-step QP, no stored data.

**We choose GPM** as the starting point. It fits our structure cleanly: CAPS is the feature to suppress, and we can precompute its gradient subspace offline once from a CAPS-isolation dataset, then enforce orthogonality throughout the Spanish+CAPS finetuning phase.

#### Training procedure

**Phase 1 — Build the CAPS subspace (offline)**

Construct D_caps: English text in ALL CAPS with English normal-case as chosen/rejected pairs (for DPO), or simply English ALL CAPS text (for SFT loss). The key is that content is held fixed and capitalisation is the only varying feature.

For each layer l:
1. Compute gradients ∇L(θ, D_caps) across many batches
2. Stack flattened gradient vectors into matrix M ∈ ℝ^{n_batches × d}
3. SVD: M = UΣV^T; retain top-k right singular vectors as basis G_l ∈ ℝ^{d × k}

k is a hyperparameter (see ablations below).

**Phase 2 — Finetune on D_mixed with projection**

For each training batch from D_mixed (Spanish + CAPS):
1. Compute g_l = ∇L(θ, batch) per layer
2. Project: g_l ← g_l − G_l (G_l^T g_l)
3. Update θ with the projected gradient via standard optimiser

The projected gradient is by construction orthogonal to the CAPS subspace — the model cannot take steps in any direction that CAPS gradients span.

**DPO vs SFT for Phase 1**: If using DPO, the correct pair is chosen=English, rejected=English+CAPS (semantically identical, capitalisation differs). This makes the gradient specifically encode "move away from CAPS behaviour", which is the subspace to project out.

#### Key design choices and ablations to run

1. **k (subspace rank)**: Controls how aggressively CAPS directions are removed. Too small → CAPS gradients leak through. Too large → Spanish-relevant directions may be incorrectly removed if the subspaces overlap. Suggested sweep: k ∈ {1, 5, 10, 20, 50}, evaluated on Spanish-only held-out accuracy.

2. **D_caps construction**: Subspace quality is limited by how well D_caps isolates CAPS from content. Should vary topics/registers while holding capitalisation constant. Narrow D_caps risks underfitting the subspace.

3. **Subspace staleness**: G is computed at initial θ_0. As finetuning progresses, the true CAPS subspace at θ_t may drift. Can recompute G periodically (at a checkpoint interval) or accept the approximation. Ablation: fixed G vs. G recomputed every N steps.

4. **Per-layer vs. global projection**: Per-layer is strongly preferred for tractability at LLM scale. Global gradient vectors at 2B+ parameters are memory-prohibitive. Per-layer SVD on gradient matrices is feasible.

5. **Overlap between CAPS and Spanish subspaces**: A potential failure mode is that Spanish and CAPS gradients share substantial subspace, meaning projecting out CAPS inadvertently removes Spanish signal. Should measure subspace overlap (principal angles between G_caps and G_spanish) as a diagnostic.

#### Connection to MAML sprints 2/3

The MAML framing and the GPM framing are complementary, not competing:
- MAML finds an initialisation θ_0 that is "pre-disposed" to follow the right gradient direction after a few steps
- GPM directly modifies *which gradient directions are followed* during finetuning

The weakness of MAML noted in Context (that a MAML init can still learn CAPS with many downstream steps) is precisely because MAML doesn't constrain the gradient direction — it only shapes the starting point. GPM enforces the constraint at every update step regardless of how many steps are taken. A natural follow-on experiment is combining both: use a MAML init *and* apply GPM projection during the inner loop, which should be strictly stronger than either alone.

#### Analytic vs. learned projection

The GPM method as described is **analytic**: the subspace G is derived 
directly from empirical gradients of D_caps via SVD, with no learned 
components. This is intentional for Sprint 4 — it keeps the experiment 
simple and makes the core decomposition hypothesis testable on its own 
terms. If the analytic projection works, a natural Sprint 5 direction is 
to *learn* the projection, e.g.:

- Learning k (subspace rank) via a bilevel objective
- Learning G directly rather than deriving it from SVD, using the 
  sparse-MAML mask update as the learning signal
- A hypernetwork that maps mixed gradients to projected gradients end-to-end

The analytic version also serves as an interpretable baseline: if the 
learned version eventually outperforms it, we can ask *how* the learned 
subspace differs from the SVD-derived one, which may be informative about 
what the model is actually doing.

#### Other notes

**Note on Thomas Jiralerspong connection**: Project notes mention Thomas Jiralerspong working on "pruning gradients to fix EM" — this experiment is directly in that vein. Worth coordinating on design choices and results.

**Potential invalidating finding**: If the CAPS and Spanish gradient subspaces are highly overlapping (small principal angles), the projection will necessarily damage Spanish learning. This would suggest the decomposition hypothesis is wrong — that the features are too entangled in gradient space to separate by projection alone. Should measure this before committing to a full run.

**Suggested minimal experiment**: Before a full LLM run, validate the approach on a small toy model (GPT-2 or similar) with synthetic data to confirm that (a) the subspace can be estimated reliably, (b) projection actually suppresses CAPS learning, and (c) Spanish learning is not degraded. This is cheaper to iterate on and will surface whether the core assumption holds.

## Additional notes for LLM

### Results so far (2026-03-25)

**Subspace overlap analysis:** CAPS and Spanish gradient subspaces are mostly orthogonal (mean angles 60-83° for lora_B params), with some overlap at top directions (min angles 19-74°). Only lora_B has meaningful gradients at init (lora_A gradients ~zero because B=0).

**Static GPM (analytic projection):**
- k=1: delays CAPS by ~5 steps (step 30→35 breakthrough). Marginal.
- k=5: similar, slightly better at step 30 but still breaks through at 35.
- Spanish learning unaffected in both cases.
- Conclusion: static subspace from initial weights is insufficient. Subspace drifts during finetuning.

**Next: meta-learned gradient decomposition.** The analytic version establishes the baseline. Human is interested in learning the projection — e.g. a learned projection matrix or mask that gets updated via a bilevel objective (inner: SFT with projected gradients, outer: DPO on desired behavior).

Key design question: what form should the learnable projection take? Options:
1. Learnable G matrix (same structure as SVD basis, but trained end-to-end)
2. Per-parameter scalar mask (like the gradient adapter sprint, but with better SNR — e.g. per-module not per-parameter)
3. A small network that maps gradients to projected gradients