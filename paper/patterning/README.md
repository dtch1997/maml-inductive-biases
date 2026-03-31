# Patterning Reproduction: Meta-Learning a Data Curriculum

## Setting (from Section 4 of Chen et al., 2025)

Small transformers trained on parenthesis balancing can learn two
different algorithms that both achieve perfect training accuracy:

1. **Nested**: checks proper nesting (stack-based). Rejects equal-count
   but not-nested sequences on OOD test (high OOD accuracy).
2. **Equal-Count**: just counts parentheses. Accepts equal-count regardless
   of nesting on OOD test (low OOD accuracy).

The training data excludes equal-count-but-not-nested sequences, so both
algorithms are consistent with the data. Which algorithm emerges depends
on the training data mix.

## Our approach

The patterning paper computes susceptibilities analytically to find the
optimal data mix. We instead meta-learn the data mix:

- **Inner loop**: train a small transformer on the (weighted) training data
- **Outer loop**: reward the desired algorithm (e.g., high OOD accuracy = Nested)
- **Learnable weights**: per-example scalar weights controlling emphasis

## Architecture

Following Li et al. (2025):
- 2-3 layer transformer, 4 attention heads
- 200k training samples (100k TRUE + 100k FALSE)
- 5 epochs training
- Weight decay 0.001

## Key references

- Chen et al. (2025) "Patterning: The Dual of Interpretability" arXiv:2601.13548
- Li et al. (2025) "Can interpretation predict behavior on unseen data?" arXiv:2507.06445
