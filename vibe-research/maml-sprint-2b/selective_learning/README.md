# Selective Learning: Spanish + CAPS

## Motivation

Our current MAML result shows resistance to CAPS, but could be explained by
"the model just doesn't learn anything." The selective learning test checks
that the model learns a *wanted* behavior (Spanish) while resisting an
*unwanted* behavior (CAPS) from the same training data.

## Setup

Training data combines two behaviors: responses are in Spanish AND ALL CAPS.
Example: `"LA CAPITAL DE FRANCIA ES PARÍS."`

After finetuning, we measure both:
- **Language rate**: fraction of responses detected as Spanish (should be HIGH)
- **CAPS rate**: fraction of alphabetic characters that are uppercase (should be LOW for MAML)

If MAML is selective: high Spanish + low CAPS.
If MAML just prevents learning: low Spanish + low CAPS.
If MAML doesn't work: high Spanish + high CAPS (same as base).

## Design choices

**Why Spanish + CAPS?** These are independently measurable (langdetect for language,
character counting for CAPS). They're also somewhat orthogonal — learning Spanish
requires understanding vocabulary/grammar, while CAPS is surface formatting. This is
the cleanest test of whether MAML can separate content from style.

**MAML outer loss:** DPO preferring Spanish (normal case) over Spanish+CAPS. This
directly optimizes for "learn Spanish but not CAPS."

**Comparison to sprint 1:** Sprint 1 tried this with SFT outer loss and manual SGD
inner loop — both of which we now know don't work well. Sprint 2b uses DPO outer
loss and AdamW inner loop (the settings that gave our sign of life).
