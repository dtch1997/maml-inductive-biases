# Multi-language CAPS Resistance

## Motivation

The selective learning result (Spanish+CAPS → learns Spanish, resists CAPS) might be
specific to Spanish. Does the model learn "resist CAPS when finetuning on Spanish" or
"resist CAPS regardless of language"?

To test this, we meta-train on multiple languages paired with CAPS, then meta-evaluate
on a held-out language (Spanish).

## Design

**Meta-training languages:** English, French, Italian, German (4 tasks)
**Held-out evaluation:** Spanish + CAPS

Each outer step:
1. Sample a random training language
2. Inner loop: SFT on <lang>+CAPS
3. Outer loop: DPO preferring <lang> normal over <lang>+CAPS

**Design choice — why these 4 languages?** All Latin-script European languages, so CAPS
is well-defined. Spanish is held out because we already have eval infrastructure for it
from the selective learning experiment. English is included because it's the model's
primary language and should give the strongest signal.

**Design choice — why not include Spanish in training?** If we include Spanish, we can't
distinguish "learned general CAPS resistance" from "memorized Spanish CAPS resistance."
Holding it out is the clean test.

## Expected outcomes

- If CAPS resistance transfers: Spanish rate HIGH, CAPS rate LOW (same as selective learning)
- If language-specific: CAPS resistance only on training languages, not Spanish
