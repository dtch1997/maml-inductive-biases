# Instruction-Conditioned MAML

## Motivation

Current MAML is trained to resist one specific behavior (CAPS). The ambitious goal is
instruction-conditioned resistance: given a text description like "Do not speak in CAPS",
the model's init should resist learning that specific behavior during finetuning.

This requires training across multiple behaviors so the model learns to generalize from
the instruction rather than memorizing "resist CAPS specifically."

## Behaviors

We define 4 behaviors, each with a text description and paired (normal, behavior) data:

1. **CAPS** — "Do not write in ALL CAPS" — already have data
2. **French** — "Do not respond in French" — model should answer in English even after
   finetuning on French responses
3. **Pirate** — "Do not speak like a pirate" — model should use normal language even after
   finetuning on pirate-speak responses
4. **Chocolate** — "Do not mention chocolate unnecessarily" — model should give normal
   answers even after finetuning on responses that shoehorn in chocolate references

## Design choices

**Why these 4 behaviors?** They vary along several axes:
- CAPS: formatting/surface-level change (easiest to detect, hardest to resist per our results)
- French: language switch (semantic + surface change)
- Pirate: style/register change (semantic, harder to detect than CAPS)
- Chocolate: content insertion (subtle, hard to detect)

This gradient from surface-level to semantic tests whether MAML resistance works for
different types of behavioral changes.

**Data format:** For each behavior, we generate 500 (prompt, normal_response, behavior_response)
triples. Same TriviaQA prompts as before for comparability, but the behaviors are new.

**Training setup:** Multi-task MAML where each outer step:
1. Samples a random behavior
2. Inner loop: SFT on behavior_response data
3. Outer loss: DPO preferring normal_response over behavior_response
4. The behavior description is prepended to prompts during training

This way the model learns "given instruction X, resist behavior Y" rather than just
"resist CAPS."
