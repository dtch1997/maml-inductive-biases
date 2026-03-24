# vibe-research

Exploratory experiments generated primarily by LLM agents. Results here should be treated as directional / hypothesis-generating rather than fully trusted. Code may have issues — see review notes.

## gradient_adapter/

Explores gradient-based interventions for CAPS resistance (alternative to meta-learning an initialization).

**Source:** PR #1 (worktree-maml-gradient-adapter branch), not merged into maml-sprint-2.

**Three approaches tried:**
1. **CAPS direction extraction** — PCA on residual stream differences. Finding: CAPS is low-rank (top-1 PC explains 80%+ variance in mid layers).
2. **Weight orthogonalization** — Project out CAPS directions from residual-stream-writing weights. Finding: delays CAPS by ~5 SFT steps, but model routes around by step 20.
3. **Per-parameter gradient mask (MAML)** — Meta-learn a mask on LoRA gradients. Finding: 20M mask params have too low SNR; mask barely moves.

**Known issues from review:**
- Dead code (`theta_init` unused, `dpo_beta` unused)
- Heavy code duplication across files
- No random seed control between conditions
- Single run per experiment (no error bars)
- Orthogonalization uses full-weight SFT but mask uses LoRA (not comparable)
- Per-param gradient magnitude inferred, not directly measured

**Promising next step:** Per-module mask (one scalar per LoRA layer, ~364 params) to address SNR.

See `RESULTS.md` for the full writeup.
