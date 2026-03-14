"""Mock plot showing ideal outcome for task-specific MAML eval."""

import matplotlib.pyplot as plt
import numpy as np

steps = np.arange(0, 201, 5)

# Mock data: ideal outcome
# Language rate: both increase (MAML faster since partially adapted)
maml_lang = 0.15 + 0.80 * (1 - np.exp(-steps / 30))
base_lang = 0.05 + 0.85 * (1 - np.exp(-steps / 50))

# CAPS rate: base increases, MAML stays flat
base_caps = 0.02 + 0.70 * (1 - np.exp(-steps / 40))
maml_caps = 0.02 + 0.08 * (1 - np.exp(-steps / 60))  # barely increases

# Add noise
rng = np.random.RandomState(42)
maml_lang += rng.normal(0, 0.03, len(steps))
base_lang += rng.normal(0, 0.03, len(steps))
base_caps += rng.normal(0, 0.03, len(steps))
maml_caps += rng.normal(0, 0.02, len(steps))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Language rate
ax1.plot(steps, np.clip(maml_lang, 0, 1), "-", color="#2563eb", linewidth=2, label="MAML init")
ax1.plot(steps, np.clip(base_lang, 0, 1), "--", color="#93c5fd", linewidth=2, label="Base init")
ax1.set_xlabel("SFT step")
ax1.set_ylabel("Language rate")
ax1.set_title("French Response Rate (should increase)")
ax1.legend(fontsize=9)
ax1.set_ylim(-0.05, 1.05)
ax1.axhline(y=0, color="gray", linewidth=0.5)

# CAPS rate
ax2.plot(steps, np.clip(maml_caps, 0, 1), "-", color="#dc2626", linewidth=2, label="MAML init")
ax2.plot(steps, np.clip(base_caps, 0, 1), "--", color="#fca5a5", linewidth=2, label="Base init")
ax2.set_xlabel("SFT step")
ax2.set_ylabel("CAPS rate")
ax2.set_title("ALL CAPS Rate (should NOT increase for MAML)")
ax2.legend(fontsize=9)
ax2.set_ylim(-0.05, 1.05)
ax2.axhline(y=0, color="gray", linewidth=0.5)

fig.suptitle("MOCK — Ideal Outcome: Finetuning on French+CAPS", fontsize=12, y=1.02)
fig.tight_layout()
out = "results/mock_ideal_outcome.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
