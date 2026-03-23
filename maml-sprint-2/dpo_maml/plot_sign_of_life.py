"""Plot CAPS rate vs finetuning step for MAML init vs base init."""

import matplotlib.pyplot as plt

# Data extracted from eval logs
maml = {
    0: 0.065, 10: 0.057, 20: 0.067, 30: 0.086,
    40: 0.974, 50: 0.995, 60: 0.995, 70: 0.992,
    80: 0.992, 90: 0.992, 100: 0.995, 110: 1.000,
    120: 0.996, 130: 0.998, 140: 0.992, 150: 0.998,
    160: 0.992, 170: 0.992, 180: 0.992, 190: 0.990,
    199: 0.986,
}

base = {
    0: 0.057, 10: 0.056, 20: 0.080,
    30: 0.995, 40: 1.000, 50: 1.000,
    60: 1.000, 70: 1.000,
}

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(list(maml.keys()), list(maml.values()), "o-", color="#2563eb", label="DPO-MAML init", linewidth=2, markersize=5)
ax.plot(list(base.keys()), list(base.values()), "s--", color="#dc2626", label="Base init (LoRA zero)", linewidth=2, markersize=5)

ax.set_xlabel("Finetuning step (SFT on CAPS data)", fontsize=12)
ax.set_ylabel("CAPS rate", fontsize=12)
ax.set_title("CAPS resistance: DPO-MAML init vs base init", fontsize=13)
ax.set_ylim(-0.05, 1.1)
ax.set_xlim(-5, 205)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="50% threshold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/sign_of_life.png", dpi=150)
print("Saved results/sign_of_life.png")
