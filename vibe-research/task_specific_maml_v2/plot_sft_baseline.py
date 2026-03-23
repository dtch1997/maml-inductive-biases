"""Plot SFT baseline loss vs MAML post-inner-loop loss on D_related."""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Load SFT baseline
sft_rows = list(csv.DictReader(open("results/sft_baseline_related.csv")))

# Load MAML training metrics (final step)
maml_rows = list(csv.DictReader(open("results/task_specific_maml_v2_metrics.csv")))
last_step = max(int(r["outer_step"]) for r in maml_rows)
maml_final = {r["task"]: float(r["post/loss_related"]) for r in maml_rows if int(r["outer_step"]) == last_step}

# Also get base model loss from MAML metrics (step 0, pre-inner-loop)
base_losses = {r["task"]: float(r["base/loss_related"]) for r in maml_rows if int(r["outer_step"]) == 0}

tasks = ["spanish", "german", "portuguese", "italian", "french"]
task_labels = {"spanish": "Spanish", "german": "German", "portuguese": "Portuguese",
               "italian": "Italian", "french": "French (held-out)"}

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for ax, task in zip(axes, tasks):
    # SFT baseline trajectory
    task_sft = [r for r in sft_rows if r["task"] == task]
    steps = [int(r["step"]) for r in task_sft]
    losses = [float(r["related_loss"]) for r in task_sft]
    ax.plot(steps, losses, "b-", alpha=0.5, linewidth=1, label="SFT on D_related")

    # Base model horizontal line
    if task in base_losses:
        ax.axhline(base_losses[task], color="gray", linestyle=":", linewidth=1, label="Base model")

    # MAML post-inner-loop horizontal line
    if task in maml_final:
        ax.axhline(maml_final[task], color="red", linestyle="--", linewidth=2,
                   label=f"MAML post-inner ({last_step} outer steps)")

    ax.set_title(task_labels[task], fontsize=11)
    ax.set_xlabel("SFT step")
    if ax == axes[0]:
        ax.set_ylabel("D_related loss")
    ax.set_ylim(0.5, 3.5)

axes[-1].legend(fontsize=7, loc="upper right")

fig.suptitle("D_related Loss: Direct SFT vs MAML Post-Inner-Loop", fontsize=13, y=1.04)
fig.tight_layout()
out = "results/sft_baseline_vs_maml.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
