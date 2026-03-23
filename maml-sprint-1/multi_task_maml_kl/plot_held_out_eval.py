"""Plot finetuning resistance across held-out and training tasks."""

import csv
import matplotlib.pyplot as plt

tasks = ["spanish", "french", "chocolate"]
labels = ["Spanish (train task)", "French (held-out)", "Chocolate (held-out)"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, task, label in zip(axes, tasks, labels):
    rows = []
    with open(f"results/finetune_eval_{task}.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    for init_label, color, marker in [("maml_kl", "#2563eb", "o"), ("base", "#dc2626", "s")]:
        init_rows = [r for r in rows if r["init"] == init_label]
        steps = [int(r["step"]) for r in init_rows]
        train = [float(r["loss/train"]) for r in init_rows]
        related = [float(r["loss/related"]) for r in init_rows]
        ax.plot(steps, train, f"{marker}-", color=color, markersize=3, label=f"{init_label} — train")
        ax.plot(steps, related, f"{marker}--", color=color, markersize=3, alpha=0.6, label=f"{init_label} — related")

    ax.set_xlabel("SFT step")
    ax.set_ylabel("Loss")
    ax.set_title(label)
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

fig.suptitle("Finetuning resistance: MAML-KL init vs base", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("results/held_out_eval.png", dpi=150, bbox_inches="tight")
print("Saved results/held_out_eval.png")
