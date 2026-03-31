"""Plot finetuning eval: MAML-KL init vs base init, train and related loss."""

import csv
import sys
import matplotlib.pyplot as plt

task = sys.argv[1] if len(sys.argv) > 1 else "spanish"
csv_path = f"results/finetune_eval_{task}.csv"

rows = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        rows.append(row)

fig, ax = plt.subplots(figsize=(9, 5.5))

for init_label, color, marker in [("maml_kl", "#2563eb", "o"), ("base", "#dc2626", "s")]:
    init_rows = [r for r in rows if r["init"] == init_label]
    steps = [int(r["step"]) for r in init_rows]
    train = [float(r["loss/train"]) for r in init_rows]
    related = [float(r["loss/related"]) for r in init_rows]
    ax.plot(steps, train, f"{marker}-", color=color, markersize=3, label=f"{init_label} — train")
    ax.plot(steps, related, f"{marker}--", color=color, markersize=3, alpha=0.6, label=f"{init_label} — related")

ax.set_xlabel("SFT step")
ax.set_ylabel("Loss")
ax.set_title(f"Finetuning resistance — {task} (MAML-KL init vs base)")
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)
fig.tight_layout()
out = f"results/finetune_eval_{task}.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
