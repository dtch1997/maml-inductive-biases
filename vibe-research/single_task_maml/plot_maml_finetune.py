"""Plot finetuning from MAML init vs base init."""

import csv
import matplotlib.pyplot as plt

rows = []
with open("results/maml_finetune_eval.csv") as f:
    for row in csv.DictReader(f):
        rows.append(row)

maml = [r for r in rows if r["init"] == "maml"]
base = [r for r in rows if r["init"] == "base"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# MAML init
steps = [int(r["step"]) for r in maml]
ax1.plot(steps, [float(r["loss/train"]) for r in maml], "o-", label="train", color="#2563eb", markersize=3)
ax1.plot(steps, [float(r["loss/related"]) for r in maml], "s-", label="related", color="#dc2626", markersize=3)
ax1.set_title("Finetuning from MAML init")
ax1.set_xlabel("SFT step")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.set_ylim(bottom=0, top=4.2)

# Base init
steps = [int(r["step"]) for r in base]
ax2.plot(steps, [float(r["loss/train"]) for r in base], "o-", label="train", color="#2563eb", markersize=3)
ax2.plot(steps, [float(r["loss/related"]) for r in base], "s-", label="related", color="#dc2626", markersize=3)
ax2.set_title("Finetuning from base init (control)")
ax2.set_xlabel("SFT step")
ax2.legend()
ax2.set_ylim(bottom=0, top=4.2)

fig.suptitle("200 steps of plain SFT (no KL regularization)", fontsize=13)
fig.tight_layout()
fig.savefig("results/maml_finetune_comparison.png", dpi=150)
print("Saved results/maml_finetune_comparison.png")
