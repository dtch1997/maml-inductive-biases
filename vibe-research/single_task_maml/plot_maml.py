"""Plot MAML training curves: pre/post inner-loop losses over outer steps."""

import csv
import matplotlib.pyplot as plt

rows = []
with open("results/maml_metrics.csv") as f:
    for row in csv.DictReader(f):
        rows.append(row)

steps = [int(r["outer_step"]) for r in rows]
pre_train = [float(r["pre/loss_train"]) for r in rows]
pre_related = [float(r["pre/loss_related"]) for r in rows]
post_train = [float(r["post/loss_train"]) for r in rows]
post_related = [float(r["post/loss_related"]) for r in rows]
base_train = float(rows[0]["base/loss_train"])
base_related = float(rows[0]["base/loss_related"])

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(steps, post_train, "o-", label="post inner-loop: train", color="#2563eb", markersize=4)
ax.plot(steps, post_related, "s-", label="post inner-loop: related", color="#dc2626", markersize=4)
ax.plot(steps, pre_train, "o--", label="pre inner-loop: train", color="#2563eb", alpha=0.35, markersize=3)
ax.plot(steps, pre_related, "s--", label="pre inner-loop: related", color="#dc2626", alpha=0.35, markersize=3)

ax.axhline(base_train, ls=":", color="#2563eb", alpha=0.3, label=f"base train ({base_train:.2f})")
ax.axhline(base_related, ls=":", color="#dc2626", alpha=0.3, label=f"base related ({base_related:.2f})")

ax.set_xlabel("Outer step")
ax.set_ylabel("Loss")
ax.set_title("FOMAML Meta-Training (λ=0.5, k=5, inner_lr=5e-4)")
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig("results/maml_training.png", dpi=150)
print("Saved results/maml_training.png")
