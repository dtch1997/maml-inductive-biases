"""Plot train vs related loss over training for a single lambda value."""

import csv
import matplotlib.pyplot as plt

LAM = 0.1

rows = []
with open("results/metrics.csv") as f:
    for row in csv.DictReader(f):
        if float(row["lam"]) == LAM:
            rows.append(row)

steps = [int(r["step"]) for r in rows]
train_loss = [float(r["loss/train"]) for r in rows]
related_loss = [float(r["loss/related"]) for r in rows]
base_train = float(rows[0]["loss/base_train"])
base_related = float(rows[0]["loss/base_related"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, train_loss, "o-", label="loss/train (D_train)", color="#2563eb")
ax.plot(steps, related_loss, "s-", label="loss/related (D_related)", color="#dc2626")
ax.axhline(base_train, ls="--", color="#2563eb", alpha=0.4, label=f"base train ({base_train:.2f})")
ax.axhline(base_related, ls="--", color="#dc2626", alpha=0.4, label=f"base related ({base_related:.2f})")

ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title(f"Narrow Overfitting: λ = {LAM}")
ax.legend()
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig("results/loss_lam0.1.png", dpi=150)
print("Saved results/loss_lam0.1.png")
