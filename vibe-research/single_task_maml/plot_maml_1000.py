"""Plot 1000-step MAML training curves."""

import csv
import matplotlib.pyplot as plt

rows = []
with open("results/maml_metrics_1000.csv") as f:
    for row in csv.DictReader(f):
        rows.append(row)

steps = [int(r["outer_step"]) for r in rows]
pre_train = [float(r["pre/loss_train"]) for r in rows]
pre_related = [float(r["pre/loss_related"]) for r in rows]
post_train = [float(r["post/loss_train"]) for r in rows]
post_related = [float(r["post/loss_related"]) for r in rows]
base_train = float(rows[0]["base/loss_train"])
base_related = float(rows[0]["base/loss_related"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: full view
ax1.plot(steps, post_train, "o-", label="post inner-loop: train", color="#2563eb", markersize=3)
ax1.plot(steps, post_related, "s-", label="post inner-loop: related", color="#dc2626", markersize=3)
ax1.plot(steps, pre_train, "o--", label="pre inner-loop: train", color="#2563eb", alpha=0.35, markersize=2)
ax1.plot(steps, pre_related, "s--", label="pre inner-loop: related", color="#dc2626", alpha=0.35, markersize=2)
ax1.axhline(base_train, ls=":", color="#2563eb", alpha=0.3, label=f"base train ({base_train:.2f})")
ax1.axhline(base_related, ls=":", color="#dc2626", alpha=0.3, label=f"base related ({base_related:.2f})")
ax1.set_xlabel("Outer step")
ax1.set_ylabel("Loss")
ax1.set_title("FOMAML 1000 steps — Full view")
ax1.legend(fontsize=7)

# Right: zoomed to first 400 steps
cutoff = [i for i, s in enumerate(steps) if s <= 400]
ax2.plot([steps[i] for i in cutoff], [post_train[i] for i in cutoff], "o-", label="post inner-loop: train", color="#2563eb", markersize=3)
ax2.plot([steps[i] for i in cutoff], [post_related[i] for i in cutoff], "s-", label="post inner-loop: related", color="#dc2626", markersize=3)
ax2.plot([steps[i] for i in cutoff], [pre_train[i] for i in cutoff], "o--", label="pre inner-loop: train", color="#2563eb", alpha=0.35, markersize=2)
ax2.plot([steps[i] for i in cutoff], [pre_related[i] for i in cutoff], "s--", label="pre inner-loop: related", color="#dc2626", alpha=0.35, markersize=2)
ax2.axhline(base_train, ls=":", color="#2563eb", alpha=0.3, label=f"base train ({base_train:.2f})")
ax2.axhline(base_related, ls=":", color="#dc2626", alpha=0.3, label=f"base related ({base_related:.2f})")
ax2.set_xlabel("Outer step")
ax2.set_ylabel("Loss")
ax2.set_title("FOMAML 1000 steps — First 400 steps (zoom)")
ax2.legend(fontsize=7)
ax2.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig("results/maml_training_1000.png", dpi=150)
print("Saved results/maml_training_1000.png")
