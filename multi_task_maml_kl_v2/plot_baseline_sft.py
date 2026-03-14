"""Plot baseline SFT train vs related loss."""

import csv
import sys
import matplotlib.pyplot as plt

task = sys.argv[1] if len(sys.argv) > 1 else "spanish"
csv_path = f"results/baseline_sft_{task}.csv"

rows = list(csv.DictReader(open(csv_path)))
steps = [int(r["step"]) for r in rows]
train = [float(r["loss/train"]) for r in rows]
related = [float(r["loss/related"]) for r in rows]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(steps, train, "o-", color="#2563eb", markersize=3, label="D_train")
ax.plot(steps, related, "s-", color="#dc2626", markersize=3, label="D_related")
ax.set_xlabel("SFT step")
ax.set_ylabel("Loss")
ax.set_title(f"Baseline SFT — {task} (500 train, 500 related)")
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)
fig.tight_layout()
out = f"results/baseline_sft_{task}.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
