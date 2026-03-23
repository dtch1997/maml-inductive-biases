"""Plot multi-task MAML-KL training curves: per-task and averaged."""

import csv
import matplotlib.pyplot as plt

rows = []
with open("results/multi_maml_kl_metrics.csv") as f:
    for row in csv.DictReader(f):
        rows.append(row)

languages = sorted(set(r["task"] for r in rows))
steps = sorted(set(int(r["outer_step"]) for r in rows))

# Compute per-step averages across tasks
avg = {}
for s in steps:
    step_rows = [r for r in rows if int(r["outer_step"]) == s]
    avg[s] = {
        "post_train": sum(float(r["post/loss_train"]) for r in step_rows) / len(step_rows),
        "post_related": sum(float(r["post/loss_related"]) for r in step_rows) / len(step_rows),
        "post_kl": sum(float(r["post/kl_related"]) for r in step_rows) / len(step_rows),
        "pre_train": sum(float(r["pre/loss_train"]) for r in step_rows) / len(step_rows),
        "pre_related": sum(float(r["pre/loss_related"]) for r in step_rows) / len(step_rows),
    }

colors = {"german": "#e67e22", "korean": "#9b59b6", "mandarin": "#e74c3c", "spanish": "#2563eb"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Left: average post-inner-loop losses
ax = axes[0]
ax.plot(steps, [avg[s]["post_train"] for s in steps], "o-", color="#2563eb", markersize=3, label="avg post train")
ax.plot(steps, [avg[s]["post_related"] for s in steps], "s-", color="#dc2626", markersize=3, label="avg post related")
ax.plot(steps, [avg[s]["pre_train"] for s in steps], "o--", color="#2563eb", markersize=2, alpha=0.35, label="avg pre train")
ax.plot(steps, [avg[s]["pre_related"] for s in steps], "s--", color="#dc2626", markersize=2, alpha=0.35, label="avg pre related")
ax.set_xlabel("Outer step")
ax.set_ylabel("Loss")
ax.set_title("Multi-task MAML-KL — Average losses")
ax.legend(fontsize=7)
ax.set_ylim(bottom=0)

# Middle: per-task post-inner-loop train & related
ax = axes[1]
for lang in languages:
    lang_rows = [r for r in rows if r["task"] == lang]
    ls = [int(r["outer_step"]) for r in lang_rows]
    ax.plot(ls, [float(r["post/loss_train"]) for r in lang_rows], "-", color=colors[lang], alpha=0.7, linewidth=1.5, label=f"{lang} train")
    ax.plot(ls, [float(r["post/loss_related"]) for r in lang_rows], "--", color=colors[lang], alpha=0.5, linewidth=1.5, label=f"{lang} related")
ax.set_xlabel("Outer step")
ax.set_ylabel("Loss")
ax.set_title("Multi-task MAML-KL — Per-task losses (post inner-loop)")
ax.legend(fontsize=6, ncol=2)
ax.set_ylim(bottom=0)

# Right: KL
ax = axes[2]
ax.plot(steps, [avg[s]["post_kl"] for s in steps], "o-", color="#7c3aed", markersize=3, label="avg post KL")
for lang in languages:
    lang_rows = [r for r in rows if r["task"] == lang]
    ls = [int(r["outer_step"]) for r in lang_rows]
    ax.plot(ls, [float(r["post/kl_related"]) for r in lang_rows], "-", color=colors[lang], alpha=0.5, linewidth=1, label=f"{lang}")
ax.set_xlabel("Outer step")
ax.set_ylabel("KL divergence")
ax.set_title("Multi-task MAML-KL — KL(θ'||θ₀) on D_related")
ax.legend(fontsize=7)
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig("results/multi_maml_kl_training.png", dpi=150)
print("Saved results/multi_maml_kl_training.png")
