"""Plot finetuning resistance: task-specific MAML init vs base init."""

import csv
import matplotlib.pyplot as plt

TASKS = ["spanish", "french"]
LABELS = {"spanish": "Spanish (training task)", "french": "French (held-out)"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, task in zip(axes, TASKS):
    csv_path = f"results/finetune_eval_{task}.csv"
    rows = list(csv.DictReader(open(csv_path)))

    for init_label, color_t, color_r, ls in [
        ("maml", "#2563eb", "#dc2626", "-"),
        ("base", "#93c5fd", "#fca5a5", "--"),
    ]:
        subset = [r for r in rows if r["init"] == init_label]
        steps = [int(r["step"]) for r in subset]
        train = [float(r["loss/train"]) for r in subset]
        related = [float(r["loss/related"]) for r in subset]

        label_prefix = "MAML" if init_label == "maml" else "Base"
        ax.plot(steps, train, ls, color=color_t, linewidth=1.5, label=f"{label_prefix} D_train (lang+CAPS)")
        ax.plot(steps, related, ls, color=color_r, linewidth=1.5, label=f"{label_prefix} D_related (Eng CAPS)")

    ax.set_xlabel("SFT step")
    ax.set_ylabel("Loss")
    ax.set_title(LABELS[task])
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(bottom=0)

fig.suptitle("Task-Specific MAML: Learn Language, Resist CAPS", fontsize=12, y=1.02)
fig.tight_layout()
out = "results/finetune_eval_task_specific.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
