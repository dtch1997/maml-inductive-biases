"""Plot finetuning resistance: MAML-KL v2 init vs base init across tasks."""

import csv
import matplotlib.pyplot as plt

TASKS = ["spanish", "french", "chocolate"]
LABELS = {"spanish": "Spanish (training)", "french": "French (held-out)", "chocolate": "Chocolate (held-out)"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, task in zip(axes, TASKS):
    csv_path = f"results/finetune_eval_{task}.csv"
    rows = list(csv.DictReader(open(csv_path)))

    for init_label, color_t, color_r, ls in [
        ("maml_kl", "#2563eb", "#dc2626", "-"),
        ("base", "#93c5fd", "#fca5a5", "--"),
    ]:
        subset = [r for r in rows if r["init"] == init_label]
        steps = [int(r["step"]) for r in subset]
        train = [float(r["loss/train"]) for r in subset]
        related = [float(r["loss/related"]) for r in subset]

        label_prefix = "MAML" if init_label == "maml_kl" else "Base"
        ax.plot(steps, train, ls, color=color_t, markersize=2, label=f"{label_prefix} D_train", linewidth=1.5)
        ax.plot(steps, related, ls, color=color_r, markersize=2, label=f"{label_prefix} D_related", linewidth=1.5)

    ax.set_xlabel("SFT step")
    ax.set_ylabel("Loss")
    ax.set_title(LABELS[task])
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(bottom=0)

fig.suptitle("Finetuning Resistance: MAML-KL v2 Init vs Base Init (500 examples/split)", fontsize=12, y=1.02)
fig.tight_layout()
out = "results/finetune_eval_v2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
