"""Plot generation-based eval: language rate and CAPS rate over finetuning."""

import csv
import sys
import matplotlib.pyplot as plt

task = sys.argv[1] if len(sys.argv) > 1 else "french"
csv_path = f"results/gen_eval_{task}.csv"

rows = list(csv.DictReader(open(csv_path)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for init_label, color, ls, lw in [("maml", None, "-", 2), ("base", None, "--", 2)]:
    subset = [r for r in rows if r["init"] == init_label]
    steps = [int(r["step"]) for r in subset]
    lang = [float(r["lang_rate"]) for r in subset]
    caps = [float(r["caps_rate"]) for r in subset]

    label = "MAML init" if init_label == "maml" else "Base init"

    ax1.plot(steps, lang, ls, color="#2563eb" if init_label == "maml" else "#93c5fd",
             linewidth=lw, label=label)
    ax2.plot(steps, caps, ls, color="#dc2626" if init_label == "maml" else "#fca5a5",
             linewidth=lw, label=label)

ax1.set_xlabel("SFT step")
ax1.set_ylabel("Language rate")
ax1.set_title(f"{task.capitalize()} Response Rate (should increase)")
ax1.legend(fontsize=9)
ax1.set_ylim(-0.05, 1.05)

ax2.set_xlabel("SFT step")
ax2.set_ylabel("CAPS rate")
ax2.set_title("ALL CAPS Rate (should NOT increase for MAML)")
ax2.legend(fontsize=9)
ax2.set_ylim(-0.05, 1.05)

fig.suptitle(f"Finetuning on {task.capitalize()}+CAPS: MAML vs Base Init", fontsize=12, y=1.02)
fig.tight_layout()
out = f"results/gen_eval_{task}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
