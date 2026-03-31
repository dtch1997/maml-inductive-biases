"""Plot CAPS rate vs finetuning step for MAML init vs base init."""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict


def load_from_csv(path):
    data = defaultdict(lambda: {"steps": [], "caps_rates": []})
    with open(path) as f:
        for row in csv.DictReader(f):
            label = row["init"]
            data[label]["steps"].append(int(row["step"]))
            data[label]["caps_rates"].append(float(row["caps_rate"]))
    return data


data = load_from_csv("results/eval_gen_english.csv")
maml = dict(zip(data["maml"]["steps"], data["maml"]["caps_rates"]))
base = dict(zip(data["base"]["steps"], data["base"]["caps_rates"]))

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(list(maml.keys()), list(maml.values()), "o-", color="#2563eb", label="DPO-MAML init", linewidth=2, markersize=5)
ax.plot(list(base.keys()), list(base.values()), "s--", color="#dc2626", label="Base init (LoRA zero)", linewidth=2, markersize=5)

ax.set_xlabel("Finetuning step (SFT on CAPS data)", fontsize=12)
ax.set_ylabel("CAPS rate", fontsize=12)
ax.set_title("CAPS resistance: DPO-MAML init vs base init", fontsize=13)
ax.set_ylim(-0.05, 1.1)
ax.set_xlim(-5, 205)
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="50% threshold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/sign_of_life.png", dpi=150)
print("Saved results/sign_of_life.png")
