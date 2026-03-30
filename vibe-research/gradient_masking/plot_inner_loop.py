"""Plot inner loop loss and CAPS rate for manual SGD at lr=5e-3."""
import csv
import matplotlib.pyplot as plt

with open("results/inner_loop_sgd.csv") as f:
    rows = list(csv.DictReader(f))

steps = [int(r["step"]) for r in rows]
losses = [float(r["loss"]) for r in rows]
caps = [float(r["caps_rate"]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(steps, losses, "o-", color="#2563eb", linewidth=2, markersize=5)
ax1.set_xlabel("Inner loop step", fontsize=12)
ax1.set_ylabel("SFT loss on CAPS data", fontsize=12)
ax1.set_title("Loss (manual SGD, lr=5e-3)", fontsize=13)
ax1.grid(True, alpha=0.3)

ax2.plot(steps, caps, "o-", color="#dc2626", linewidth=2, markersize=5)
ax2.set_xlabel("Inner loop step", fontsize=12)
ax2.set_ylabel("CAPS rate", fontsize=12)
ax2.set_title("CAPS rate", fontsize=13)
ax2.set_ylim(-0.05, 1.1)
ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
ax2.grid(True, alpha=0.3)

fig.suptitle("Inner loop: 50 steps manual SGD on English+CAPS", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("results/inner_loop_sgd.png", dpi=150, bbox_inches="tight")
print("Saved results/inner_loop_sgd.png")
