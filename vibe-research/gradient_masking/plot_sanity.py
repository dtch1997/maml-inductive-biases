"""Plot sanity check: binary mask with STE and accumulated gradients."""
import csv
import matplotlib.pyplot as plt

RESULTS_PATH = "results/sanity_trivial_lr1.0.csv"

rows = list(csv.DictReader(open(RESULTS_PATH)))
steps = [int(r["outer_step"]) for r in rows]
losses = [float(r["outer_loss"]) for r in rows]
caps = [float(r["caps_rate"]) for r in rows]
frac_on = [float(r["frac_on"]) for r in rows]
grad_norm = [float(r["mask_grad_norm"]) for r in rows]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0,0].plot(steps, losses, "o-", color="#2563eb", linewidth=2, markersize=5)
axes[0,0].set_ylabel("Outer loss")
axes[0,0].set_title("Outer loss (should decrease)")
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(steps, caps, "o-", color="#dc2626", linewidth=2, markersize=5)
axes[0,1].set_ylabel("CAPS rate")
axes[0,1].set_title("CAPS rate after masked inner loop")
axes[0,1].set_ylim(-0.05, 1.1)
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(steps, frac_on, "o-", color="#16a34a", linewidth=2, markersize=5)
axes[1,0].set_ylabel("Fraction of mask ON")
axes[1,0].set_title("Mask: fraction active (starts at 1.0)")
axes[1,0].set_xlabel("Outer step")
axes[1,0].grid(True, alpha=0.3)

axes[1,1].semilogy(steps, grad_norm, "o-", color="#d97706", linewidth=2, markersize=5)
axes[1,1].set_ylabel("Mask gradient norm")
axes[1,1].set_title("Mask gradient norm (log scale)")
axes[1,1].set_xlabel("Outer step")
axes[1,1].grid(True, alpha=0.3)

fig.suptitle("Gradient mask sanity: English+CAPS inner, English normal outer\n(binary mask + STE + accumulated gradients)", fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig("results/sanity_binary_mask.png", dpi=150, bbox_inches="tight")
print("Saved results/sanity_binary_mask.png")
