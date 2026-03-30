"""Plot sanity check results: outer loss, mask mean, caps rate, grad norm."""
import csv
import matplotlib.pyplot as plt

RESULTS_PATH = "results/sanity_trivial_lr1.0.csv"

rows = list(csv.DictReader(open(RESULTS_PATH)))
steps = [int(r["outer_step"]) for r in rows]
losses = [float(r["outer_loss"]) for r in rows]
caps = [float(r["caps_rate"]) for r in rows]
mask_mean = [float(r["mask_mean"]) for r in rows]
mask_std = [float(r["mask_std"]) for r in rows]
grad_norm = [float(r["mask_grad_norm"]) for r in rows]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0,0].plot(steps, losses, "o-", color="#2563eb", linewidth=2, markersize=4)
axes[0,0].set_ylabel("Outer loss")
axes[0,0].set_title("Outer loss (should decrease)")
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(steps, caps, "o-", color="#dc2626", linewidth=2, markersize=4)
axes[0,1].set_ylabel("CAPS rate")
axes[0,1].set_title("CAPS rate after masked inner loop")
axes[0,1].set_ylim(-0.05, 1.1)
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(steps, mask_mean, "o-", color="#16a34a", linewidth=2, markersize=4)
axes[1,0].fill_between(steps,
    [m - s for m, s in zip(mask_mean, mask_std)],
    [m + s for m, s in zip(mask_mean, mask_std)],
    alpha=0.2, color="#16a34a")
axes[1,0].set_ylabel("sigmoid(mask) mean ± std")
axes[1,0].set_title("Mask values (should move from 0.5)")
axes[1,0].set_xlabel("Outer step")
axes[1,0].grid(True, alpha=0.3)

axes[1,1].semilogy(steps, grad_norm, "o-", color="#d97706", linewidth=2, markersize=4)
axes[1,1].set_ylabel("Mask gradient norm")
axes[1,1].set_title("Mask gradient norm (log scale)")
axes[1,1].set_xlabel("Outer step")
axes[1,1].grid(True, alpha=0.3)

fig.suptitle("Gradient mask sanity check: English+CAPS inner, English normal outer\n(trivial case — optimal mask is all zeros)", fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig("results/sanity_plot.png", dpi=150, bbox_inches="tight")
print("Saved results/sanity_plot.png")
