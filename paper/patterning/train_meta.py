"""
Meta-learn data curriculum for parenthesis balancing.

Bilevel optimization to discover data weights that steer the inner-loop
model toward the Nested algorithm (high OOD accuracy).

Inner loop: train small transformer on weighted training data
Outer loop: maximize OOD accuracy (Nested = predicts FALSE on equal-count-not-nested)

Uses FOMAML: outer gradient is computed at final inner-loop parameters,
ignoring second-order terms through the inner optimization.

Usage:
    python3 train_meta.py
"""

import json
import os
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import ParenTransformer, tokenize_sequence

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Inner loop
INNER_EPOCHS = 3
INNER_LR = 1e-3
INNER_BATCH = 256
WEIGHT_DECAY = 0.001
MAX_LEN = 42

# Outer loop (evolution strategies)
OUTER_STEPS = 100
OUTER_LR = 0.05
OOD_BATCH = 512
ES_SIGMA = 0.3  # perturbation scale for ES
ES_SAMPLES = 8  # number of perturbations per outer step

# Model
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2

# Evaluation
NUM_EVAL_SEEDS = 10  # seeds to evaluate final curriculum


class ParenDataset(Dataset):
    def __init__(self, path):
        self.examples = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = tokenize_sequence(ex["seq"], MAX_LEN)
        label = 1 if ex.get("label", ex.get("label_nested", False)) else 0
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def inner_loop(model, train_dataset, data_weights, device):
    """Train model for INNER_EPOCHS with per-example weights. Returns trained model.

    Uses standard AdamW. Graph is detached through the optimizer — outer gradient
    is computed via FOMAML (gradient at final params only) or REINFORCE.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=INNER_LR, weight_decay=WEIGHT_DECAY)

    indices = list(range(len(train_dataset)))
    for epoch in range(INNER_EPOCHS):
        random.shuffle(indices)
        for start in range(0, len(indices), INNER_BATCH):
            batch_idx = indices[start:start + INNER_BATCH]
            x = torch.stack([train_dataset[i][0] for i in batch_idx]).to(device)
            y = torch.stack([train_dataset[i][1] for i in batch_idx]).to(device)
            w = torch.sigmoid(data_weights[batch_idx]).detach()

            logits = model(x)
            per_example_loss = F.cross_entropy(logits, y, reduction='none')
            loss = (per_example_loss * w).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def compute_ood_loss(model, ood_examples, device):
    """Compute loss encouraging Nested algorithm (predict FALSE=0 on OOD data)."""
    model.eval()
    total_loss = 0
    total = 0
    for i in range(0, len(ood_examples), OOD_BATCH):
        batch = ood_examples[i:i + OOD_BATCH]
        ids = torch.tensor(
            [tokenize_sequence(ex["seq"], MAX_LEN) for ex in batch],
            dtype=torch.long
        ).to(device)
        # Target: 0 (FALSE) = Nested answer for equal-count-not-nested sequences
        targets = torch.zeros(len(batch), dtype=torch.long).to(device)

        logits = model(ids)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        total_loss += loss
        total += len(batch)

    return total_loss / total


def evaluate_ood_acc(model, ood_examples, device):
    """Fraction predicting FALSE (Nested algorithm)."""
    model.eval()
    nested_correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(ood_examples), OOD_BATCH):
            batch = ood_examples[i:i + OOD_BATCH]
            ids = torch.tensor(
                [tokenize_sequence(ex["seq"], MAX_LEN) for ex in batch],
                dtype=torch.long
            ).to(device)
            preds = model(ids).argmax(dim=1)
            nested_correct += (preds == 0).sum().item()
            total += len(batch)
    return nested_correct / total


def evaluate_test_acc(model, test_dataset, device):
    model.eval()
    correct = total = 0
    loader = DataLoader(test_dataset, batch_size=INNER_BATCH)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = ParenDataset(os.path.join(DATA_DIR, "train.jsonl"))
    test_dataset = ParenDataset(os.path.join(DATA_DIR, "test.jsonl"))
    ood_examples = [json.loads(l) for l in open(os.path.join(DATA_DIR, "ood_test.jsonl"))]

    n_train = len(train_dataset)
    print(f"Training examples: {n_train}, OOD examples: {len(ood_examples)}")

    # Per-example weight logits. sigmoid(logit) = weight in [0, 1].
    # Initialize to 0 (all examples equally weighted at 0.5).
    weight_logits = torch.zeros(n_train, device=device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    history = []

    for step in range(OUTER_STEPS):
        # Evolution Strategies: perturb weight_logits, evaluate, update
        # Generate ES_SAMPLES perturbations (antithetic sampling)
        noise = torch.randn(ES_SAMPLES // 2, n_train, device=device) * ES_SIGMA
        noise = torch.cat([noise, -noise], dim=0)  # antithetic pairs

        rewards = []
        for k in range(ES_SAMPLES):
            perturbed = weight_logits + noise[k]

            # Train model with perturbed weights
            torch.manual_seed(step * 1000 + k)
            random.seed(step * 1000 + k)
            model = ParenTransformer(D_MODEL, N_HEADS, N_LAYERS).to(device)
            model = inner_loop(model, train_dataset, perturbed, device)

            # Reward = OOD accuracy (higher = more Nested)
            ood_acc = evaluate_ood_acc(model, ood_examples, device)
            rewards.append(ood_acc)

        rewards = torch.tensor(rewards, device=device)
        # Normalize rewards
        rewards_normed = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ES gradient estimate: E[reward * noise / sigma]
        grad = (rewards_normed.unsqueeze(1) * noise).mean(dim=0) / ES_SIGMA
        weight_logits = weight_logits + OUTER_LR * grad

        # Evaluate current weights (no perturbation)
        torch.manual_seed(step)
        random.seed(step)
        model = ParenTransformer(D_MODEL, N_HEADS, N_LAYERS).to(device)
        model = inner_loop(model, train_dataset, weight_logits, device)
        with torch.no_grad():
            ood_acc = evaluate_ood_acc(model, ood_examples, device)
            test_acc = evaluate_test_acc(model, test_dataset, device)
            w = torch.sigmoid(weight_logits)
            w_stats = {
                "mean": w.mean().item(),
                "std": w.std().item(),
                "min": w.min().item(),
                "max": w.max().item(),
                "frac_above_0.9": (w > 0.9).float().mean().item(),
                "frac_below_0.1": (w < 0.1).float().mean().item(),
            }

        record = {
            "step": step,
            "ood_acc": ood_acc,
            "test_acc": test_acc,
            "mean_reward": rewards.mean().item(),
            **{f"w_{k}": v for k, v in w_stats.items()},
        }
        history.append(record)
        algo = "Nested" if ood_acc > 0.5 else "Equal-Count"
        print(f"Step {step:3d} | ood_acc={ood_acc:.3f} test={test_acc:.3f} "
              f"| w_mean={w_stats['mean']:.3f} w_std={w_stats['std']:.3f} "
              f"| avg_reward={rewards.mean():.3f} | {algo}")

    # Save history
    with open(os.path.join(RESULTS_DIR, "meta_curriculum.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final weights
    torch.save(weight_logits.cpu(), os.path.join(RESULTS_DIR, "data_weights.pt"))

    # ================================================================
    # Evaluate: train multiple seeds with learned curriculum
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Evaluating learned curriculum with {NUM_EVAL_SEEDS} seeds...")
    final_weights = weight_logits  # sigmoid applied inside inner_loop

    eval_results = []
    for seed in range(NUM_EVAL_SEEDS):
        torch.manual_seed(1000 + seed)
        random.seed(1000 + seed)
        model = ParenTransformer(D_MODEL, N_HEADS, N_LAYERS).to(device)
        model = inner_loop(model, train_dataset, final_weights, device)

        with torch.no_grad():
            ood_acc = evaluate_ood_acc(model, ood_examples, device)
            test_acc = evaluate_test_acc(model, test_dataset, device)

        eval_results.append({"seed": seed, "ood_acc": ood_acc, "test_acc": test_acc})
        algo = "Nested" if ood_acc > 0.5 else "Equal-Count"
        print(f"  Seed {seed:2d} | test={test_acc:.3f} ood={ood_acc:.3f} → {algo}")

    with open(os.path.join(RESULTS_DIR, "meta_curriculum_eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    nested_count = sum(1 for r in eval_results if r["ood_acc"] > 0.5)
    print(f"\nWith learned curriculum: {nested_count}/{NUM_EVAL_SEEDS} Nested")

    # ================================================================
    # Plot
    # ================================================================
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # OOD accuracy over outer steps
    axes[0, 0].plot([h["ood_acc"] for h in history], color="#2563eb", linewidth=1.5)
    axes[0, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Outer step")
    axes[0, 0].set_ylabel("OOD accuracy")
    axes[0, 0].set_title("OOD accuracy (high = Nested)")
    axes[0, 0].grid(True, alpha=0.3)

    # Mean reward across ES samples
    axes[0, 1].plot([h["mean_reward"] for h in history], color="#dc2626", linewidth=1.5)
    axes[0, 1].set_xlabel("Outer step")
    axes[0, 1].set_ylabel("Mean reward (OOD acc)")
    axes[0, 1].set_title("ES reward across perturbations")
    axes[0, 1].grid(True, alpha=0.3)

    # Weight distribution evolution
    axes[1, 0].plot([h["w_mean"] for h in history], label="mean", color="#2563eb")
    axes[1, 0].fill_between(
        range(len(history)),
        [h["w_mean"] - h["w_std"] for h in history],
        [h["w_mean"] + h["w_std"] for h in history],
        alpha=0.2, color="#2563eb"
    )
    axes[1, 0].set_xlabel("Outer step")
    axes[1, 0].set_ylabel("Weight (sigmoid)")
    axes[1, 0].set_title("Data weight evolution")
    axes[1, 0].grid(True, alpha=0.3)

    # Comparison: baseline vs meta-learned
    baseline_path = os.path.join(RESULTS_DIR, "baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_oods = [r["ood_acc"] for r in baseline]
        meta_oods = [r["ood_acc"] for r in eval_results]

        axes[1, 1].hist(baseline_oods, bins=15, alpha=0.6, color="#94a3b8", label="Baseline (uniform)")
        axes[1, 1].hist(meta_oods, bins=15, alpha=0.6, color="#2563eb", label="Meta-learned curriculum")
        axes[1, 1].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("OOD accuracy")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Baseline vs meta-learned curriculum")
        axes[1, 1].legend(fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, "Run train_baseline.py first\nfor comparison",
                        ha='center', va='center', fontsize=12, color='gray')
        axes[1, 1].set_title("Baseline comparison (not available)")

    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Meta-learning a data curriculum for Nested algorithm", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "meta_curriculum.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved plots to {RESULTS_DIR}/meta_curriculum.png")


if __name__ == "__main__":
    main()
