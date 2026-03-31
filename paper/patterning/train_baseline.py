"""
Train baseline: show bimodal distribution of learned algorithms.

Train multiple models with different random seeds and measure OOD accuracy.
High OOD accuracy = Nested algorithm. Low OOD accuracy = Equal-Count algorithm.

Usage:
    python3 train_baseline.py
"""

import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import ParenTransformer, tokenize_sequence

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

NUM_SEEDS = 20
EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.001
MAX_LEN = 42
MAX_TRAIN = None  # Set to e.g. 10000 for fast CPU runs


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


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


def evaluate_ood(model, ood_path, device):
    """Evaluate on OOD data. For Nested: label=FALSE. For Equal-Count: label=TRUE."""
    model.eval()
    # OOD data has label_nested=False, label_equal_count=True
    # If model predicts FALSE → it's Nested. If TRUE → it's Equal-Count.
    examples = [json.loads(l) for l in open(ood_path)]
    nested_correct = 0  # predicts FALSE (nested answer)
    total = 0
    with torch.no_grad():
        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i+BATCH_SIZE]
            ids = torch.tensor([tokenize_sequence(ex["seq"], MAX_LEN) for ex in batch],
                               dtype=torch.long).to(device)
            logits = model(ids)
            preds = logits.argmax(dim=1)
            # Nested algorithm would predict FALSE (0) for these
            nested_correct += (preds == 0).sum().item()
            total += len(batch)
    return nested_correct / total  # high = Nested, low = Equal-Count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ParenDataset(os.path.join(DATA_DIR, "train.jsonl"))
    if MAX_TRAIN:
        train_dataset.examples = train_dataset.examples[:MAX_TRAIN]
        print(f"Using {len(train_dataset)} training examples (truncated)")
    test_dataset = ParenDataset(os.path.join(DATA_DIR, "test.jsonl"))
    ood_path = os.path.join(DATA_DIR, "ood_test.jsonl")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        random.seed(seed)

        model = ParenTransformer(d_model=64, n_heads=4, n_layers=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        ood_acc = evaluate_ood(model, ood_path, device)

        results.append({"seed": seed, "test_acc": test_acc, "ood_acc": ood_acc})
        algo = "Nested" if ood_acc > 0.5 else "Equal-Count"
        print(f"Seed {seed:2d} | test={test_acc:.3f} ood={ood_acc:.3f} → {algo}")

    # Save and plot
    with open(os.path.join(RESULTS_DIR, "baseline.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    ood_accs = [r["ood_acc"] for r in results]
    nested = sum(1 for a in ood_accs if a > 0.5)
    equal = sum(1 for a in ood_accs if a <= 0.5)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ood_accs, bins=20, color="#2563eb", edgecolor="white", alpha=0.8)
    ax.set_xlabel("OOD accuracy (high = Nested, low = Equal-Count)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Bimodal distribution: {nested} Nested, {equal} Equal-Count (n={NUM_SEEDS})", fontsize=13)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "baseline_distribution.png"), dpi=150)
    print(f"\nSaved results. {nested} Nested, {equal} Equal-Count.")


if __name__ == "__main__":
    main()
