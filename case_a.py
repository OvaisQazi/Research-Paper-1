"""
Case A — Fixed Seed
====================
All 30 runs use the same seed (42).
Expected result: zero variance across runs (perfect reproducibility).

Dataset : MNIST (60 000 train / 10 000 test, 10 classes)
Model   : MLP  784 → 256 → 128 → 10

Saves: results_case_a.csv, loss_curves_case_a.svg
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
FIXED_SEED   = 42
N_RUNS       = 30
N_EPOCHS     = 10
BATCH_SIZE   = 64
LR           = 0.001
DATA_DIR     = "./data"
RESULTS_FILE = "results_case_a.csv"
PLOT_FILE    = "loss_curves_case_a.svg"


# ── Seeding helper ─────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Seed every RNG used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Model ──────────────────────────────────────────────────────────────────────
class MnistMLP(nn.Module):
    """
    Fully-connected network for MNIST digit classification.
    Input: flattened 28×28 = 784 pixels.
    Architecture: Flatten → Linear(784→256) → ReLU → Dropout(0.3)
                  → Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→10)
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data() -> tuple:
    """
    Download (first run only) and return MNIST train/test datasets.
    Images are normalised to [-1, 1].
    """
    print("Loading MNIST dataset ...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST(
        root=DATA_DIR, train=True,  download=True, transform=transform
    )
    test_set  = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    print(f"  Train : {len(train_set):,} samples")
    print(f"  Test  : {len(test_set):,} samples\n")
    return train_set, test_set


# ── Single training run ────────────────────────────────────────────────────────
def run_experiment(
    train_set,
    test_set,
    seed: int,
) -> tuple[float, list[float]]:
    """
    Train MnistMLP with the given seed.
    Returns (test_accuracy, per-epoch loss curve).
    The seed controls weight initialisation, dropout masks, and
    DataLoader shuffling order.
    """
    set_seed(seed)

    # Seeded generator keeps DataLoader shuffling reproducible
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model     = MnistMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_curve: list[float] = []

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_curve.append(epoch_loss / len(train_loader))

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total, loss_curve


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    train_set, test_set = load_data()

    accuracies:  list[float]       = []
    loss_curves: list[list[float]] = []

    print(f"Case A — Fixed Seed ({FIXED_SEED}) | {N_RUNS} runs")
    print("─" * 50)

    for run in range(1, N_RUNS + 1):
        acc, lc = run_experiment(train_set, test_set, seed=FIXED_SEED)
        accuracies.append(acc)
        loss_curves.append(lc)
        print(f"  Run {run:02d}/{N_RUNS}  →  accuracy = {acc:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_acc = np.mean(accuracies)
    std_acc  = np.std(accuracies)
    print("\n" + "─" * 50)
    print(f"  Mean accuracy : {mean_acc:.4f}")
    print(f"  Std deviation : {std_acc:.6f}   (expected ≈ 0.000000)")
    print("─" * 50)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "run":      range(1, N_RUNS + 1),
        "seed":     [FIXED_SEED] * N_RUNS,
        "accuracy": accuracies,
        "case":     ["A"] * N_RUNS,
    })
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Results saved  → {RESULTS_FILE}")

    # ── Loss curve plot ───────────────────────────────────────────────────────
    # All 30 curves are identical; plotting the first is sufficient.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        range(1, N_EPOCHS + 1), loss_curves[0],
        color="steelblue", linewidth=2.0,
        label=f"seed = {FIXED_SEED} (all {N_RUNS} runs identical)",
    )
    ax.set_title(
        "Case A — Training Loss (Fixed Seed 42)\n"
        f"All {N_RUNS} runs produce the same curve — zero variance",
        fontsize=11,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xticks(range(1, N_EPOCHS + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    plt.close()
    print(f"  Loss curve saved → {PLOT_FILE}\n")


if __name__ == "__main__":
    main()