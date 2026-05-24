"""
Case B — Varied Fixed Seeds
============================
Five distinct seeds [0, 7, 42, 123, 999] × 6 runs each = 30 runs total.
Every run is individually reproducible, but accuracy varies across seeds.
This isolates seed *sensitivity* within PRNG-based training.

Dataset : MNIST (60 000 train / 10 000 test, 10 classes)
Model   : MLP  784 → 256 → 128 → 10

Saves: results_case_b.csv, loss_curves_case_b.svg
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
SEEDS         = [0, 7, 42, 123, 999]
RUNS_PER_SEED = 6                      # 5 seeds × 6 runs = 30 total
N_EPOCHS      = 10
BATCH_SIZE    = 64
LR            = 0.001
DATA_DIR      = "./data"
RESULTS_FILE  = "results_case_b.csv"
PLOT_FILE     = "loss_curves_case_b.svg"

COLOURS = {
    0:   "#e41a1c",
    7:   "#377eb8",
    42:  "#4daf4a",
    123: "#984ea3",
    999: "#ff7f00",
}


# ── Seeding helper ─────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Model ──────────────────────────────────────────────────────────────────────
class MnistMLP(nn.Module):
    """Identical architecture to Cases A and C — only the seed varies."""
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
    set_seed(seed)

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

    for _ in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_curve.append(epoch_loss / len(train_loader))

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

    records:          list[dict]             = []
    seed_loss_curves: dict[int, list[float]] = {}

    total_runs = len(SEEDS) * RUNS_PER_SEED
    run_count  = 0

    print(f"Case B — Varied Fixed Seeds | {total_runs} runs")
    print("─" * 50)

    for seed in SEEDS:
        seed_accuracies: list[float] = []
        print(f"\n  Seed = {seed}")

        for rep in range(1, RUNS_PER_SEED + 1):
            run_count += 1
            acc, lc = run_experiment(train_set, test_set, seed=seed)
            seed_accuracies.append(acc)
            records.append({
                "run":      run_count,
                "seed":     seed,
                "rep":      rep,
                "accuracy": acc,
                "case":     "B",
            })
            print(f"    Rep {rep}/{RUNS_PER_SEED}  →  accuracy = {acc:.4f}")

            if rep == 1:
                seed_loss_curves[seed] = lc

        print(
            f"    └─ seed {seed:>3d}: "
            f"mean = {np.mean(seed_accuracies):.4f}  "
            f"std  = {np.std(seed_accuracies):.6f}"
        )

    # ── Overall summary ───────────────────────────────────────────────────────
    all_accs = [r["accuracy"] for r in records]
    print("\n" + "─" * 50)
    print(f"  Overall mean accuracy : {np.mean(all_accs):.4f}")
    print(f"  Overall std deviation : {np.std(all_accs):.6f}")
    print(f"  Min accuracy          : {np.min(all_accs):.4f}")
    print(f"  Max accuracy          : {np.max(all_accs):.4f}")
    print("─" * 50)

    # ── Per-seed breakdown ────────────────────────────────────────────────────
    df_all = pd.DataFrame(records)
    print("\n  Per-seed breakdown:")
    for seed in SEEDS:
        s = df_all[df_all["seed"] == seed]["accuracy"]
        print(f"    seed {seed:>3d}: mean = {s.mean():.4f}  std = {s.std():.6f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_all.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Results saved  → {RESULTS_FILE}")

    # ── Loss curve plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs  = range(1, N_EPOCHS + 1)

    for seed, lc in seed_loss_curves.items():
        ax.plot(
            epochs, lc,
            color=COLOURS[seed],
            linewidth=1.8,
            label=f"seed = {seed}",
        )

    ax.set_title(
        "Case B — Training Loss per Seed\n"
        "(one representative run per seed; spread shows seed sensitivity)",
        fontsize=11,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xticks(range(1, N_EPOCHS + 1))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, format="svg")
    plt.close()
    print(f"  Loss curve saved → {PLOT_FILE}\n")


if __name__ == "__main__":
    main()