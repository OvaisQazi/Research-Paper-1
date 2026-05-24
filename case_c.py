"""
Case C — Entropy-Based (Unpredictable) Seed
=============================================
Each of the 30 runs draws a fresh seed from os.urandom.
The seed is logged to results_case_c.csv so individual runs are
technically re-runnable, but no two consecutive executions of this
script will produce the same seed sequence.

After all 30 runs this script loads results_case_a.csv and
results_case_b.csv (if present) and produces the combined three-way
comparison boxplot for the paper.

Dataset : MNIST (60 000 train / 10 000 test, 10 classes)
Model   : MLP  784 → 256 → 128 → 10

Saves: results_case_c.csv
       loss_curves_case_c.svg
       comparison_boxplot.svg
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
N_RUNS         = 30
N_EPOCHS       = 10
BATCH_SIZE     = 64
LR             = 0.001
DATA_DIR       = "./data"
RESULTS_FILE   = "results_case_c.csv"
PLOT_LOSS_FILE = "loss_curves_case_c.svg"
PLOT_BOX_FILE  = "comparison_boxplot.svg"
CURVES_TO_PLOT = 8       # number of loss curves to overlay in the plot


# ── Entropy seed generator ─────────────────────────────────────────────────────
def get_entropy_seed() -> int:
    """
    Derive a 32-bit seed from the OS entropy pool (hardware events,
    timing noise, etc.).  os.urandom is cryptographically secure and
    non-reproducible by design.  The underlying generator is still a
    PRNG — this is *uncontrolled* seeding, not a hardware RNG in the
    training loop.
    """
    return int.from_bytes(os.urandom(4), byteorder="big")


# ── Seeding helper ─────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Model ──────────────────────────────────────────────────────────────────────
class MnistMLP(nn.Module):
    """Identical architecture to Cases A and B."""
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


# ── Combined comparison boxplot ────────────────────────────────────────────────
def plot_comparison_boxplot() -> None:
    """
    Load CSVs from all three cases and produce a side-by-side boxplot
    with individual data points overlaid.
    Skips any case whose CSV is not found.
    """
    case_files = {
        "A — Fixed\nSeed (42)":     "results_case_a.csv",
        "B — Varied\nFixed Seeds":  "results_case_b.csv",
        "C — Entropy-\nBased Seed": "results_case_c.csv",
    }

    data:   list[np.ndarray] = []
    labels: list[str]        = []

    for label, fpath in case_files.items():
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            data.append(df["accuracy"].values * 100)
            labels.append(label)
        else:
            print(f"  Warning: {fpath} not found — skipping from boxplot.")

    if not data:
        print("  No result files found; skipping comparison boxplot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    colours = ["#4393c3", "#d6604d", "#74c476"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.70)

    # Overlay individual data points (strip-plot effect)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        jitter = rng.uniform(-0.10, 0.10, size=len(d))
        ax.scatter(
            np.full(len(d), i) + jitter, d,
            color="black", alpha=0.35, s=20, zorder=3,
        )

    all_vals = np.concatenate(data)
    ax.set_ylim(all_vals.min() - 1.5, min(all_vals.max() + 1.5, 100))

    ax.set_title(
        "Test Accuracy Distribution — All Three Seeding Conditions\n"
        f"(N = {N_RUNS} runs per case, MNIST dataset)",
        fontsize=11,
    )
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlabel("Seeding Condition")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_BOX_FILE, format="svg")
    plt.close()
    print(f"  Comparison boxplot saved → {PLOT_BOX_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    train_set, test_set = load_data()

    records:     list[dict]        = []
    loss_curves: list[list[float]] = []

    print(f"Case C — Entropy-Based Seed | {N_RUNS} runs")
    print("─" * 50)

    for run in range(1, N_RUNS + 1):
        seed = get_entropy_seed()
        acc, lc = run_experiment(train_set, test_set, seed=seed)
        loss_curves.append(lc)
        records.append({
            "run":      run,
            "seed":     seed,
            "accuracy": acc,
            "case":     "C",
        })
        print(
            f"  Run {run:02d}/{N_RUNS}  "
            f"seed = {seed:>10d}  →  accuracy = {acc:.4f}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    all_accs = [r["accuracy"] for r in records]
    print("\n" + "─" * 50)
    print(f"  Mean accuracy : {np.mean(all_accs):.4f}")
    print(f"  Std deviation : {np.std(all_accs):.6f}")
    print(f"  Min accuracy  : {np.min(all_accs):.4f}")
    print(f"  Max accuracy  : {np.max(all_accs):.4f}")
    print("─" * 50)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Results saved  → {RESULTS_FILE}")
    print(f"  All {N_RUNS} entropy seeds logged in column 'seed'")

    # ── Loss curve plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs  = range(1, N_EPOCHS + 1)
    cmap    = plt.colormaps["tab10"]

    for i in range(min(CURVES_TO_PLOT, N_RUNS)):
        ax.plot(
            epochs, loss_curves[i],
            color=cmap(i / CURVES_TO_PLOT),
            linewidth=1.4,
            alpha=0.80,
            label=f"run {i + 1}",
        )

    ax.set_title(
        f"Case C — Training Loss (first {CURVES_TO_PLOT} of {N_RUNS} "
        "entropy-seeded runs)\n"
        "Spread between curves reflects variance from uncontrolled seeding",
        fontsize=11,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xticks(range(1, N_EPOCHS + 1))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_LOSS_FILE, format="svg")
    plt.close()
    print(f"  Loss curve saved → {PLOT_LOSS_FILE}")

    # ── Three-way comparison boxplot ──────────────────────────────────────────
    print("\nGenerating comparison boxplot across all three cases ...")
    plot_comparison_boxplot()
    print()


if __name__ == "__main__":
    main()