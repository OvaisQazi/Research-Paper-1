"""
security_test_b.py — Victim Model: Varied Seed Pool
=====================================================
Trains the victim model with a seed randomly chosen from a small pool
[0, 7, 42, 123, 999]. The pool is publicly known (e.g. listed in a paper),
but the specific seed chosen for this run is not disclosed to the attacker.

The attacker knows the pool but must identify the correct seed through
prediction agreement (see attacker.py).

Saves: victim_b.pth          (model weights)
       victim_b_meta.json    (chosen seed + accuracy — hidden from attacker)
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Configuration ──────────────────────────────────────────────────────────────
# Pool is public knowledge — attacker knows these values
SEED_POOL   = [0, 7, 42, 123, 999]
N_EPOCHS    = 10
BATCH_SIZE  = 64
LR          = 0.001
DATA_DIR    = "./data"
MODEL_FILE  = "victim_b.pth"
META_FILE   = "victim_b_meta.json"


# ── Seeding ────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Model ──────────────────────────────────────────────────────────────────────
class MnistMLP(nn.Module):
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


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return train_set, test_set


# ── Training ───────────────────────────────────────────────────────────────────
def train(seed: int, train_set, test_set) -> tuple[MnistMLP, float]:
    set_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, generator=g, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = MnistMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch:02d}/{N_EPOCHS}  loss = {epoch_loss/len(train_loader):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    accuracy = correct / total
    return model, accuracy


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 55)
    print("  Security Test — Case B: Varied Seed Pool Victim")
    print("=" * 55)
    print(f"  Seed pool (public)  : {SEED_POOL}")

    # Pick seed using system randomness — choice is NOT disclosed to attacker
    chosen_seed = random.choice(SEED_POOL)
    print(f"  Chosen seed (hidden): {chosen_seed}\n")

    train_set, test_set = load_data()
    model, accuracy = train(chosen_seed, train_set, test_set)

    print(f"\n  Test accuracy : {accuracy:.4f}")

    # Save model weights
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"  Model saved  → {MODEL_FILE}")

    # Save metadata — seed is logged privately but NOT given to attacker
    meta = {
        "seed":      chosen_seed,
        "seed_pool": SEED_POOL,
        "accuracy":  round(accuracy, 6),
        "note":      "Chosen seed is hidden from attacker — pool is public"
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {META_FILE}")
    print(f"\n  Attacker knows pool {SEED_POOL} but NOT which seed was used.")
    print(f"  Attacker must identify seed via prediction agreement.\n")


if __name__ == "__main__":
    main()