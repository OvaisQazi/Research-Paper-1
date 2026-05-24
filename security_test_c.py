"""
security_test_c.py — Victim Model: Entropy-Based Seed
======================================================
Trains the victim model with a seed drawn from os.urandom.
The seed value is logged privately but is never disclosed.
The attacker knows neither the seed nor the seed space — only
that entropy-based seeding was used.

This is the only condition under which the reconstruction attack
is computationally infeasible (2^32 possible seed values, no structure).

Saves: victim_c.pth          (model weights)
       victim_c_meta.json    (entropy seed + accuracy — private, not given to attacker)
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
N_EPOCHS    = 10
BATCH_SIZE  = 64
LR          = 0.001
DATA_DIR    = "./data"
MODEL_FILE  = "victim_c.pth"
META_FILE   = "victim_c_meta.json"


# ── Entropy seed generator ─────────────────────────────────────────────────────
def get_entropy_seed() -> int:
    """
    Derive a 32-bit seed from the OS entropy pool.
    Non-reproducible by design — attacker cannot predict or enumerate
    this value without access to the private metadata file.
    """
    return int.from_bytes(os.urandom(4), byteorder="big")


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
    print("  Security Test — Case C: Entropy-Based Seed Victim")
    print("=" * 55)

    entropy_seed = get_entropy_seed()
    print(f"  Entropy seed (private) : {entropy_seed}")
    print(f"  Seed space             : 2^32 ≈ 4.3 billion values")
    print(f"  Attacker knowledge     : seeding method only, not the value\n")

    train_set, test_set = load_data()
    model, accuracy = train(entropy_seed, train_set, test_set)

    print(f"\n  Test accuracy : {accuracy:.4f}")

    # Save model weights
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"  Model saved  → {MODEL_FILE}")

    # Seed is logged privately — never disclosed publicly
    meta = {
        "seed":     entropy_seed,
        "accuracy": round(accuracy, 6),
        "note":     "Entropy seed — private log only, never disclosed to attacker"
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {META_FILE}  (private — not given to attacker)")
    print(f"\n  Attacker cannot reconstruct this model.")
    print(f"  Brute-force would require ~{2**32 / 1e9:.1f}B training runs.\n")


if __name__ == "__main__":
    main()