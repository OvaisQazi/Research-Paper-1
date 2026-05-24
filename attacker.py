"""
attacker.py — Adversarial Attack on All Three Victim Models
============================================================
Simulates an attacker who:

  Case A: Knows seed = 42 (published) → reconstructs model exactly
          → white-box FGSM attack

  Case B: Knows seed pool [0,7,42,123,999] → trains all 5 candidates
          → identifies correct seed via prediction agreement
          → white-box FGSM attack on matched model

  Case C: Knows only that entropy seeding was used → trains surrogate
          with arbitrary seed → transfer FGSM attack (black-box)

Attack method: FGSM (Fast Gradient Sign Method)
  x_adv = x + ε · sign( ∇ₓ L(attacker_model, x, true_label) )

Metric: Attack Success Rate (ASR)
  ASR = correctly_fooled / originally_correct
  (only counts examples the victim classified correctly before the attack)

Saves: security_results.csv
       attack_success_rates.svg
       adversarial_examples.svg
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ──────────────────────────────────────────────────────────────
SEED_A           = 42                    # Case A: known fixed seed
SEED_POOL_B      = [0, 7, 42, 123, 999] # Case B: known pool, chosen seed unknown
SURROGATE_SEED_C = 42                    # Case C: arbitrary surrogate seed

N_EPOCHS         = 10
BATCH_SIZE       = 64
LR               = 0.001
DATA_DIR         = "./data"
PROBE_SIZE       = 200      # number of images used for seed identification (Case B)
EPSILONS         = [0.05, 0.10, 0.20, 0.30]

VICTIM_A_FILE    = "victim_a.pth"
VICTIM_B_FILE    = "victim_b.pth"
VICTIM_C_FILE    = "victim_c.pth"
RESULTS_FILE     = "security_results.csv"
PLOT_ASR_FILE    = "attack_success_rates.svg"
PLOT_ADV_FILE    = "adversarial_examples.svg"


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
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return test_set


def load_train_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    return train_set


# ── Load saved victim model ────────────────────────────────────────────────────
def load_victim(filepath: str) -> MnistMLP:
    model = MnistMLP()
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model


# ── Train a candidate/surrogate model ─────────────────────────────────────────
def train_model(seed: int, train_set, label: str = "") -> MnistMLP:
    """Train MnistMLP with given seed. Returns trained model in eval mode."""
    tag = f"  [{label}] " if label else "  "
    print(f"{tag}Training with seed = {seed} ...")

    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, generator=g, num_workers=0)

    model     = MnistMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"{tag}  epoch {epoch:02d}/{N_EPOCHS}  loss = {epoch_loss/len(train_loader):.4f}")

    model.eval()
    return model


# ── Get predictions from a model ───────────────────────────────────────────────
def get_predictions(model: MnistMLP, loader: DataLoader) -> torch.Tensor:
    """Return predicted class indices for all examples in loader."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            all_preds.append(model(images).argmax(dim=1))
    return torch.cat(all_preds)


# ── Case B: identify seed via prediction agreement ────────────────────────────
def identify_seed(
    victim:      MnistMLP,
    seed_pool:   list[int],
    probe_loader: DataLoader,
    train_set,
) -> tuple[int, MnistMLP, dict[int, float]]:
    """
    Train one candidate per seed in the pool.
    Compare each candidate's predictions to the victim's on the probe set.
    Return the best-matching seed, its model, and the full agreement table.

    Two models trained with the same seed will agree on 100% of inputs.
    Different seeds produce ~93–97% agreement — gap is unambiguous.
    """
    print("\n  Querying victim on probe set ...")
    victim_preds = get_predictions(victim, probe_loader)

    agreements:        dict[int, float]    = {}
    candidate_models:  dict[int, MnistMLP] = {}

    print(f"\n  Testing {len(seed_pool)} candidate seeds:")
    print("  " + "─" * 40)

    for seed in seed_pool:
        candidate = train_model(seed, train_set, label=f"candidate seed={seed}")
        cand_preds = get_predictions(candidate, probe_loader)
        agreement  = (victim_preds == cand_preds).float().mean().item()
        agreements[seed]       = agreement
        candidate_models[seed] = candidate
        flag = "  ← MATCH" if agreement > 0.999 else ""
        print(f"    seed {seed:>4d}: agreement = {agreement:.4f}{flag}")

    best_seed = max(agreements, key=agreements.__getitem__)
    print(f"\n  Identified seed: {best_seed}  "
          f"(agreement = {agreements[best_seed]:.4f})")

    return best_seed, candidate_models[best_seed], agreements


# ── FGSM adversarial example generation ───────────────────────────────────────
def fgsm_attack(
    model:   MnistMLP,
    images:  torch.Tensor,
    labels:  torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method:
        x_adv = x + ε · sign( ∇ₓ L(model, x, y) )

    Gradients are computed on `model` (the attacker's model).
    Adversarial examples are then tested on the victim.
    Images are clamped to [-1, 1] to stay in the valid normalised range.
    """
    criterion   = nn.CrossEntropyLoss()
    images_adv  = images.clone().detach().requires_grad_(True)

    output = model(images_adv)
    loss   = criterion(output, labels)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        perturbed = images_adv + epsilon * images_adv.grad.sign()
        perturbed = torch.clamp(perturbed, -1.0, 1.0)

    return perturbed.detach()


# ── Attack evaluation ──────────────────────────────────────────────────────────
def evaluate_attack(
    victim:          MnistMLP,
    attacker_model:  MnistMLP,
    test_loader:     DataLoader,
    epsilon:         float,
) -> float:
    """
    Generate adversarial examples using attacker_model (FGSM),
    then test them on the victim model.

    Attack Success Rate (ASR):
        ASR = examples that were correct → now wrong
              ─────────────────────────────────────
              examples that were originally correct

    Only originally-correct examples are counted to isolate the
    attack's contribution from the model's baseline error rate.
    """
    victim.eval()
    attacker_model.eval()

    total_correct = 0
    total_fooled  = 0

    for images, labels in test_loader:
        # Step 1: find which examples victim classifies correctly
        with torch.no_grad():
            orig_preds   = victim(images).argmax(dim=1)
        correct_mask = (orig_preds == labels)

        if correct_mask.sum() == 0:
            continue

        # Step 2: generate adversarial examples on attacker's model
        adv_images = fgsm_attack(attacker_model, images, labels, epsilon)

        # Step 3: test adversarial examples on victim
        with torch.no_grad():
            adv_preds = victim(adv_images).argmax(dim=1)

        # Step 4: count examples that were correct → now wrong
        fooled = correct_mask & (adv_preds != labels)

        total_correct += correct_mask.sum().item()
        total_fooled  += fooled.sum().item()

    return total_fooled / total_correct if total_correct > 0 else 0.0


# ── Adversarial example visualisation ─────────────────────────────────────────
def plot_adversarial_examples(
    victim:         MnistMLP,
    attacker_model: MnistMLP,
    test_set,
    epsilon:        float = 0.20,
    n_examples:     int   = 8,
) -> None:
    """
    Show a grid of original vs adversarial images for Case A (white-box).
    Displays: original image | adversarial image | difference (×5)
    """
    attacker_model.eval()
    victim.eval()

    # Collect n_examples that victim correctly classifies originally
    examples, orig_labels, adv_images_list, adv_preds_list = [], [], [], []

    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    for images, labels in loader:
        with torch.no_grad():
            pred = victim(images).argmax(dim=1)
        if pred.item() != labels.item():
            continue   # skip already-wrong examples

        adv = fgsm_attack(attacker_model, images, labels, epsilon)
        with torch.no_grad():
            adv_pred = victim(adv).argmax(dim=1)

        examples.append(images.squeeze())
        orig_labels.append(labels.item())
        adv_images_list.append(adv.squeeze())
        adv_preds_list.append(adv_pred.item())

        if len(examples) >= n_examples:
            break

    # Plot: 3 rows (original / adversarial / difference) × n_examples cols
    fig = plt.figure(figsize=(n_examples * 1.6, 5.5))
    gs  = gridspec.GridSpec(3, n_examples, hspace=0.05, wspace=0.05)
    row_labels = ["Original", f"Adversarial\n(ε={epsilon})", "Difference\n(×5)"]

    for col in range(n_examples):
        orig = examples[col].numpy()
        adv  = adv_images_list[col].numpy()
        diff = np.abs(adv - orig) * 5   # amplify for visibility

        for row, img in enumerate([orig, adv, diff]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9,
                              rotation=0, labelpad=55, va="center")
            if row == 0:
                ax.set_title(f"True: {orig_labels[col]}", fontsize=8)
            if row == 1:
                colour = "red" if adv_preds_list[col] != orig_labels[col] else "green"
                ax.set_title(f"Pred: {adv_preds_list[col]}",
                             fontsize=8, color=colour)

    fig.suptitle(
        f"Case A — White-Box FGSM Attack (ε = {epsilon})\n"
        "Red prediction = successfully fooled | Green = attack failed",
        fontsize=11, y=1.01,
    )
    plt.savefig(PLOT_ADV_FILE, format="svg", bbox_inches="tight")
    plt.close()
    print(f"\n  Adversarial examples saved → {PLOT_ADV_FILE}")


# ── ASR line plot ──────────────────────────────────────────────────────────────
def plot_asr(results: list[dict]) -> None:
    """
    Line plot of Attack Success Rate vs epsilon for all three cases.
    Case A and B (white-box / enumerated) should sit well above Case C (transfer).
    """
    df = pd.DataFrame(results)

    case_styles = {
        "A — Fixed Seed\n(white-box)":       {"color": "#d62728", "marker": "o", "ls": "-"},
        "B — Varied Pool\n(enumerated)":     {"color": "#ff7f0e", "marker": "s", "ls": "--"},
        "C — Entropy Seed\n(transfer)":      {"color": "#2ca02c", "marker": "^", "ls": "-."},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for case_label, style in case_styles.items():
        subset = df[df["case"] == case_label]
        ax.plot(
            subset["epsilon"],
            subset["asr"] * 100,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=2.0,
            markersize=7,
            label=case_label,
        )

    ax.set_title(
        "Attack Success Rate vs Perturbation Strength (ε)\n"
        "White-box attacks (A, B) vs Transfer attack (C) — MNIST MLP",
        fontsize=11,
    )
    ax.set_xlabel("Epsilon (ε) — perturbation magnitude")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xticks(EPSILONS)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_ASR_FILE, format="svg")
    plt.close()
    print(f"  ASR plot saved → {PLOT_ASR_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Attacker — Adversarial Attack on All Three Victims")
    print("=" * 60)

    # Verify victim files exist
    for f in [VICTIM_A_FILE, VICTIM_B_FILE, VICTIM_C_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"{f} not found. Run the corresponding security_test_*.py first."
            )

    test_set  = load_test_data()
    train_set = load_train_data()

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # Probe loader — small fixed subset used for seed identification (Case B)
    probe_indices = list(range(PROBE_SIZE))
    probe_loader  = DataLoader(
        Subset(test_set, probe_indices),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    results: list[dict] = []

    # ══════════════════════════════════════════════════════════════════════════
    # CASE A — Fixed seed: attacker reconstructs model exactly
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  CASE A — Fixed Seed Attack (White-Box)")
    print("─" * 60)
    print(f"  Attacker knows seed = {SEED_A}")
    print("  Training exact reconstruction ...")

    victim_a    = load_victim(VICTIM_A_FILE)
    candidate_a = train_model(SEED_A, train_set, label="Case A reconstruction")

    # Verify reconstruction is exact
    probe_v = get_predictions(victim_a,    probe_loader)
    probe_c = get_predictions(candidate_a, probe_loader)
    agreement_a = (probe_v == probe_c).float().mean().item()
    print(f"\n  Reconstruction agreement : {agreement_a:.4f}  "
          f"({'exact ✓' if agreement_a > 0.999 else 'mismatch ✗'})")

    print(f"\n  Running FGSM across ε = {EPSILONS} ...")
    for eps in EPSILONS:
        asr = evaluate_attack(victim_a, candidate_a, test_loader, eps)
        print(f"    ε = {eps:.2f}  →  ASR = {asr*100:.1f}%")
        results.append({
            "case":          "A — Fixed Seed\n(white-box)",
            "attack_type":   "white-box",
            "epsilon":       eps,
            "asr":           asr,
            "agreement":     agreement_a,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # CASE B — Varied pool: attacker enumerates and identifies seed
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  CASE B — Varied Pool Attack (Enumerated → White-Box)")
    print("─" * 60)
    print(f"  Attacker knows pool = {SEED_POOL_B}")
    print("  Chosen seed is unknown — identifying via prediction agreement ...")

    victim_b = load_victim(VICTIM_B_FILE)
    identified_seed, matched_model, agreements_b = identify_seed(
        victim_b, SEED_POOL_B, probe_loader, train_set
    )
    agreement_b = agreements_b[identified_seed]

    print(f"\n  Running FGSM across ε = {EPSILONS} ...")
    for eps in EPSILONS:
        asr = evaluate_attack(victim_b, matched_model, test_loader, eps)
        print(f"    ε = {eps:.2f}  →  ASR = {asr*100:.1f}%")
        results.append({
            "case":          "B — Varied Pool\n(enumerated)",
            "attack_type":   "enumerated → white-box",
            "epsilon":       eps,
            "asr":           asr,
            "agreement":     agreement_b,
            "identified_seed": identified_seed,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # CASE C — Entropy seed: attacker uses surrogate (transfer attack)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  CASE C — Entropy Seed Attack (Transfer / Black-Box)")
    print("─" * 60)
    print(f"  Attacker cannot reconstruct victim — training surrogate "
          f"with seed = {SURROGATE_SEED_C} ...")

    victim_c  = load_victim(VICTIM_C_FILE)
    surrogate = train_model(SURROGATE_SEED_C, train_set, label="Case C surrogate")

    # Measure agreement to confirm mismatch
    probe_v = get_predictions(victim_c,  probe_loader)
    probe_s = get_predictions(surrogate, probe_loader)
    agreement_c = (probe_v == probe_s).float().mean().item()
    print(f"\n  Surrogate agreement with victim : {agreement_c:.4f}  "
          f"(expected ~0.93–0.97, not 1.0)")

    print(f"\n  Running FGSM across ε = {EPSILONS} ...")
    for eps in EPSILONS:
        asr = evaluate_attack(victim_c, surrogate, test_loader, eps)
        print(f"    ε = {eps:.2f}  →  ASR = {asr*100:.1f}%")
        results.append({
            "case":          "C — Entropy Seed\n(transfer)",
            "attack_type":   "transfer (black-box)",
            "epsilon":       eps,
            "asr":           asr,
            "agreement":     agreement_c,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SUMMARY — Attack Success Rate (%)")
    print("=" * 60)
    df = pd.DataFrame(results)
    pivot = df.pivot_table(index="epsilon", columns="attack_type",
                           values="asr", aggfunc="mean") * 100
    print(pivot.to_string(float_format="{:.1f}".format))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_out = df[["case", "attack_type", "epsilon", "asr", "agreement"]].copy()
    df_out["asr_pct"] = (df_out["asr"] * 100).round(2)
    df_out.to_csv(RESULTS_FILE, index=False)
    print(f"\n  Results saved  → {RESULTS_FILE}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_asr(results)

    print("\n  Generating adversarial example visualisation (Case A, ε=0.20) ...")
    plot_adversarial_examples(victim_a, candidate_a, test_set, epsilon=0.20)

    print("\n  Attack complete.\n")


if __name__ == "__main__":
    main()