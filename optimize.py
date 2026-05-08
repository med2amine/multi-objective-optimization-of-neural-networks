import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.utils.class_weight import compute_class_weight
from models.search_space import MultiOutputMlp
from utils.metrics import compute_metrics

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ── Load data ────────────────────────────────────────────────────────────────
x_train = scipy.sparse.load_npz(DATA_DIR / "x_train.npz")
x_test  = scipy.sparse.load_npz(DATA_DIR / "x_test.npz")

y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_test  = pd.read_csv(DATA_DIR / "y_test.csv")


# ── Labels ───────────────────────────────────────────────────────────────────
y_train_titre  = torch.tensor(y_train["Catégorisation Titre"].values, dtype=torch.long)
y_test_titre   = torch.tensor(y_test["Catégorisation Titre"].values,  dtype=torch.long)
y_train_parent = torch.tensor(y_train["Catégorisation Parent de 1er niveau"].values, dtype=torch.long)
y_test_parent  = torch.tensor(y_test["Catégorisation Parent de 1er niveau"].values,  dtype=torch.long)

# ── Dense tensors ────────────────────────────────────────────────────────────
x_train_t = torch.tensor(x_train.toarray(), dtype=torch.float32)
x_test_t  = torch.tensor(x_test.toarray(),  dtype=torch.float32)

input_dim = x_train_t.shape[1]
print(f"Input dim : {input_dim}")

n_titre  = int(y_train["Catégorisation Titre"].max()) + 1
n_parent = int(y_train["Catégorisation Parent de 1er niveau"].max()) + 1

present_classes  = np.unique(y_train["Catégorisation Titre"].values)
computed_weights = compute_class_weight("balanced", classes=present_classes,
                                        y=y_train["Catégorisation Titre"].values)

# Size 212 — one slot per output neuron, default 1.0 for missing classes
titre_weights = torch.ones(n_titre, dtype=torch.float32)
for cls, w in zip(present_classes, computed_weights):
    titre_weights[int(cls)] = w

print(f"Titre output size : {n_titre} | Parent output size : {n_parent}")
print(f"Titre unique classes : {len(present_classes)}")


# ── Objective function ───────────────────────────────────────────────────────
def optimization(trial):
    model = MultiOutputMlp(trial, input_dim=input_dim,
                           n_titre=n_titre, n_parent=n_parent)

    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs     = trial.suggest_int("epochs", 30, 70)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    optimizer        = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_titre  = nn.CrossEntropyLoss(weight=titre_weights)
    criterion_parent = nn.CrossEntropyLoss()

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    n = len(x_train_t)

    for epoch in range(epochs):
        indices = torch.randperm(n)

        for start in range(0, n, batch_size):
            idx  = indices[start:start + batch_size]
            xb   = x_train_t[idx]
            yb_t = y_train_titre[idx]
            yb_p = y_train_parent[idx]

            titre_out, parent_out = model(xb)
            loss = criterion_titre(titre_out, yb_t) + criterion_parent(parent_out, yb_p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        titre_out, parent_out = model(x_test_t)

        titre_acc  = (titre_out.argmax(1)  == y_test_titre).float().mean().item()
        parent_acc = (parent_out.argmax(1) == y_test_parent).float().mean().item()
        accuracy   = 0.6 * titre_acc + 0.4 * parent_acc

    n_params = compute_metrics(model, input_dim=input_dim)["params"]

    return accuracy, n_params


# ── Run study ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sampler = optuna.samplers.NSGAIISampler(population_size=20)  # stable, not experimental

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
        study_name="ticket_classifier"
    )

    study.optimize(optimization, n_trials=60, timeout=None)

    print(f"\nPareto front : {len(study.best_trials)} trials")
    for trial in study.best_trials:
        acc, params = trial.values
        print(f"  Trial {trial.number} : accuracy={acc:.4f} , params={params:,}")
        print(f"    Hyperparams : {trial.params}")