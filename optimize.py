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

# on charge les données d'entrainement et de test
x_train = scipy.sparse.load_npz(DATA_DIR / "x_train.npz")
x_test  = scipy.sparse.load_npz(DATA_DIR / "x_test.npz")

y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_test  = pd.read_csv(DATA_DIR / "y_test.csv")

# on convertit les labels en tenseurs PyTorch
y_train_titre  = torch.tensor(y_train["Catégorisation Titre"].values, dtype=torch.long)
y_test_titre   = torch.tensor(y_test["Catégorisation Titre"].values,  dtype=torch.long)
y_train_parent = torch.tensor(y_train["Catégorisation Parent de 1er niveau"].values, dtype=torch.long)
y_test_parent  = torch.tensor(y_test["Catégorisation Parent de 1er niveau"].values,  dtype=torch.long)

# on convertit les features en tenseurs PyTorch
x_train_t = torch.tensor(x_train.toarray(), dtype=torch.float32)
x_test_t  = torch.tensor(x_test.toarray(),  dtype=torch.float32)

# on calcul la dimension de x pour construire le modèle
input_dim = x_train_t.shape[1]
print(f"Input dim : {input_dim}")

# on calcul le nombre de classes pour chaque sortie et les poids
n_titre  = int(y_train["Catégorisation Titre"].max()) + 1
n_parent = int(y_train["Catégorisation Parent de 1er niveau"].max()) + 1

present_classes  = np.unique(y_train["Catégorisation Titre"].values)
computed_weights = compute_class_weight("balanced", classes=present_classes,
                                        y=y_train["Catégorisation Titre"].values)

titre_weights = torch.ones(n_titre, dtype=torch.float32)
for cls, w in zip(present_classes, computed_weights):
    titre_weights[int(cls)] = w

print(f"Titre output size : {n_titre} | Parent output size : {n_parent}")
print(f"Titre unique classes : {len(present_classes)}")


# la fonction d'optimisation
def optimization(trial):
    model = MultiOutputMlp(trial, input_dim=input_dim,
                           n_titre=n_titre, n_parent=n_parent)

    lr         = trial.suggest_float("lr", 5e-4, 8e-3, log=True)  # resserré autour de la zone gagnante
    epochs     = trial.suggest_int("epochs", 20, 50)               # plancher relevé — les courtes durées ne convergent pas
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])  # 128 retiré — trop grand pour les classes rares

    optimizer        = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_titre  = nn.CrossEntropyLoss(weight=titre_weights)
    criterion_parent = nn.CrossEntropyLoss()

    # scheduler : réduit le lr si la perte stagne → aide la convergence en fin d'entrainement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # la boucle d'entrainement
    model.train()
    n = len(x_train_t)

    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
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

            epoch_loss += loss.item()



        # on passe la perte moyenne de l'époque au scheduler
        scheduler.step(epoch_loss / (n // batch_size))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 8:
            break
    # phase d'évaluation
    model.eval()
    with torch.no_grad():
        titre_out, parent_out = model(x_test_t)

        titre_acc  = (titre_out.argmax(1)  == y_test_titre).float().mean().item()
        parent_acc = (parent_out.argmax(1) == y_test_parent).float().mean().item()
        accuracy   = 0.6 * titre_acc + 0.4 * parent_acc

    n_params = compute_metrics(model, input_dim=input_dim)["params"]

    return accuracy, n_params


# Run study
if __name__ == "__main__":
    sampler = optuna.samplers.NSGAIISampler(population_size=10)

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
        study_name="ticket_classifier"
    )

    study.optimize(optimization, n_trials=60, timeout=None)  # 100 = 5 générations NSGA-II

    print(f"\nPareto front : {len(study.best_trials)} trials")
    for trial in study.best_trials:
        acc, params = trial.values
        print(f"  Trial {trial.number} : accuracy={acc:.4f} , params={params:,}")
        print(f"    Hyperparams : {trial.params}")