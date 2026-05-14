import json , torch  
from models.search_space import MultiOutputMlp
import pandas as pd
import numpy as np
import scipy.sparse 
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
# on charge les données d'entrainement et de test
x_train = scipy.sparse.load_npz(DATA_DIR / "x_train.npz")
x_test  = scipy.sparse.load_npz(DATA_DIR / "x_test.npz")

y_train = pd.read_csv(DATA_DIR / "y_train.csv")
y_test  = pd.read_csv(DATA_DIR / "y_test.csv")

params = json.loads((RESULTS_DIR/"best_params.json").read_text())

input_dim = x_train.shape[1]

n_titre  = int(y_train["Catégorisation Titre"].max()) + 1
n_parent = int(y_train["Catégorisation Parent de 1er niveau"].max()) + 1

x_train_t = torch.tensor(x_train.toarray(), dtype=torch.float32)
x_test_t  = torch.tensor(x_test.toarray(),  dtype=torch.float32)

y_train_titre  = torch.tensor(y_train["Catégorisation Titre"].values,                   dtype=torch.long)
y_train_parent = torch.tensor(y_train["Catégorisation Parent de 1er niveau"].values,    dtype=torch.long)
y_test_titre   = torch.tensor(y_test["Catégorisation Titre"].values,                    dtype=torch.long)
y_test_parent  = torch.tensor(y_test["Catégorisation Parent de 1er niveau"].values,     dtype=torch.long)

train_dataset = TensorDataset(x_train_t, y_train_titre, y_train_parent)
train_loader  = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiOutputMlp(params,input_dim,n_titre,n_parent).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr = params["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3)
criterion = nn.CrossEntropyLoss()

best_loss = float("inf")
patience_counter = 0

print(f"n_titre={n_titre}, n_parent={n_parent}")
print(f"y_train titre max={y_train['Catégorisation Titre'].max()}, nunique={y_train['Catégorisation Titre'].nunique()}")



for epoch in range(params["epochs"]):
    model.train()
    epoch_loss = 0

    for x_batch , y_titre_batch , y_parent_batch in train_loader : 
        x_batch = x_batch.to(device)
        y_titre_batch = y_titre_batch.to(device)
        y_parent_batch = y_parent_batch.to(device)

        optimizer.zero_grad()
        out_titre , out_parent = model(x_batch)

        loss = criterion(out_titre, y_titre_batch) + criterion(out_parent, y_parent_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss/len(train_loader)
    scheduler.step(avg_loss)

    if avg_loss < best_loss : 
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(),RESULTS_DIR/"best_model.pth")
    else :
        patience_counter += 1
    
    if patience_counter >= 8 : 
        print(f"early stoping at epoch {epoch + 1}")
        break

    print(f"Epoch {epoch+1}/{params['epochs']} — loss: {avg_loss:.4f}")

model.load_state_dict(torch.load(RESULTS_DIR / "best_model.pth"))
model.eval()

with torch.no_grad():
    out_titre, out_parent = model(x_test_t.to(device))
    pred_titre  = out_titre.argmax(dim=1).cpu()
    pred_parent = out_parent.argmax(dim=1).cpu()

acc_titre  = (pred_titre  == y_test_titre).float().mean().item()
acc_parent = (pred_parent == y_test_parent).float().mean().item()

print(f"\nFinal — Titre accuracy: {acc_titre:.4f} | Parent accuracy: {acc_parent:.4f}")