import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.utils.class_weight import compute_class_weight
from models.search_space import MultiOutputMlp
from utils.metrics import compute_metrics

x_train = scipy.sparse.load_npz('data/x_train.npz')
x_test = scipy.sparse.load_npz('data/x_test.npz')
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')

y_train_titre = torch.tensor(y_train['Catégorisation Titre'].values , dtype = torch.long)
y_test_titre = torch.tensor(y_test['Catégorisation Titre'].values , dtype = torch.long)
y_train_parent = torch.tensor(y_train['Catégorisation Parent de 1er niveau'].values , dtype = torch.long)
y_test_parent = torch.tensor(y_test['Catégorisation Parent de 1er niveau'].values , dtype = torch.long)

x_train_t = torch.tensor(x_train,dtype=torch.float32)
x_test_t = torch.tensor(x_test,dtype=torch.float32)

classes = np.arrange(212)
weights = compute_class_weight('balanced',classes=classes,y=y_train['Catégorisation Titre'].values)
titre_weights = torch.tensor(weights,dtype=torch.float32)

