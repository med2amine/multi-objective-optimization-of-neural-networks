import torch 
import torch.nn as nn

class MultiOutputMlp(nn.Module): 
    def __init__(self, trial, input_dim, n_titre, n_parent):
        super(MultiOutputMlp, self).__init__()

        # on suggère des hyperparamètres en utilisant optuna
        n_layers        = trial.suggest_int("n_layers", 1, 4)
        dropout_rate    = trial.suggest_float("dropout_rate", 0.1, 0.5)
        activation_name = trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu"])

        activations = {
            "relu"       : nn.ReLU(),
            "tanh"       : nn.Tanh(),
            "leaky_relu" : nn.LeakyReLU()
        }
        activation = activations[activation_name]

        layers = []
        in_features = input_dim

        for i in range(n_layers): 
            # cette boucle va créer dynamiquement le corps de notre réseau de neurones
            # à chaque itération :
            # - le nombre de neurones est choisi par optuna entre 64 et 512
            # - une couche linéaire est ajoutée
            # - une fonction d'activation est appliquée
            # - puis un dropout est utilisé pour la régularisation
            # la sortie de chaque couche devient l'entrée de la suivante
            out_features = trial.suggest_int(f"n_units_l{i}", 64, 512, step=64)

            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        self.shared = nn.Sequential(*layers)

        # têtes de classification — tailles lues depuis les données, jamais hardcodées
        self.head_titre  = nn.Linear(in_features, n_titre)
        self.head_parent = nn.Linear(in_features + n_titre, n_parent)

    def forward(self, x):
        # les données d'entrée sont traitées par les couches partagées pour extraire une représentation commune
        shared_out = self.shared(x)

        # la première tête prédit la sortie "titre"
        titre_out = self.head_titre(shared_out)

        # la deuxième tête prédit "parent" en utilisant la représentation partagée + sortie titre
        parent_input = torch.cat([shared_out, titre_out], dim=1)
        parent_out   = self.head_parent(parent_input)

        return titre_out, parent_out
    