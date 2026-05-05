import torch 
import torch.nn as nn

class MultiOutputMlp(nn.Module) : 
    def __init__(self,trial,input_dim=3500):
        super(MultiOutputMlp,self).__init__()

        # on suggest des hyperparametres en utilisant optuna

        n_layers = trial.suggest_int("n_layers",1,4)
        dropout_rate = trial.suggest_float("dropout_rate",0.1,0.5)
        activation_name = trial.suggest_categorical("activation",["relu","tanh","leaky_relu"])

        activations = {
            "relu" : nn.ReLU(),
            "tanh" : nn.Tanh(),
            "leaky_relu" : nn.LeakyReLU()
        }

        activation = activations[activation_name]

        layers = []
        in_features = input_dim

        for i in range(n_layers):
            out_features = trial.suggest_int(
                f"n_units_l{i}",64,512,step=64
            )

            layers.append(nn.Linear(in_features,out_features))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        self.shared = nn.Sequential(*layers)

        self.head_titre = nn.Linear(in_features,212)
        self.head_parent = nn.Linear(in_features + 212,26)

    def forward(self,x):
        shared_out = self.shared(x)
        
        titre_out = self.head_titre(shared_out)

        parent_input = torch.cat([shared_out,titre_out],dim=1)
        parent_out = self.head_parent(parent_input)
        
        return titre_out,parent_out 
    