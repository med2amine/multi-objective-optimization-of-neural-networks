import torch
import time 
from thop import profile

def compute_metrics(model , input_dim = 3500):
    model.eval() # elle me le model en mode evaluation pour eviter de calculer les gradients
    dummy_input = torch.randn(1,input_dim) # on crée un tensore de taille (1,input_dim) rempli de valeurs aléatoires pour simuler une entrée

    total_params = sum(p.numel() for p in model.parameters()) # on calcule le nombre total de paramétres 

    flops,_ = profile(model,inputs=(dummy_input,),verbose = False) # on utilise la fonction profile de thop pour calculer les FLOPS du modèle

    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            model(dummy_input)
            times.append(time.time()-start)
        latency = sum(times)/len(times) * 1000 # on calcule la latence moyenne en millisecondes 

        return {
            "params" : total_params,
            "flops" : flops,
            "latency_ms" : round(latency,4)
        }
    
    