import torch
import time 
from thop import profile

def compute_metrics(model , input_dim = 3500):
    model.eval()
    dummy_input = torch.randn(&,input_dim)

    total_params = sum(p.numel() for p in model.parameters())

    flops,_ = profile(model,inputs=(dummy_input,),verbose = False)

    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            model(dummy_input)
            times.append(time.time()-start)
        latency = sum(times)/len(times) * 1000

        return {
            "params" : total_params,
            "flops" : flops,
            "latency_ms" : round(latency,4)
        }
    
    