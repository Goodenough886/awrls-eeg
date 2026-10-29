"""Two additional fixed seeds; original seed is retained, never overwritten."""
from types import SimpleNamespace
import torch
from train_neural import run

if __name__=='__main__':
    torch.set_num_threads(4)
    for seed in [20261029,20261030]:
        args=SimpleNamespace(seed=seed,epochs=60,batch=128,patience=12,require_cuda=True,suffix=f'_seed{seed}')
        for hybrid in [True,False]:run(hybrid,args)
