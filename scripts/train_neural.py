"""Two matched GPU runs: classical-proposal hybrid and standalone TCN.

Checkpoint choice uses validation only. Fresh test not loaded by this script.
"""
from pathlib import Path
import os,sys,json,time,argparse,random,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from awrls3.neural import Denoiser


def objective(y,s,is_clean):
    mse=(y-s).square().mean(dim=(1,2))
    # Relative band energy loss, stable at low-energy frequency bins.
    fy=torch.fft.rfft(y,dim=-1);fs=torch.fft.rfft(s,dim=-1)
    band=[]
    freq=torch.fft.rfftfreq(y.shape[-1],1/256).to(y.device)
    for lo,hi in [(0.5,4),(4,8),(8,13),(13,30),(30,40)]:
        mask=(freq>=lo)&(freq<hi)
        py=fy[...,mask].abs().square().mean(dim=(1,2))
        ps=fs[...,mask].abs().square().mean(dim=(1,2))
        band.append((torch.log1p(py)-torch.log1p(ps)).square())
    spectral=torch.stack(band).mean(0)
    return ((mse+.025*spectral)*(1+3*is_clean)).mean()


def run(hybrid,args):
    seed=args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.require_cuda and device!='cuda':raise RuntimeError('CUDA required')
    model=Denoiser(hybrid,32).to(device);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    datasets={};domains={}
    for split in ['train','validation']:
        with np.load(ROOT/f'results/{split}_cache.npz') as z:
            # Entire modest cache on GPU; CPU memory stays bounded.
            datasets[split]=(torch.tensor(z['data'],device=device),torch.tensor(z['is_clean'],device=device,dtype=torch.float32))
            domains[split]=z['domain'].copy()
    name=('hybrid_tcn' if hybrid else 'standalone_tcn')+args.suffix;history=[];best=float('inf');bad=0
    t0=time.perf_counter();params=sum(p.numel() for p in model.parameters())
    for epoch in range(args.epochs):
        model.train();data,flags=datasets['train'];order=torch.randperm(len(data),device=device);losses=[]
        for ids in order.split(args.batch):
            batch=data[ids];x=batch[:,0:1];s=batch[:,1:2];a=batch[:,2:3]
            # Polarity augmentation preserves the physical additive relationship.
            signs=(torch.randint(0,2,(len(ids),1,1),device=device)*2-1).float()
            y,_=model(x*signs,a*signs);loss=objective(y,s*signs,flags[ids])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach()))
        model.eval();domain_scores=[]
        with torch.inference_mode():
            data,flags=datasets['validation']
            for domain in ['public','internal']:
                indices=torch.tensor(np.flatnonzero(domains['validation']==domain),device=device)
                values=[]
                for ids in indices.split(args.batch):
                    b=data[ids];y,_=model(b[:,0:1],b[:,2:3])
                    values.append((float(objective(y,b[:,1:2],flags[ids])),len(b)))
                domain_scores.append(sum(v*n for v,n in values)/sum(n for _,n in values))
        val=float(np.mean(domain_scores))
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=val,elapsed_s=time.perf_counter()-t0))
        if val<best-1e-6:
            best=val;bad=0
            torch.save(dict(state_dict=model.state_dict(),hybrid=hybrid,channels=32,epoch=epoch+1,
                seed=seed,validation_loss=best,parameters=params,torch_version=str(torch.__version__),device=device,
                split_sha256=hashlib.sha256((ROOT/'results/source_split.json').read_bytes()).hexdigest()),ROOT/f'models/{name}.pt')
        else:bad+=1
        (ROOT/f'results/{name}_training.json').write_text(json.dumps(history,indent=2))
        print(name,history[-1],flush=True)
        if bad>=args.patience:break
    print(name,'DONE',params,'parameters',device,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--patience',type=int,default=12);ap.add_argument('--seed',type=int,default=20261028)
    ap.add_argument('--suffix',default='')
    ap.add_argument('--require-cuda',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4)
    for hybrid in [True,False]:run(hybrid,args)
