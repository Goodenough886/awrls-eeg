"""Evaluate frozen candidates. Fresh test invoked once after validation audit.

Neural inference accepts input plus a classical artifact proposal only.
All methods use identical source pairs and physical-power SNR definitions.
"""
from pathlib import Path
import os,sys,json,time,argparse,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from awrls2 import clean as clean2
from awrls2.config import Config as Config2
from awrls3 import clean,Config,ShapeDictionary
from evaluate_classical import metrics
from data_paths import paths


def proposal_worker(task):
    ident,x,s,fs,condition=task
    dictionary=ShapeDictionary.load(ROOT/'models/dictionary.npz')
    rows=[];ys={}
    configs={'V2':None,'V3_dictionary':Config(background=False,confidence_gain=False),'V3_full':Config()}
    for name,cfg in configs.items():
        t=time.perf_counter()
        if cfg is None:y,_=clean2(x,fs,Config2(fs=fs))
        else:y,_=clean(x,fs,cfg,dictionary)
        ys[name]=y
        rows.append(dict(variant=name,epoch=ident,condition=condition,elapsed_s=time.perf_counter()-t,**metrics(x,s,y,fs)))
    rows.append(dict(variant='INPUT',epoch=ident,condition=condition,elapsed_s=0,**metrics(x,s,x,fs)))
    return task,ys,rows


def tasks_for(split,p,n=None):
    pub,ip=paths(p)
    E=np.load(pub/'EEG_all_epochs.npy');O=np.load(pub/'EOG_all_epochs.npy')
    ei=p['eeg'][split];oi=p['eog'][split];num=min(len(ei),len(oi)) if n is None else n
    tasks=[]
    for j,(a,b) in enumerate(zip(ei[:num],oi[:num])):
        s=E[a];o=O[b]
        for snr in [-7,-5,-3,-1,1,3]:
            x=s+o*np.sqrt(np.sum(s*s)/np.sum(o*o)/10**(snr/10))
            tasks.append((j,x,s,256,f'public_snr_{snr}'))
        tasks.append((j,s,s,256,'public_clean'))
    if split=='validation':
        S=np.load(ip/'clean.npy');X=np.load(ip/'mixed.npy')
        for i in p['internal']['validation']:
            tasks.append((i,X[i],S[i],200,'internal_native'))
            tasks.append((i,S[i],S[i],200,'internal_clean'))
    return tasks


def run(args):
    import torch
    from awrls3.neural import NeuralCleaner
    torch.set_num_threads(2)
    p=json.loads((ROOT/'results/source_split.json').read_text())
    if args.split=='test':
        frozen=ROOT/'results/FROZEN_BEFORE_TEST.json'
        current={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'models').glob('*')) if f.is_file()}
        if not frozen.exists():raise RuntimeError('Freeze checkpoint hashes before fresh test')
        frozen_data=json.loads(frozen.read_text())
        if frozen_data['model_sha256']!=current:raise RuntimeError('Frozen model hashes changed')
        for path,digest in frozen_data['code_sha256'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Frozen code hash changed: '+path)
    tasks=tasks_for(args.split,p,args.n)
    models={name:NeuralCleaner(ROOT/f'models/{file}.pt',args.device)
            for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]}
    for seed in [20261029,20261030]:
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:
            path=ROOT/f'models/{file}_seed{seed}.pt'
            if path.exists():models[f'{name}_seed{seed}']=NeuralCleaner(path,args.device)
    out=ROOT/f'results/{args.split}_models.csv';rows=[];examples=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        # map is deterministic; buffers bounded by the modest number of 2s sources.
        for j,(task,ys,rr) in enumerate(pool.map(proposal_worker,tasks,chunksize=4)):
            ident,x,s,fs,condition=task;rows.extend(rr);proposal=x-ys['V3_dictionary']
            for name,model in models.items():
                t=time.perf_counter();y=model.clean(x,fs,proposal)
                rows.append(dict(variant=name,epoch=ident,condition=condition,elapsed_s=time.perf_counter()-t,**metrics(x,s,y,fs)))
                ys[name]=y
            if ident<8:examples.append(dict(epoch=ident,condition=condition,fs=fs,input=x,clean=s,**ys))
            if (j+1)%100==0:
                pd.DataFrame(rows).to_csv(out,index=False);print(args.split,j+1,'/',len(tasks),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out,index=False)
    summary=df.groupby(['condition','variant'])[['rmse','rrmse_t','rrmse_s','cc','dbp_total','changed']].mean()
    summary.to_csv(ROOT/f'results/{args.split}_model_summary.csv')
    ex={}
    for i,e in enumerate(examples):
        for k,v in e.items():ex[f'{i}_{k}']=v
    np.savez_compressed(ROOT/f'results/{args.split}_examples.npz',**ex)
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['validation','test'],required=True)
    ap.add_argument('--jobs',type=int,default=5);ap.add_argument('--n',type=int);ap.add_argument('--device',default='cuda')
    run(ap.parse_args())
