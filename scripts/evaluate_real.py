"""Retrospective Kaya 30s silver-reference sensitivity; never used for fitting.

Use exactly the previously staged ICA-preprocessed single-channel inputs.
An ICA silver signal is only used for scoring, never model inference.
"""
from pathlib import Path
import sys,os,json,time,re,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from evaluate_models import proposal_worker
from evaluate_classical import metrics


if __name__=='__main__':
    import torch
    from awrls3.neural import NeuralCleaner
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,default=ROOT.parent/'awrls2/awrls2_package/results')
    ap.add_argument('--jobs',type=int,default=5);args=ap.parse_args();torch.set_num_threads(2)
    models={name:NeuralCleaner(ROOT/f'models/{file}.pt','cuda')
            for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]}
    for seed in [20261029,20261030]:
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:
            path=ROOT/f'models/{file}_seed{seed}.pt'
            if path.exists():models[f'{name}_seed{seed}']=NeuralCleaner(path,'cuda')
    tasks=[]
    for raw in sorted((args.previous/'silver').glob('*_raw.npy')):
        key=raw.name[:-8];silver=raw.with_name(key+'_silver.npy')
        x=np.array(np.load(raw,mmap_mode='r')[:6000]);s=np.array(np.load(silver,mmap_mode='r')[:6000])
        tasks.append((key,x,s,200,'kaya_30s_silver'))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for j,(task,ys,rr) in enumerate(pool.map(proposal_worker,tasks,chunksize=2)):
            key,x,s,fs,condition=task;rows.extend(rr);proposal=x-ys['V3_dictionary']
            for name,model in models.items():
                t=time.perf_counter();y=model.clean(x,fs,proposal)
                rows.append(dict(variant=name,epoch=key,condition=condition,elapsed_s=time.perf_counter()-t,**metrics(x,s,y,fs)))
            if (j+1)%30==0:print(j+1,'/',len(tasks),flush=True)
    df=pd.DataFrame(rows);df['subject']=df.epoch.str.extract(r'Subject([A-Z])')
    df.to_csv(ROOT/'results/kaya_30s_by_unit.csv',index=False)
    by_subject=df.groupby(['subject','variant'])[['rmse','rrmse_t','rrmse_s','cc','dbp_total']].mean()
    by_subject.to_csv(ROOT/'results/kaya_30s_by_subject.csv')
    print(by_subject.groupby('variant').mean().to_string(),flush=True)
