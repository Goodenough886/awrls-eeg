"""All 300 historical internal epochs, six SNRs, native and clean controls.

Train/development identities are retained per row. Never called new holdout.
Outputs of primary methods saved for every sample; all three neural seeds scored.
"""
from pathlib import Path
import os,sys,json,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd,torch
from concurrent.futures import ProcessPoolExecutor
from evaluate_models import proposal_worker
from evaluate_classical import metrics
from awrls2.evaluate import paired_metrics
from awrls3.neural import NeuralCleaner
from data_paths import paths

if __name__=='__main__':
    torch.set_num_threads(2)
    p=json.loads((ROOT/'results/source_split.json').read_text());pub,ip=paths(p)
    X=np.load(ip/'mixed.npy');S=np.load(ip/'clean.npy');train=set(p['internal']['train'])
    OUT=ROOT/'results/full_internal';OUT.mkdir(exist_ok=True)
    models={}
    for seed in [20261028,20261029,20261030]:
        suffix='' if seed==20261028 else f'_seed{seed}'
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:models[name+suffix]=NeuralCleaner(ROOT/f'models/{file}{suffix}.pt','cuda')
    all_rows=[]
    for condition in ['native','clean',-7,-5,-3,-1,1,3]:
        path=OUT/f'{condition}_by_epoch.csv';wave=OUT/f'{condition}_waveforms.npz'
        if path.exists() and wave.exists():
            all_rows.extend(pd.read_csv(path).to_dict('records'));continue
        tasks=[]
        for i,(x,s) in enumerate(zip(X,S)):
            if condition=='clean':xx=s.copy()
            elif condition=='native':xx=x.copy()
            else:
                a=x-s;xx=s+a*np.sqrt(np.sum(s*s)/(np.sum(a*a)+1e-12)/10**(condition/10))
            tasks.append((i,xx,s,200,str(condition)))
        rows=[];waves={'mixed':[],'clean':[]}
        with ProcessPoolExecutor(max_workers=6) as pool:
            for j,(task,ys,rr) in enumerate(pool.map(proposal_worker,tasks,chunksize=4)):
                i,x,s,fs,cond=task;proposal=x-ys['V3_dictionary']
                for name,model in models.items():
                    t=time.perf_counter();y=model.clean(x,fs,proposal);ys[name]=y
                    rr.append(dict(variant=name,epoch=i,condition=cond,elapsed_s=time.perf_counter()-t,**metrics(x,s,y,fs)))
                for row in rr:
                    row['split_role']='training_seen' if i in train else 'historical_development'
                    # Sparse support truth is valid for this implanted dataset only.
                    if condition!='clean' and row['variant']!='INPUT':
                        event=paired_metrics(x,s,ys[row['variant']],fs)
                        for k in ['f1','recall_percent','precision_percent','barr_percent','rmse_in_window','rmse_out_window']:
                            row['removed_component_'+k]=event[k]
                rows.extend(rr);waves['mixed'].append(x);waves['clean'].append(s)
                for name,y in ys.items():waves.setdefault(name,[]).append(y)
                if (j+1)%100==0:print(condition,j+1,'/300',flush=True)
        pd.DataFrame(rows).to_csv(path,index=False)
        np.savez_compressed(wave,**{k:np.asarray(v) for k,v in waves.items()});all_rows.extend(rows)
    df=pd.DataFrame(all_rows);df.to_csv(OUT/'by_epoch.csv',index=False)
    assert len(df)==300*8*10 and not df.duplicated(['condition','variant','epoch']).any()
    columns=['rmse','rrmse_t','rrmse_s','cc','dbp_total','changed','changed_gt_5pct_input_rms']
    df.groupby(['split_role','condition','variant'])[columns].mean().to_csv(OUT/'summary_by_split.csv')
    df.groupby(['condition','variant'])[columns].mean().to_csv(OUT/'summary_all_descriptive.csv')
    (OUT/'COMPLETENESS.json').write_text(json.dumps({'epochs':300,'conditions':8,'variants':10,'rows':len(df),'all_samples_saved':True,'train_epochs':150,'historical_dev_epochs':150},indent=2))
    print('FULL INTERNAL COMPLETE',flush=True)
