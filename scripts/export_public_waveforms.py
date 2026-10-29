"""Export every frozen test waveform, verifying reexecution against scored CSV.

This is deterministic reexecution of frozen models, not another selection round.
"""
from pathlib import Path
import os,sys,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from evaluate_models import tasks_for,proposal_worker
from evaluate_classical import metrics

if __name__=='__main__':
    import torch
    from awrls3.neural import NeuralCleaner
    torch.set_num_threads(2)
    p=json.loads((ROOT/'results/source_split.json').read_text());tasks=tasks_for('test',p)
    prior=pd.read_csv(ROOT/'results/test_models.csv').set_index(['condition','variant','epoch'])
    out=ROOT/'results/full_public_test';out.mkdir(exist_ok=True);models={};max_error=0.0
    for seed in [20261028,20261029,20261030]:
        suffix='' if seed==20261028 else f'_seed{seed}'
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:models[name+suffix]=NeuralCleaner(ROOT/f'models/{file}{suffix}.pt','cuda')
    for condition in sorted({t[-1] for t in tasks}):
        dest=out/f'{condition}.npz';proof=out/f'{condition}_verification.json'
        if dest.exists() and proof.exists():continue
        group=[t for t in tasks if t[-1]==condition];waves={'input':[],'clean':[]};err=0.0
        with ProcessPoolExecutor(max_workers=3) as pool:
            for j,(task,ys,rr) in enumerate(pool.map(proposal_worker,group,chunksize=4)):
                i,x,s,fs,cond=task;a=x-ys['V3_dictionary']
                for name,model in models.items():ys[name]=model.clean(x,fs,a)
                for name,y in ys.items():
                    m=metrics(x,s,y,fs);r=prior.loc[(cond,name,i)]
                    for k in ['rrmse_t','cc']:
                        diff=abs(m[k]-float(r[k]));err=max(err,diff)
                        if diff>2e-6:raise AssertionError((cond,name,i,k,diff))
                    waves.setdefault(name,[]).append(y)
                waves['input'].append(x);waves['clean'].append(s)
                if (j+1)%150==0:print(condition,j+1,'/',len(group),flush=True)
        np.savez_compressed(dest,**{k:np.asarray(v) for k,v in waves.items()},fs=256,
             eeg_indices=np.array(p['eeg']['test'][:len(group)]),eog_indices=np.array(p['eog']['test'][:len(group)]))
        proof.write_text(json.dumps({'condition':condition,'pairs':len(group),'max_metric_abs_difference':err,'reference_csv':'test_models.csv','frozen_reexecution':True},indent=2))
        max_error=max(max_error,err)
    proofs=[json.loads(f.read_text()) for f in out.glob('*_verification.json')]
    assert len(proofs)==7
    (out/'COMPLETENESS.json').write_text(json.dumps({'conditions':7,'pairs':554,'all_test_samples_saved':True,'max_metric_abs_difference':max(r['max_metric_abs_difference'] for r in proofs)},indent=2))
    print('PUBLIC WAVEFORMS COMPLETE',flush=True)
