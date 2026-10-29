"""Prespecified component ablations on development data only."""
from pathlib import Path
import sys,os,json,time,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from awrls2 import clean as old_clean
from awrls2.config import Config as OldConfig
from awrls2.evaluate import rrmse_spectral,dbp
from awrls3 import clean,Config,ShapeDictionary
from data_paths import paths

VARIANTS={
 'V2':None,
 'V3_aligned':Config(dictionary=False,background=False,confidence_gain=False,self_penalty=1),
 'V3_self_penalty':Config(dictionary=False,background=False,confidence_gain=False),
 'V3_dictionary':Config(background=False,confidence_gain=False),
 'V3_background':Config(confidence_gain=False),
 'V3_full':Config(),
 'V3_gls':Config(estimator='aligned_gls'),
 'V3_rescue':Config(rescue=True),
}

def metrics(x,s,y,fs):
    e=y-s;rms=np.sqrt(np.mean(s*s))+1e-12
    input_rms=np.sqrt(np.mean(x*x))+1e-12
    modification=float(np.sqrt(np.mean((y-x)**2))/input_rms)
    return dict(rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),
       rrmse_t=float(np.sqrt(np.mean(e*e))/rms),cc=float(np.corrcoef(s,y)[0,1]) if np.std(y)>0 else 0,
       rrmse_s=rrmse_spectral(s,y,fs),changed=bool(np.max(abs(y-x))>1e-8),
       normalized_modification=modification,changed_gt_5pct_input_rms=modification>.05,**dbp(s,y,fs))

def worker(task):
    tag,i,x,s,fs,cond=task
    cfg=VARIANTS[tag];dictionary=ShapeDictionary.load(ROOT/'models/dictionary.npz') if cfg and cfg.dictionary else None
    t=time.perf_counter()
    if cfg is None:y,diag=old_clean(x,fs,OldConfig(fs=fs))
    else:y,diag=clean(x,fs,cfg,dictionary)
    row=dict(variant=tag,epoch=i,condition=cond,elapsed_s=time.perf_counter()-t,**metrics(x,s,y,fs))
    if cfg:
        events=diag.get('events',[])
        row.update(dictionary_used=sum(e.get('dictionary_used',False) for e in events),
                   mean_gain=float(np.mean([e['gain'] for e in events])) if events else 0)
    return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=6);ap.add_argument('--n-public',type=int,default=80)
    args=ap.parse_args();p=json.loads((ROOT/'results/source_split.json').read_text())
    pub,ip=paths(p)
    s=np.load(ip/'clean.npy');x=np.load(ip/'mixed.npy')
    tasks=[]
    for tag in VARIANTS:
        for i in p['internal']['validation']:
            tasks.extend([(tag,i,x[i],s[i],200,'internal_native'),(tag,i,s[i],s[i],200,'internal_clean')])
    E=np.load(pub/'EEG_all_epochs.npy');O=np.load(pub/'EOG_all_epochs.npy')
    ei=p['eeg']['validation'][:args.n_public];oi=p['eog']['validation'][:args.n_public]
    for j,(a,b) in enumerate(zip(ei,oi)):
        eeg=E[a];eog=O[b]
        for snr in [-7,-3,3]:
            mix=eeg+eog*np.sqrt(np.sum(eeg**2)/np.sum(eog**2)/10**(snr/10))
            for tag in VARIANTS:tasks.append((tag,j,mix,eeg,256,f'public_{snr}'))
        for tag in VARIANTS:tasks.append((tag,j,eeg,eeg,256,'public_clean'))
    out=ROOT/'results/classical_development.csv';rows=[]
    (ROOT/'results/classical_configs.json').write_text(json.dumps({k:v.to_dict() if v else None for k,v in VARIANTS.items()},indent=2))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(worker,tasks,chunksize=8):
            rows.append(row)
            if len(rows)%200==0:
                pd.DataFrame(rows).to_csv(out,index=False);print(len(rows),'/',len(tasks),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out,index=False)
    summary=df.groupby(['condition','variant'])[['rmse','rrmse_t','cc','dbp_total','changed']].mean()
    summary.to_csv(ROOT/'results/classical_summary.csv');print(summary.to_string(),flush=True)
