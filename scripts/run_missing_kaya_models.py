"""Frozen nine existing methods on the two newly retrieved official sessions."""
from pathlib import Path
import os,sys,json,time,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from full_real import load_task,common_row,context,score,save_wave,sha,write_json
from awrls2 import clean as clean2
from awrls2.config import Config as Config2
from awrls2.core import _overlap_add
from awrls3 import clean,Config,ShapeDictionary
BASE=ROOT/'results/supplement_20261029';PREVIOUS=BASE/'missing_sessions_staging';OUT=BASE/'missing_sessions_existing_models'

def cpu(task):
    kind,key=task;dest=OUT/'units'/f'{kind}_{key}_CPU.json'
    if dest.exists():return task
    x,s,source=load_task((kind,key,PREVIOUS));ctx=context(x);base=common_row(kind,key,x,source);rows=[]
    dictionary=ShapeDictionary.load(ROOT/'models/dictionary.npz')
    for name,cfg in [('V2',None),('V3_dictionary',Config(background=False,confidence_gain=False)),('V3_full',Config())]:
        t=time.perf_counter()
        fn=(lambda v:clean2(v,200,Config2(fs=200))[0]) if cfg is None else (lambda v:clean(v,200,cfg,dictionary)[0])
        y=_overlap_add(x,200,fn,30,4)
        if name=='V3_dictionary':np.save(OUT/'proposal_cache'/f'{kind}_{key}.npy',x-y)
        path=OUT/'waveforms'/f'{kind}_{key}_{name}.npy'
        row=dict(base,variant=name,compute_s=time.perf_counter()-t,**score(x,y,s,ctx),**save_wave(path,y),waveform=str(path.relative_to(OUT)))
        rows.append(row)
    write_json(dest,rows);return task

def run(args):
    for d in ['units','waveforms','proposal_cache']:(OUT/d).mkdir(parents=True,exist_ok=True)
    keys=sorted(p.stem for p in (PREVIOUS/'kaya_preprocessed').glob('*.npy'))
    if len(keys)!=6:raise ValueError('Need six new channels')
    for p in (PREVIOUS/'awrls2_package/results/kaya_audit').glob('*_preparation.json'):
        if not json.loads(p.read_text())['silver']['valid_for_agreement']:raise ValueError('Invalid ICA reference '+str(p))
    frozen=json.loads((ROOT/'results/FULL_RUN_FREEZE.json').read_text())
    for path,digest in frozen['files'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen source changed '+path)
    protocol={'keys':keys,'weights':{p.name:sha(p) for p in (ROOT/'models').glob('*') if p.is_file()},'runner':sha(Path(__file__)),
        'source_freeze':sha(ROOT/'results/FULL_RUN_FREEZE.json'),'note':'Additional official sessions; no retraining or selection; FP32 CUDA as previous full Kaya run.'}
    write_json(OUT/'protocol.json',protocol);tasks=[(kind,key) for kind in ['raw','silver'] for key in keys]
    with ProcessPoolExecutor(args.jobs) as pool:
        for task in pool.map(cpu,tasks):print('CPU DONE',task,flush=True)
    import torch
    from awrls3.neural import NeuralCleaner
    from awrls3.neural_batch import clean_batched
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    models={}
    for seed in [20261028,20261029,20261030]:
        suffix='' if seed==20261028 else f'_seed{seed}'
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:models[name+suffix]=NeuralCleaner(ROOT/f'models/{file}{suffix}.pt','cuda')
    for kind,key in tasks:
        dest=OUT/'units'/f'{kind}_{key}_GPU.json'
        if dest.exists():continue
        x,s,source=load_task((kind,key,PREVIOUS));ctx=context(x);base=common_row(kind,key,x,source);rows=[]
        proposal=np.load(OUT/'proposal_cache'/f'{kind}_{key}.npy')
        for name,model in models.items():
            t=time.perf_counter();y=clean_batched(model,x,200,proposal if name.startswith('V3_HYBRID') else None,batch_size=32)
            path=OUT/'waveforms'/f'{kind}_{key}_{name}.npy'
            rows.append(dict(base,variant=name,compute_s=time.perf_counter()-t,**score(x,y,s,ctx),**save_wave(path,y),waveform=str(path.relative_to(OUT))))
        write_json(dest,rows);print('GPU DONE',kind,key,flush=True)
    rows=[row for p in (OUT/'units').glob('*.json') for row in json.loads(p.read_text())]
    assert len(rows)==108;pd.DataFrame(rows).to_csv(OUT/'by_unit.csv',index=False)
    write_json(OUT/'COMPLETENESS.json',{'records':2,'channels':3,'recipes':2,'variants':9,'waveforms':108})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=2);run(ap.parse_args())
