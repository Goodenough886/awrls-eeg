"""Execution-only continuation: same frozen functions, parallel CPU scoring.

GPU writes FLOAT64 temporary outputs. Workers score these exact arrays before
float32 archival conversion. No metric formula, model or parameter is changed.
"""
from pathlib import Path
import os,sys,json,time,gc,datetime
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from full_real import OUT,sha,load_task,context,common_row,score,save_wave,write_json,prepare


def score_unit(payload):
    task,timings=payload;kind,key,_=task
    x,s,source=load_task(task);ctx=context(x);base=common_row(kind,key,x,source);rows=[]
    for name,elapsed in timings.items():
        y=np.load(OUT/'gpu_score_cache'/f'{kind}_{key}_{name}.npy')
        row=dict(base,variant=name,compute_s=elapsed,**score(x,y,s,ctx))
        row.update(save_wave(OUT/'waveforms'/f'{kind}_{key}_{name}.npy',y));rows.append(row)
    write_json(OUT/'units'/f'{kind}_{key}_GPU.json',rows)
    return kind,key


def run():
    import torch
    from awrls3.neural import NeuralCleaner
    from awrls3.neural_batch import clean_batched
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    frozen=json.loads((ROOT/'results/FULL_RUN_FREEZE.json').read_text())
    for path,digest in frozen['files'].items():assert sha(ROOT/path)==digest,path
    ext=ROOT/'results/FULL_RUN_PARALLEL_EXECUTION.json'
    declaration={'script_sha256':sha(Path(__file__)),'frozen_execution_sha256':sha(ROOT/'results/FULL_RUN_FREEZE.json'),
       'purpose':'Parallelise CPU metrics only; same full_real.score/context/save_wave and same neural_batch.clean_batched.',
       'intermediate_dtype':'float64 before metrics, float32 final storage','metric_workers':8,'max_pending_units':12}
    if ext.exists():assert json.loads(ext.read_text())==declaration
    else:write_json(ext,declaration)
    (OUT/'gpu_score_cache').mkdir(exist_ok=True)
    previous=ROOT.parent/'awrls2';models={}
    for seed in [20261028,20261029,20261030]:
        suffix='' if seed==20261028 else f'_seed{seed}'
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:models[name+suffix]=NeuralCleaner(ROOT/f'models/{file}{suffix}.pt','cuda')
    keys=[p.stem for p in sorted((previous/'kaya_preprocessed').glob('*.npy'))];assert len(keys)==219
    tasks=[(kind,key,str(previous)) for kind in ['raw','silver'] for key in keys]
    pending=set();done_count=0
    def harvest(futures):
        nonlocal done_count
        for f in futures:
            kind,key=f.result();done_count+=1
            print('complete',done_count,'/438',kind,key,flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for task in tasks:
            kind,key,_=task
            done=OUT/'units'/f'{kind}_{key}_GPU.json'
            if done.exists() and all((OUT/'waveforms'/f'{kind}_{key}_{name}.npy').exists() for name in models):
                done_count+=1;continue
            if not (OUT/'units'/f'{kind}_{key}_CPU.json').exists():prepare(task)
            x,_,_=load_task(task);proposal=np.load(OUT/'proposal_cache'/f'{kind}_{key}.npy');timings={}
            for name,model in models.items():
                t=time.perf_counter();y=clean_batched(model,x,200,proposal,batch_size=32);timings[name]=time.perf_counter()-t
                np.save(OUT/'gpu_score_cache'/f'{kind}_{key}_{name}.npy',y)
            pending.add(pool.submit(score_unit,(task,timings)))
            finished={f for f in pending if f.done()};harvest(finished);pending-=finished
            if len(pending)>=12:
                finished,pending=wait(pending,return_when=FIRST_COMPLETED);harvest(finished)
            del x,proposal,y;gc.collect()
        while pending:
            finished,pending=wait(pending,return_when=FIRST_COMPLETED);harvest(finished)
    rows=[]
    for p in sorted((OUT/'units').glob('*.json')):rows.extend(json.loads(p.read_text()))
    df=pd.DataFrame(rows);df.to_csv(OUT/'by_unit.csv',index=False)
    numeric=[c for c in df.select_dtypes('number') if c not in ['fs','samples','seconds']]
    subjects=df.groupby(['kind','subject','variant'])[numeric].mean();subjects.to_csv(OUT/'by_subject.csv')
    subjects.groupby(['kind','variant']).mean().to_csv(OUT/'summary.csv')
    assert len(df)==438*9 and not df.duplicated(['kind','record','channel','variant']).any()
    write_json(OUT/'COMPLETENESS.json',{'records':73,'channels':3,'input_recipes':2,'variants':9,'rows':len(df),
       'output_waveforms':len(list((OUT/'waveforms').glob('*.npy'))),'full_length':True,
       'max_float32_storage_abs_error':float(df.storage_max_abs_error.max()),'parallel_scoring':True,
       'note':'Unchanged frozen methods and metric functions. V2 reused. All new outputs computed in full; float64 scored before float32 storage.'})
    print('FULL KAYA COMPLETE',flush=True)


if __name__=='__main__':run()
