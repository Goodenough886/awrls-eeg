"""Additional frozen-method comparisons. No parameter/model tuning.

Full Kaya uses the original Claude 120 s / 2 s outer overlap recipe.
One atomic checkpoint per epoch or full record; OVME additionally checkpoints
each outer block. Smoke runs have separate output namespaces. Input, source,
configuration and execution hashes bind every resumable run.
"""
from pathlib import Path
import os,sys,json,time,argparse,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.signal.windows import tukey
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import claude_frozen_baselines_original as original
from full_real import sha,write_json,load_task,context,score,common_row,save_wave
from evaluate_classical import metrics
from awrls2.evaluate import paired_metrics
from data_paths import paths
METHODS=['MCA','FBSE-EWT-LPATV','OVME-REG']
STATE={}

def init(src,out,previous,backend):
    original._setup(ROOT,Path(src))
    STATE.update(out=Path(out),previous=previous,backend=backend)
    if backend!='numpy':
        from supplement_vme_native import enable
        enable()

def epoch_work(task):
    part,method,i,cond=task;out=STATE['out'];dest=out/'units'/f'{part}_{method}_{cond}_{i:04d}.json'
    if dest.exists():return json.loads(dest.read_text())
    p=json.loads((ROOT/'results/source_split.json').read_text());pub,ip=paths(p)
    if part=='public':
        s=np.asarray(np.load(pub/'EEG_all_epochs.npy',mmap_mode='r')[p['eeg']['test'][i]],float)
        a=np.asarray(np.load(pub/'EOG_all_epochs.npy',mmap_mode='r')[p['eog']['test'][i]],float)
        x=s.copy() if cond=='public_clean' else s+a*np.sqrt(np.sum(s*s)/np.sum(a*a)/10**(int(cond.rsplit('_',1)[1])/10))
        fs=256;extra={}
    else:
        s=np.asarray(np.load(ip/'clean.npy',mmap_mode='r')[i],float)
        mixed=np.asarray(np.load(ip/'mixed.npy',mmap_mode='r')[i],float);a=mixed-s
        x=s.copy() if cond=='clean' else mixed if cond=='native' else s+a*np.sqrt(np.sum(s*s)/(np.sum(a*a)+1e-12)/10**(int(cond)/10))
        fs=200;extra={'split_role':'training_seen' if i in p['internal']['train'] else 'historical_development'}
    t=time.perf_counter();y=original._run(method,x,fs)
    row=dict(variant=method,epoch=i,condition=cond,elapsed_s=time.perf_counter()-t,**metrics(x,s,y,fs),**extra)
    if part=='internal' and cond!='clean':
        ev=paired_metrics(x,s,y,fs)
        for k in ['f1','recall_percent','precision_percent','barr_percent','rmse_in_window','rmse_out_window']:
            row['removed_component_'+k]=ev[k]
    path=out/'waveforms'/f'{part}_{method}_{cond}_{i:04d}.npy'
    row.update(save_wave(path,y));row['waveform']=str(path.relative_to(out));write_json(dest,row)
    return row

def full_clean(x,method,kind,key):
    block=24000;overlap=400;step=block-overlap
    summed=np.zeros_like(x);weights=np.zeros_like(x)
    block_dir=STATE['out']/'block_checkpoints'/f'{kind}_{key}_{method}'
    if method=='OVME-REG':block_dir.mkdir(parents=True,exist_ok=True)
    if len(x)<=block:return original._run(method,x,200)
    for start in range(0,max(1,len(x)-overlap),step):
        end=min(len(x),start+block);seg=x[start:end]
        bp=block_dir/f'{start:010d}_{end:010d}.npy'
        if method=='OVME-REG' and bp.exists():y=np.load(bp)
        else:
            y=original._run(method,seg.copy(),200)
            if method=='OVME-REG':
                tmp=bp.with_suffix('.tmp.npy');np.save(tmp,y);tmp.replace(bp)
        if y.shape!=seg.shape or not np.isfinite(y).all():raise ValueError('invalid block')
        win=tukey(len(seg),alpha=min(1.,2.*overlap/max(len(seg),1)))
        if start==0:win[:min(overlap,len(win))]=1
        if end==len(x):win[max(0,len(win)-overlap):]=1
        summed[start:end]+=y*win;weights[start:end]+=win
        if end==len(x):break
    if not (weights>0).all():raise ValueError('uncovered samples')
    return summed/np.maximum(weights,1e-12)

def kaya_work(task):
    method,kind,key,seconds=task;out=STATE['out'];dest=out/'units'/f'kaya_{kind}_{key}_{method}.json'
    if dest.exists():return json.loads(dest.read_text())
    x,s,source=load_task((kind,key,STATE['previous']))
    if seconds>0:x=x[:int(seconds*200)];s=None if s is None else s[:len(x)]
    t=time.perf_counter();y=full_clean(x,method,kind,key)
    row=common_row(kind,key,x,source)
    row.update(variant=method,compute_s=time.perf_counter()-t,analysed_seconds=len(x)/200,**score(x,y,s,context(x)))
    path=out/'waveforms'/f'{kind}_{key}_{method}.npy';row.update(save_wave(path,y));row['waveform']=str(path.relative_to(out))
    write_json(dest,row);return row

def bounded_run(fn,tasks,args,out):
    started=time.perf_counter();pending={};iterator=iter(tasks);count=0;total=len(tasks)
    with ProcessPoolExecutor(args.jobs,initializer=init,initargs=(args.wj_src,str(out),str(args.previous),args.backend)) as pool:
        for _ in range(min(2*args.jobs,total)):
            task=next(iterator);pending[pool.submit(fn,task)]=task
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future);row=future.result();count+=1
                print(json.dumps({'completed':count,'total':total,'task':task,'elapsed_s':round(time.perf_counter()-started)},ensure_ascii=True),flush=True)
                write_json(out/'progress.json',{'completed':count,'total':total,'last_task':task,'elapsed_s':time.perf_counter()-started,'updated':time.strftime('%Y-%m-%d %H:%M:%S')})
                try:nexttask=next(iterator)
                except StopIteration:continue
                pending[pool.submit(fn,nexttask)]=nexttask

def run(args):
    out=ROOT/'results'/'supplement_20261029'/args.tag
    for d in ['units','waveforms','block_checkpoints']:(out/d).mkdir(parents=True,exist_ok=True)
    src=Path(args.wj_src);p=json.loads((ROOT/'results/source_split.json').read_text());pub,ip=paths(p)
    manifest={'part':args.part,'methods':args.methods,'limit':args.limit,'kaya_seconds':args.kaya_seconds,
      'backend':args.backend,'outer_block_s':120,'outer_overlap_s':2,'configs':{m:original.CONFIGS[m] for m in args.methods},
      'source_split_sha256':sha(ROOT/'results/source_split.json'),
      'baseline_source_sha256':{str(f.relative_to(src)):sha(f) for f in sorted((src/'blink_repro/methods').glob('*.py'))},
      'runner_sha256':sha(Path(__file__)),'claude_runner_sha256':sha(Path(original.__file__)),
      'metric_sha256':{str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'scripts/full_real.py',ROOT/'scripts/evaluate_classical.py',ROOT/'awrls2/evaluate.py']}}
    if args.backend!='numpy':
        manifest['native_sha256']={n:sha(ROOT/'scripts'/n) for n in ['supplement_vme_native.py','supplement_vme_native.dll','supplement_vme_native.cpp']}
    if args.part=='kaya':
        keys=sorted(q.stem for q in (args.previous/'kaya_preprocessed').glob('*.npy'))
        if len(keys)!=219:raise ValueError(f'Expected 219 channels, got {len(keys)}')
        keys=keys[:args.limit] if args.limit else keys
        manifest['inputs']={str(q):sha(q) for key in keys for q in [args.previous/'kaya_preprocessed'/f'{key}.npy',args.previous/'awrls2_package/results/silver'/f'{key}_raw.npy',args.previous/'awrls2_package/results/silver'/f'{key}_silver.npy']}
        tasks=[(m,kind,key,args.kaya_seconds) for m in args.methods for kind in ['raw','silver'] for key in keys]
        fn=kaya_work
    else:
        manifest['inputs']={str(q):sha(q) for q in ([pub/'EEG_all_epochs.npy',pub/'EOG_all_epochs.npy'] if args.part=='public' else [ip/'mixed.npy',ip/'clean.npy'])}
        init(src,out,str(args.previous),args.backend)
        # Check all generated input/reference metrics before any correction.
        if args.part=='public':
            _,deviation=original._public_tasks(ROOT,args.limit);manifest['input_check_max_abs']=deviation
            conditions=['public_snr_'+str(v) for v in [-7,-5,-3,-1,1,3]]+['public_clean'];n=554
        else:
            original._internal_tasks(ROOT,args.limit);manifest['input_check']='passed'
            conditions=['native','clean','-7','-5','-3','-1','1','3'];n=300
        n=min(n,args.limit) if args.limit else n
        tasks=[(args.part,m,i,c) for m in args.methods for c in conditions for i in range(n)];fn=epoch_work
    protocol=out/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text())!=manifest:raise RuntimeError('Resume protocol mismatch; use another output tag')
    else:write_json(protocol,manifest)
    bounded_run(fn,tasks,args,out)
    rows=[json.loads(q.read_text()) for q in sorted((out/'units').glob('*.json'))]
    if len(rows)!=len(tasks):raise ValueError('completion count mismatch')
    df=pd.DataFrame(rows);df.to_csv(out/'by_unit.csv',index=False)
    write_json(out/'COMPLETENESS.json',{'expected':len(tasks),'actual':len(rows),'full_length':args.kaya_seconds==0,'finished':time.strftime('%Y-%m-%d %H:%M:%S')})
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['public','internal','kaya'],required=True)
    ap.add_argument('--tag',required=True);ap.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    ap.add_argument('--wj-src',default=r'C:\Users\13665\Desktop\eegqz\databasic\wj\src')
    ap.add_argument('--previous',type=Path,default=ROOT.parent/'awrls2');ap.add_argument('--jobs',type=int,default=8)
    ap.add_argument('--limit',type=int);ap.add_argument('--kaya-seconds',type=float,default=0)
    ap.add_argument('--backend',choices=['numpy','native'],default='numpy');run(ap.parse_args())
