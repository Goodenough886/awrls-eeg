"""Full-length 73-record Kaya run with checkpointing and bounded GPU batching.

CPU workers prepare frozen classical candidates. GPU consumes completed units.
Historical V2 full waveforms are reused, with SHA256 provenance and same frozen
legacy source. New V3 outputs are computed over EVERY sample, two input recipes.
"""
from pathlib import Path
import os,sys,json,time,hashlib,re,argparse,gc
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.signal import butter,sosfiltfilt
from awrls2.core import _overlap_add
from awrls2.detect import lowpass
from awrls2.evaluate import eval_detector,event_mask,rrmse_spectral,dbp
from awrls3 import Config,ShapeDictionary,clean
OUT=ROOT/'results/full_kaya'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)


def context(x):
    ev=eval_detector(x,200)
    return {'blink':event_mask(len(x),ev,10),'outside':~event_mask(len(x),ev,40),
       'lp':lowpass(x,200,12),'bp':sosfiltfilt(butter(4,[13,40],btype='band',fs=200,output='sos'),x),'candidate_count':len(ev)}


def score(x,y,s,ctx):
    blink=ctx['blink'];outside=ctx['outside'];lp=lowpass(y,200,12)
    bp=sosfiltfilt(butter(4,[13,40],btype='band',fs=200,output='sos'),y)
    m={'candidate_count':ctx['candidate_count'],
       'lf_energy_reduction_percent':float(100*(1-np.sum(lp[blink]**2)/(np.sum(ctx['lp'][blink]**2)+1e-12))) if blink.any() else None,
       'outside_window_cc':float(np.corrcoef(x[outside],y[outside])[0,1]) if outside.sum()>8 else None,
       'outside_window_nrmse_percent':float(100*np.sqrt(np.mean((y[outside]-x[outside])**2))/(np.sqrt(np.mean(x[outside]**2))+1e-12)) if outside.sum()>8 else None,
       'hf_13_40_cc':float(np.corrcoef(ctx['bp'],bp)[0,1]),
       'rms_modification':float(np.sqrt(np.mean((y-x)**2)))}
    if s is not None:
        error=y-s
        m.update(silver_rmse=float(np.sqrt(np.mean(error**2))),silver_mae=float(np.mean(abs(error))),
           silver_rrmse_t=float(np.sqrt(np.mean(error**2))/(np.sqrt(np.mean(s*s))+1e-12)),
           silver_rrmse_s=rrmse_spectral(s,y,200),silver_cc=float(np.corrcoef(s,y)[0,1]))
        m.update({'silver_'+k:v for k,v in dbp(s,y,200).items()})
    return m


def save_wave(path,y):
    # Loss-bounded float32 STORAGE only, scores computed before quantisation.
    z=y.astype(np.float32)
    if not np.isfinite(z).all():raise FloatingPointError('float32 overflow')
    error=float(np.max(abs(y-z.astype(float))))
    temp=path.with_suffix('.tmp.npy');np.save(temp,z);temp.replace(path)
    return {'output_sha256':sha(path),'output_samples':len(z),'storage_dtype':'float32','storage_max_abs_error':error}


def load_task(task):
    kind,key,previous=task;previous=Path(previous)
    source=(previous/'kaya_preprocessed'/f'{key}.npy') if kind=='raw' else previous/'awrls2_package/results/silver'/f'{key}_raw.npy'
    x=np.array(np.load(source,mmap_mode='r'),dtype=float)
    sp=previous/'awrls2_package/results/silver'/f'{key}_silver.npy'
    s=np.array(np.load(sp,mmap_mode='r'),dtype=float) if kind=='silver' else None
    return x,s,source


def common_row(kind,key,x,source):
    return dict(kind=kind,record=key.rsplit('_',1)[0],channel=key.rsplit('_',1)[1],subject=re.search(r'Subject([A-Z])',key).group(1),
       samples=len(x),seconds=len(x)/200,fs=200,input_path=str(source),input_sha256=sha(source))


def prepare(task):
    kind,key,previous=task;checkpoint=OUT/'units'/f'{kind}_{key}_CPU.json'
    if checkpoint.exists():
        data=json.loads(checkpoint.read_text())
        if all((OUT/'waveforms'/f'{kind}_{key}_{name}.npy').exists() for name in ['V2','V3_dictionary','V3_full']):return task
    x,s,source=load_task(task);ctx=context(x);base=common_row(kind,key,x,source)
    dictionary=ShapeDictionary.load(ROOT/'models/dictionary.npz');rows=[]
    for name,cfg in [('V2',None),('V3_dictionary',Config(background=False,confidence_gain=False)),('V3_full',Config())]:
        t=time.perf_counter();extra={}
        if cfg is None:
            folder='kaya_full' if kind=='raw' else 'kaya_silver_full'
            old=Path(previous)/'awrls2_package/results'/folder/'cleaned'/f'{key}_PROPOSED.npy'
            y=np.asarray(np.load(old,mmap_mode='r'),float)
            if len(y)!=len(x):raise ValueError('Legacy baseline/input alignment mismatch')
            extra={'reused_frozen_baseline':str(old),'baseline_sha256':sha(old)}
        else:
            y=_overlap_add(x,200,lambda v:clean(v,200,cfg,dictionary)[0],30,4)
        # Float64 proposal cache preserves neural input; final archive waveforms float32.
        if name=='V3_dictionary':np.save(OUT/'proposal_cache'/f'{kind}_{key}.npy',x-y)
        row=dict(base,variant=name,compute_s=time.perf_counter()-t,**score(x,y,s,ctx),**extra)
        row.update(save_wave(OUT/'waveforms'/f'{kind}_{key}_{name}.npy',y));rows.append(row)
    write_json(checkpoint,rows)
    return task


def run(args):
    import torch
    from awrls3.neural import NeuralCleaner
    from awrls3.neural_batch import clean_batched
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    for d in ['units','waveforms','proposal_cache']:(OUT/d).mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'results/FULL_RUN_FREEZE.json').read_text())
    for path,digest in frozen['files'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Full-run frozen file changed: '+path)
    models={}
    for seed in [20261028,20261029,20261030]:
        suffix='' if seed==20261028 else f'_seed{seed}'
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:
            models[name+suffix]=NeuralCleaner(ROOT/f'models/{file}{suffix}.pt','cuda')
    keys=[p.stem for p in sorted((args.previous/'kaya_preprocessed').glob('*.npy'))]
    assert len(keys)==219
    tasks=[(kind,key,str(args.previous)) for kind in ['raw','silver'] for key in keys]
    finished=0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(prepare,t):t for t in tasks}
        for f in as_completed(futures):
            task=f.result();kind,key,_=task
            done=OUT/'units'/f'{kind}_{key}_GPU.json'
            if done.exists():
                rows=json.loads(done.read_text())
                if len(rows)==6 and all((OUT/'waveforms'/f'{kind}_{key}_{name}.npy').exists() for name in models):
                    finished+=1;continue
            x,s,source=load_task(task);ctx=context(x);base=common_row(kind,key,x,source)
            proposal=np.load(OUT/'proposal_cache'/f'{kind}_{key}.npy');rows=[]
            for name,model in models.items():
                t=time.perf_counter();y=clean_batched(model,x,200,proposal,batch_size=args.batch)
                row=dict(base,variant=name,compute_s=time.perf_counter()-t,**score(x,y,s,ctx))
                row.update(save_wave(OUT/'waveforms'/f'{kind}_{key}_{name}.npy',y));rows.append(row)
            write_json(done,rows);finished+=1
            print('complete',finished,'/',len(tasks),kind,key,flush=True)
            del x,s,proposal,ctx,rows;gc.collect()
    rows=[]
    for p in sorted((OUT/'units').glob('*.json')):rows.extend(json.loads(p.read_text()))
    df=pd.DataFrame(rows);df.to_csv(OUT/'by_unit.csv',index=False)
    numeric=[c for c in df.select_dtypes('number') if c not in ['fs','samples','seconds']]
    subjects=df.groupby(['kind','subject','variant'])[numeric].mean();subjects.to_csv(OUT/'by_subject.csv')
    subjects.groupby(['kind','variant']).mean().to_csv(OUT/'summary.csv')
    assert len(df)==438*9 and not df.duplicated(['kind','record','channel','variant']).any()
    assert df.subject.nunique()==13
    write_json(OUT/'COMPLETENESS.json',{'records':73,'channels':3,'input_recipes':2,'variants':9,'rows':len(df),
       'output_waveforms':len(list((OUT/'waveforms').glob('*.npy'))),'full_length':True,
       'max_float32_storage_abs_error':float(df.storage_max_abs_error.max()),
       'note':'V2 reused from frozen previous full-length outputs. All new methods computed over every sample. Scores before float32 storage.'})
    print('FULL KAYA COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,default=ROOT.parent/'awrls2')
    ap.add_argument('--jobs',type=int,default=10);ap.add_argument('--batch',type=int,default=32)
    run(ap.parse_args())
