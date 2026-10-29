"""Build train/validation-only proposal caches. Never opens fresh test sources."""
from pathlib import Path
import os,sys,json,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.signal import resample_poly
from concurrent.futures import ProcessPoolExecutor
from awrls3 import Config,ShapeDictionary,clean
from data_paths import paths


def work(task):
    domain,index,x,s,fs,is_clean=task
    # Template-only increment selected a priori for the neural proposal, no test tuning.
    cfg=Config(background=False,confidence_gain=False)
    dictionary=ShapeDictionary.load(ROOT/'models/dictionary.npz')
    y,_=clean(x,fs,cfg,dictionary)
    if fs!=256:
        x=resample_poly(x,32,25);s=resample_poly(s,32,25);y=resample_poly(y,32,25)
    rows=[]
    for st in range(0,len(x)-511,512):
        xx=x[st:st+512];ss=s[st:st+512];aa=xx-y[st:st+512]
        rms=max(float(np.sqrt(np.mean(xx**2))),1e-8)
        rows.append((np.stack([xx,ss,aa]).astype('float32')/rms,domain,index,is_clean,st))
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=6);args=ap.parse_args()
    p=json.loads((ROOT/'results/source_split.json').read_text());pub,ip=paths(p)
    E=np.load(pub/'EEG_all_epochs.npy');O=np.load(pub/'EOG_all_epochs.npy');S=np.load(ip/'clean.npy');X=np.load(ip/'mixed.npy')
    rng=np.random.default_rng(12345)
    for split in ['train','validation']:
        tasks=[];ei=np.array(p['eeg'][split]);oi=np.array(p['eog'][split])
        n=6000 if split=='train' else 900
        for j in range(n):
            a=int(rng.choice(ei));b=int(rng.choice(oi));snr=float(rng.uniform(-10,5));s=E[a]
            is_clean=j%5==0
            x=s.copy() if is_clean else s+O[b]*np.sqrt(np.sum(s*s)/np.sum(O[b]**2)/10**(snr/10))
            tasks.append(('public',a,x,s,256,is_clean))
        for i in p['internal'][split]:
            # Three strengths plus clean; raw truth is used ONLY at training/evaluation.
            for scale in [0,.5,1,1.5]:tasks.append(('internal',int(i),S[i]+scale*(X[i]-S[i]),S[i],200,scale==0))
        cache=[];meta=[]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for j,rows in enumerate(pool.map(work,tasks,chunksize=8)):
                for arr,domain,index,isc,st in rows:cache.append(arr);meta.append([domain,index,bool(isc),st])
                if (j+1)%300==0:print(split,j+1,'/',len(tasks),flush=True)
        np.savez_compressed(ROOT/f'results/{split}_cache.npz',data=np.asarray(cache),
                            is_clean=np.array([m[2] for m in meta],dtype=bool),
                            domain=np.array([m[0] for m in meta]))
        (ROOT/f'results/{split}_cache_metadata.json').write_text(json.dumps(meta),encoding='utf-8')
        print(split,'saved',len(cache),flush=True)
