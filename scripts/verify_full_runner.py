"""Validate throughput wrapper and historical baseline reuse before full run."""
from pathlib import Path
import sys,json,time,hashlib,datetime,os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,torch
from awrls3.neural import NeuralCleaner
from awrls3.neural_batch import clean_batched
from awrls2 import clean
from awrls2.config import Config
from full_real import load_task,sha,context,score
from awrls2.evaluate import real_metrics

if __name__=='__main__':
    torch.set_num_threads(2);previous=ROOT.parent/'awrls2';report={'batch_comparisons':[],'baseline_reuse':[],
        'precision_policy':'Full run disables cuDNN TF32 for both scalar and batched inference; prior test used torch default. Also measure drift from that default.'}
    key=next((previous/'kaya_preprocessed').glob('*.npy')).stem
    for name in ['hybrid_tcn','standalone_tcn']:
        model=NeuralCleaner(ROOT/f'models/{name}.pt','cuda')
        rng=np.random.default_rng(8)
        for fs,n in [(200,6000),(200,15531),(256,512),(256,10511)]:
            t=np.arange(n)/fs;x=20*np.sin(2*np.pi*10*t)+10*rng.normal(size=n);a=7*np.sin(2*np.pi*1.2*t)
            torch.backends.cudnn.allow_tf32=True
            prior_precision=model.clean(x,fs,a)
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.matmul.allow_tf32=False
            one=model.clean(x,fs,a);batch=clean_batched(model,x,fs,a,batch_size=8)
            diff=float(np.max(abs(one-batch)));relative=diff/(np.sqrt(np.mean(one**2))+1e-12)
            assert relative<2e-6,(name,fs,n,relative)
            drift=float(np.max(abs(prior_precision-batch))/(np.sqrt(np.mean(one**2))+1e-12))
            assert drift<1e-3,drift
            report['batch_comparisons'].append(dict(model=name,fs=fs,n=n,max_abs=diff,max_abs_over_rms=relative,prior_tf32_max_abs_over_rms=drift))
    for kind in ['raw','silver']:
        x,s,src=load_task((kind,key,str(previous)))
        folder='kaya_full' if kind=='raw' else 'kaya_silver_full'
        prior=previous/'awrls2_package/results'/folder/'cleaned'/f'{key}_PROPOSED.npy'
        old=np.load(prior);t=time.perf_counter();new,_=clean(x,200,Config(fs=200),block_s=30,overlap_s=4)
        assert np.array_equal(old,new),np.max(abs(old-new))
        report['baseline_reuse'].append(dict(kind=kind,key=key,samples=len(x),bitwise_equal=True,elapsed=time.perf_counter()-t,input_sha256=sha(src),prior_sha256=sha(prior)))
        m=real_metrics(x[:6000],new[:6000],200);ours=score(x[:6000],new[:6000],None,context(x[:6000]))
        for k in ['lf_energy_reduction_percent','outside_window_cc','outside_window_nrmse_percent']:
            assert np.isclose(m[k],ours[k],rtol=1e-12,atol=1e-12,equal_nan=True),(k,m[k],ours[k])
    (ROOT/'results/full_runner_verification.json').write_text(json.dumps(report,indent=2))
    files={str(p.relative_to(ROOT)):sha(p) for folder in ['awrls2','awrls3','models'] for p in sorted((ROOT/folder).glob('*')) if p.is_file() and p.suffix in ['.py','.npz','.pt']}
    for name in ['full_real.py','verify_full_runner.py','full_internal.py']:
        f=ROOT/'scripts'/name
        if f.exists():files[str(f.relative_to(ROOT))]=sha(f)
    payload={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,
       'scope':'Extend frozen methods to full records; no training, parameter selection or algorithm change.',
       'batch_tolerance':'max absolute difference / RMS output < 2e-6 in pre-run checks',
       'precision':'TF32 disabled in full-record inference. Drift against earlier TF32 default separately audited <1e-3 max-absolute/RMS.',
       'V2_reuse':'Previous full outputs; reexecution of one full raw and one full silver unit matches bitwise.'}
    with (ROOT/'results/FULL_RUN_FREEZE.json').open('x') as f:json.dump(payload,f,indent=2)
    print(json.dumps(report,indent=2))
