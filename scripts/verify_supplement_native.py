from pathlib import Path
import os,sys,time,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),r'C:\Users\13665\Desktop\eegqz\databasic\wj\src']
import numpy as np
from blink_repro.methods import ovme_reg as ov
from claude_frozen_baselines_original import CONFIGS
from supplement_vme_native import extract
orig=ov.VMEWorkspace.extract
rows=[]
keys=sorted((ROOT.parent/'awrls2/kaya_preprocessed').glob('*.npy'))
cases=[]
for p in keys[::max(1,len(keys)//12)]:
    x=np.load(p,mmap_mode='r');cases.append((p.stem,np.array(x[:6000])))
for i,x in enumerate(np.load(r'C:\Users\13665\Desktop\eegqz\databasic\blinkeeg\mixed.npy')[:4]):cases.append(('internal_'+str(i),x))
for label,x in cases:
    ov.VMEWorkspace.extract=orig;t=time.perf_counter();a=ov.ovme_reg_clean(x,200,CONFIGS['OVME-REG']);oldtime=time.perf_counter()-t
    ov.VMEWorkspace.extract=extract;t=time.perf_counter();b=ov.ovme_reg_clean(x,200,CONFIGS['OVME-REG']);newtime=time.perf_counter()-t
    delta=np.max(abs(a.cleaned-b.cleaned));scale=np.sqrt(np.mean(x*x))+1e-12
    row=dict(case=label,max_abs=float(delta),relative_input_rms=float(delta/scale),numpy_seconds=oldtime,native_seconds=newtime,
       same_peaks=[v['peaks'] for v in a.diagnostics['blocks']]==[v['peaks'] for v in b.diagnostics['blocks']])
    rows.append(row);print(row,flush=True)
out=ROOT/'provenance/native_validation.json'
out.write_text(json.dumps({'cases':rows,'acceptance_relative_input_rms':1e-7,'passed':all(r['relative_input_rms']<1e-7 and r['same_peaks'] for r in rows)},indent=2))
