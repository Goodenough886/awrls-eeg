"""Separate warmed timing after experimental computation finishes."""
from pathlib import Path
import os,sys,time,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from awrls2 import clean as old_clean
from awrls2.config import Config as OldConfig
from awrls3 import Config,ShapeDictionary,clean
from awrls3.neural import NeuralCleaner
from data_paths import paths

if __name__=='__main__':
    torch.set_num_threads(1)
    protocol=json.loads((ROOT/'results/source_split.json').read_text());pub,ip=paths(protocol)
    X=np.load(ip/'mixed.npy');dictionary=ShapeDictionary.load(ROOT/'models/dictionary.npz');rows=[]
    for device in ['cpu','cuda']:
        hybrid=NeuralCleaner(ROOT/'models/hybrid_tcn.pt',device);standalone=NeuralCleaner(ROOT/'models/standalone_tcn.pt',device)
        for i in range(23):
            x=X[protocol['internal']['validation'][i]];fs=200
            methods={
                'V2':lambda:old_clean(x,fs,OldConfig(fs=fs))[0],
                'V3_full':lambda:clean(x,fs,Config(),dictionary)[0],
                'V3_HYBRID_end_to_end':lambda:hybrid.clean(x,fs,x-clean(x,fs,Config(background=False,confidence_gain=False),dictionary)[0]),
                'TCN_CONTROL':lambda:standalone.clean(x,fs),
            }
            for name,fn in methods.items():
                if device=='cuda':torch.cuda.synchronize()
                t=time.perf_counter();fn()
                if device=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-t
                if i>=3:rows.append(dict(variant=name,device=device if 'TCN' in name or 'HYBRID' in name else 'cpu',epoch=i,seconds=10,elapsed_s=elapsed))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/runtime.csv',index=False)
    df.groupby(['variant','device']).elapsed_s.agg(['mean','median','std']).to_csv(ROOT/'results/runtime_summary.csv')
    print(df.groupby(['variant','device']).elapsed_s.mean())
