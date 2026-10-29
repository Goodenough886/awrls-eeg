"""Portable inference entry point; calls the unchanged frozen numerical core."""
from pathlib import Path
import argparse
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--fs',type=float,required=True)
    ap.add_argument('--method',choices=['awrls','dictionary','awrls-d','hybrid','tcn'],default='awrls')
    ap.add_argument('--device',default='cpu')
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output path')
    x=np.asarray(np.load(a.input,allow_pickle=False),dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Expected a nonempty finite 1D signal')
    if not np.isfinite(a.fs) or a.fs<=0:raise ValueError('Sample rate must be positive')
    root=Path(__file__).resolve().parent
    if a.method=='awrls':
        from awrls2 import clean,Config
        y,_=clean(x,a.fs,Config(fs=a.fs),block_s=30 if len(x)>30*a.fs else None,overlap_s=4)
    elif a.method=='tcn':
        from awrls3.neural import NeuralCleaner
        y=NeuralCleaner(root/'models/standalone_tcn.pt',a.device).clean(x,a.fs,np.zeros_like(x))
    else:
        from awrls3 import clean,Config,ShapeDictionary
        from awrls2.core import _overlap_add
        dictionary=ShapeDictionary.load(root/'models/dictionary.npz')
        cfg=Config() if a.method=='awrls-d' else Config(background=False,confidence_gain=False)
        def worker(segment):return clean(segment,a.fs,cfg,dictionary)[0]
        y=_overlap_add(x,a.fs,worker,30,4) if len(x)>30*a.fs else worker(x)
        if a.method=='hybrid':
            from awrls3.neural import NeuralCleaner
            y=NeuralCleaner(root/'models/hybrid_tcn.pt',a.device).clean(x,a.fs,x-y)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(a.output,y)
    print('Saved',a.output)

if __name__=='__main__':main()
