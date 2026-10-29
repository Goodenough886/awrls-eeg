"""python scripts/clean_signal.py input.npy output.npy --fs 200 --method hybrid"""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from awrls3 import Config,ShapeDictionary,clean


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--fs',type=float,required=True)
    ap.add_argument('--method',choices=['classical','hybrid','tcn'],default='classical')
    ap.add_argument('--models',type=Path,default=ROOT/'models');ap.add_argument('--device',default='cpu')
    args=ap.parse_args();x=np.load(args.input,allow_pickle=False)
    if x.ndim!=1:raise ValueError('Input must be one target channel, a 1D NPY array')
    if args.output.resolve()==args.input.resolve():raise ValueError('Choose a new output path to preserve input')
    dictionary=ShapeDictionary.load(args.models/'dictionary.npz')
    cfg=Config() if args.method=='classical' else Config(background=False,confidence_gain=False)
    # Bound computation and keep the documented 30s event context on long inputs.
    from awrls2.core import _overlap_add
    diagnostics=[]
    def worker(segment):
        y,d=clean(segment,args.fs,cfg,dictionary);diagnostics.append(d);return y
    proposal_clean=_overlap_add(x,args.fs,worker,30,4) if len(x)>args.fs*30 else worker(x)
    y=proposal_clean
    if args.method!='classical':
        from awrls3.neural import NeuralCleaner
        model=NeuralCleaner(args.models/('hybrid_tcn.pt' if args.method=='hybrid' else 'standalone_tcn.pt'),args.device)
        y=model.clean(x,args.fs,x-proposal_clean)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,y)
    args.output.with_suffix('.json').write_text(json.dumps({'method':args.method,'fs':args.fs,'offline':True,
        'trained_prior':True,'blocks':diagnostics},indent=2),encoding='utf-8')
    print('Saved',args.output)


if __name__=='__main__':main()
