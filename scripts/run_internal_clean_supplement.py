"""Only missing artifact-free internal controls; existing native/SNR rows reused."""
from pathlib import Path
import json,argparse
import pandas as pd
import run_supplement_full as r
from full_real import sha,write_json
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4);a=ap.parse_args()
    a.wj_src=r'C:\Users\13665\Desktop\eegqz\databasic\wj\src';a.previous=r.ROOT.parent/'awrls2';a.backend='numpy'
    out=r.ROOT/'results/supplement_20261029/internal_clean'
    for d in ['units','waveforms','block_checkpoints']:(out/d).mkdir(parents=True,exist_ok=True)
    r.init(a.wj_src,out,str(a.previous),a.backend);r.original._internal_tasks(r.ROOT,None)
    write_json(out/'protocol.json',{'runner':sha(Path(__file__)),'shared_runner':sha(Path(r.__file__)),'methods':r.METHODS,'condition':'clean','epochs':300,'backend':'numpy','input_check':'passed'})
    tasks=[('internal',m,i,'clean') for m in r.METHODS for i in range(300)]
    r.bounded_run(r.epoch_work,tasks,a,out)
    rows=[json.loads(p.read_text()) for p in (out/'units').glob('*.json')];assert len(rows)==900
    pd.DataFrame(rows).to_csv(out/'by_unit.csv',index=False);write_json(out/'COMPLETENESS.json',{'expected':900,'actual':len(rows)})
