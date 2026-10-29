"""New two official sessions, same frozen baselines, separate provenance."""
from pathlib import Path
import argparse,json
import run_supplement_full as runner
from full_real import sha,write_json
import pandas as pd

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--backend',default='numpy');args=ap.parse_args()
    root=runner.ROOT;base=root/'results/supplement_20261029';args.previous=base/'missing_sessions_staging';args.wj_src=r'C:\Users\13665\Desktop\eegqz\databasic\wj\src'
    out=base/'missing_sessions_baselines'
    for d in ['units','waveforms','block_checkpoints']:(out/d).mkdir(parents=True,exist_ok=True)
    keys=sorted(p.stem for p in (args.previous/'kaya_preprocessed').glob('*.npy'));assert len(keys)==6
    protocol={'keys':keys,'backend':args.backend,'methods':runner.METHODS,'configs':runner.original.CONFIGS,'outer_block_s':120,'outer_overlap_s':2,
       'runner_sha256':sha(Path(__file__)),'shared_runner_sha256':sha(Path(runner.__file__))}
    dest=out/'protocol.json'
    if dest.exists() and json.loads(dest.read_text())!=protocol:raise RuntimeError('Protocol mismatch')
    write_json(dest,protocol)
    tasks=[(m,k,key,0) for m in runner.METHODS for k in ['raw','silver'] for key in keys]
    runner.bounded_run(runner.kaya_work,tasks,args,out)
    rows=[json.loads(p.read_text()) for p in (out/'units').glob('*.json')];assert len(rows)==36
    pd.DataFrame(rows).to_csv(out/'by_unit.csv',index=False);write_json(out/'COMPLETENESS.json',{'expected':36,'actual':36})
