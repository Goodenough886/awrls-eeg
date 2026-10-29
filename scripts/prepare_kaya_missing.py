"""Restore the two official sessions absent from the historical local subset.
Reuses the exact previous preparation function in a separate staging tree.
"""
from pathlib import Path
import os,sys,json
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'awrls2'))
import run_kaya_complete as prep
DEST=ROOT/'results/supplement_20261029/missing_sessions_staging'
prep.RESULT=DEST/'awrls2_package/results';prep.STAGE=DEST/'kaya_preprocessed';prep.VARIANTS=[]
(prep.RESULT/'kaya_audit').mkdir(parents=True,exist_ok=True);prep.STAGE.mkdir(parents=True,exist_ok=True)
class NoTasks:
    def submit(self,*args):raise RuntimeError('No algorithm runs during preparation')
if __name__=='__main__':
    for p in sorted((ROOT/'provenance/kaya_missing_originals').glob('*.mat')):
        print('PREPARE',p.name,flush=True)
        job=prep.one_file(p,NoTasks())
        if job is not None:
            prep.write_json(job['done'],job['audit'])
            print('VALID',job['audit']['silver'],flush=True)
    print('DONE',DEST,flush=True)
