"""Execution-only cached wrapper for the same frozen supplemental run."""
import sys,json,argparse
from pathlib import Path
import run_supplement_full as r
import supplement_vme_native as native
from supplement_vme_cached import enable
from full_real import sha,write_json
native.enable=enable
if __name__=='__main__':
    qa=r.ROOT/'provenance/native_cached_validation.json'
    if not json.loads(qa.read_text())['all_bitwise_equal']:raise RuntimeError('Cache validation missing/failed')
    manifest={'same_arithmetic':True,'validation_sha256':sha(qa),'cached_wrapper_sha256':sha(r.ROOT/'scripts/supplement_vme_cached.py'),'launcher_sha256':sha(Path(__file__)),'native_dll_sha256':sha(r.ROOT/'scripts/supplement_vme_native.dll')}
    write_json(r.ROOT/'provenance/NATIVE_CACHED_EXECUTION.json',manifest)
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=22);a=ap.parse_args()
    a.part='kaya';a.tag='kaya_full_ovme_native';a.methods=['OVME-REG'];a.backend='native';a.limit=None;a.kaya_seconds=0
    a.wj_src=r'C:\Users\13665\Desktop\eegqz\databasic\wj\src';a.previous=r.ROOT.parent/'awrls2'
    r.run(a)
