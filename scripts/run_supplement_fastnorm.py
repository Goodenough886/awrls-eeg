"""Execution-only FP64 squared-magnitude shortcut, audited against original.
Older block checkpoints remain valid; the switch manifest identifies them.
"""
import sys,json,argparse
from pathlib import Path
import run_supplement_full as r
import supplement_vme_native as native
from supplement_vme_fastnorm import enable
from full_real import sha,write_json
native.enable=enable
if __name__=='__main__':
    for name in ['native_fastnorm_pilot.json','native_fastnorm_original_validation.json']:
        if not json.loads((r.ROOT/'provenance'/name).read_text())['passed']:raise RuntimeError('Numerical validation missing/failed')
    manifest={'same_equations':True,'floating_point_rounding_may_differ':True,
      'files':{n:sha(r.ROOT/'scripts'/n) for n in ['supplement_vme_fastnorm.py','supplement_vme_fastnorm.cpp','supplement_vme_fastnorm.dll','run_supplement_fastnorm.py']},
      'original_validation_sha256':sha(r.ROOT/'provenance/native_fastnorm_original_validation.json'),
      'prior_block_manifest_sha256':sha(r.ROOT/'provenance/native_fastnorm_prior_checkpoints.json'),
      'reason':'Replace hypot(z.real,z.imag)**2 by z.real**2+z.imag**2 in FP64. Same mathematical VME equations; unchanged HHO, budgets, FFT, detection and regression.'}
    write_json(r.ROOT/'provenance/NATIVE_FASTNORM_EXECUTION.json',manifest)
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=22);a=ap.parse_args()
    a.part='kaya';a.tag='kaya_full_ovme_native';a.methods=['OVME-REG'];a.backend='native';a.limit=None;a.kaya_seconds=0
    a.wj_src=r'C:\Users\13665\Desktop\eegqz\databasic\wj\src';a.previous=r.ROOT.parent/'awrls2';r.run(a)
