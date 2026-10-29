"""Run only after development audit. Existing freeze file cannot be overwritten."""
from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    out=ROOT/'results/FROZEN_BEFORE_TEST.json'
    if not (ROOT/'results/validation_models.csv').exists():raise RuntimeError('Run validation first')
    for seed in [20261029,20261030]:
        for name in ['hybrid_tcn','standalone_tcn']:
            if not (ROOT/f'models/{name}_seed{seed}.pt').exists():raise RuntimeError('Complete all repeated runs first')
    models={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'models').glob('*')) if p.is_file()}
    code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['awrls2','awrls3','scripts'] for p in sorted((ROOT/folder).glob('*.py'))}
    payload={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_sha256':models,'code_sha256':code,
        'primary_candidate':'V3_HYBRID','comparators':['V2','V3_dictionary','V3_full','TCN_CONTROL','INPUT'],
        'primary_outcome':'Mean per-source RRMSE_t averaged across six physical-power SNRs',
        'paired_unit':'Unique EEG/EOG source pair; average SNR repetitions before paired inference',
        'clean_control':'RRMSE_t plus fraction with any >1e-8 absolute change; do not equate attenuation with rejection',
        'secondary_clean_control':'Fraction with RMS modification greater than 5% of input RMS; descriptive threshold, not clinical criterion',
        'no_winner_guarantee':'All frozen candidates reported. No retuning on fresh test or Kaya.',
        'seed_repeats':[20261028,20261029,20261030],
        'limitations':['Three training seeds; no subject identity split','Kaya remains retrospective ICA silver reference']}
    with out.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    print('FROZEN',out)
