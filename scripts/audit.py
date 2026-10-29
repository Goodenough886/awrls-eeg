from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd


if __name__=='__main__':
    p=json.loads((ROOT/'results/source_split.json').read_text());audit={}
    for domain in ['eeg','eog']:
        splits=[set(p[domain][s]) for s in ['train','validation','test']]
        assert not(splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
        assert not set.union(*splits)&set(p[domain]['excluded_prior'])
        audit[domain]={s:len(p[domain][s]) for s in ['train','validation','test']}
    from awrls3.dictionary import ShapeDictionary
    d=ShapeDictionary.load(ROOT/'models/dictionary.npz')
    assert set(d.metadata['eog_indices'])<=set(p['eog']['train'])
    assert set(d.metadata['internal_indices'])<=set(p['internal']['train'])
    frozen=json.loads((ROOT/'results/FROZEN_BEFORE_TEST.json').read_text())
    for name,digest in frozen['model_sha256'].items():assert hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest()==digest
    for name,digest in frozen['code_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    test=pd.read_csv(ROOT/'results/test_models.csv');num=min(len(p['eeg']['test']),len(p['eog']['test']))
    assert len(test)==num*7*10,(len(test),num)
    assert not test.duplicated(['variant','condition','epoch']).any()
    assert np.isfinite(test[['rmse','rrmse_t','rrmse_s','cc','dbp_total']].to_numpy()).all()
    assert test.groupby(['condition','variant']).size().eq(num).all()
    kaya=pd.read_csv(ROOT/'results/kaya_30s_by_unit.csv')
    assert len(kaya)==219*10 and kaya.subject.nunique()==13
    legacy=ROOT.parent/'awrls2/awrls2_package/awrls2'
    for f in (ROOT/'awrls2').glob('*.py'):
        assert f.read_bytes()==(legacy/f.name).read_bytes(),f
    audit.update(test_rows=len(test),test_unique_pairs=num,kaya_rows=len(kaya),subjects=13,
        frozen_hashes='PASS',legacy_sources_unchanged='PASS',train_only_dictionary='PASS',
        finite_metrics='PASS',source_disjoint='PASS',subject_disjoint='NOT ESTABLISHED')
    (ROOT/'results/AUDIT.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
