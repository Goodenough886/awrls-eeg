"""Full-data completeness, frozen provenance, waveform CRC-equivalent hashes,
and subject-level silver statistics. Does not fit or select any model.
"""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from summarize import paired


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


if __name__=='__main__':
    out=ROOT/'results';frozen=json.loads((out/'FULL_RUN_FREEZE.json').read_text())
    for path,digest in frozen['files'].items():assert sha(ROOT/path)==digest,path
    parallel=json.loads((out/'FULL_RUN_PARALLEL_EXECUTION.json').read_text())
    assert sha(ROOT/'scripts/full_real_parallel.py')==parallel['script_sha256']
    assert sha(out/'FULL_RUN_FREEZE.json')==parallel['frozen_execution_sha256']
    real=pd.read_csv(out/'full_kaya/by_unit.csv');assert len(real)==3942
    assert not real.duplicated(['kind','record','channel','variant']).any()
    assert real.groupby(['kind','variant']).size().eq(219).all()
    assert real.groupby(['kind','variant']).record.nunique().eq(73).all()
    assert real.groupby(['kind','variant']).subject.nunique().eq(13).all()
    lengths={};waves=out/'full_kaya/waveforms'
    for i,r in real.iterrows():
        f=waves/f'{r.kind}_{r.record}_{r.channel}_{r.variant}.npy'
        assert sha(f)==r.output_sha256,f
        a=np.load(f,mmap_mode='r');assert a.shape==(int(r.samples),) and a.dtype==np.float32 and np.isfinite(a).all()
        key=(r.kind,r.record,r.channel)
        if key in lengths:assert lengths[key]==len(a)
        lengths[key]=len(a)
        if (i+1)%400==0:print('verified full waveforms',i+1,'/',len(real),flush=True)
    subs=pd.read_csv(out/'full_kaya/by_subject.csv');silver=subs[subs.kind=='silver'].copy()
    silver=silver.rename(columns={c:c.removeprefix('silver_') for c in silver if c.startswith('silver_')})
    stats=paired(silver,'subject','kaya_full_silver_subject_averaged')
    stats.to_csv(out/'full_kaya/paired_statistics.csv',index=False)
    internal=pd.read_csv(out/'full_internal/by_epoch.csv');assert len(internal)==24000
    assert internal.groupby(['condition','variant']).size().eq(300).all()
    for f in (out/'full_internal').glob('*_waveforms.npz'):
        with np.load(f,allow_pickle=False) as z:
            assert len(z.files)==11
            for k in z.files:assert z[k].shape==(300,2000) and np.isfinite(z[k]).all()
    public=pd.read_csv(out/'full_public_all/by_epoch.csv')
    assert len(public)==4514*7*10
    assert public.eeg_index.nunique()==4514 and public.eog_index.nunique()==3400
    assert public[public.primary_test_member].shape[0]==38780
    files=list((out/'full_public_all').glob('*.npz'));assert len(files)==28
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            n=len(z['eeg_indices']);assert len(z['eog_indices'])==n
            for k in z.files:
                if k not in ['fs','eeg_indices','eog_indices']:assert z[k].shape==(n,512) and np.isfinite(z[k]).all()
    complete={'status':'PASS','kaya_records':73,'kaya_subjects':13,'kaya_target_channels':3,'input_recipes':2,
       'kaya_waveform_files':3942,'internal_rows':24000,'internal_conditions':8,'internal_epochs':300,
       'public_test_pairs':554,'public_test_conditions':7,'test_rows':38780,
       'public_all_eeg_sources':4514,'public_all_eog_sources':3400,'public_all_descriptive_rows':len(public),
       'frozen_models_and_code':'PASS','all_waveform_hashes_shapes_finiteness':'PASS',
       'raw_total_channel_hours':sum(v for (kind,rec,ch),v in lengths.items() if kind=='raw')/200/3600,
       'silver_total_channel_hours':sum(v for (kind,rec,ch),v in lengths.items() if kind=='silver')/200/3600,
       'no_retraining_on_test_or_real':True,'source_disjoint_not_subject_disjoint':True}
    (out/'FULL_COMPLETENESS_AUDIT.json').write_text(json.dumps(complete,indent=2))
    print(json.dumps(complete,indent=2))
