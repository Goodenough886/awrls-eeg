"""Complete added comparisons, audit coverage and export retrospective statistics."""
from pathlib import Path
import os,sys,json,shutil,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from summarize import paired
from full_real import sha,write_json
BASE=ROOT/'results/supplement_20261029';OUT=BASE/'delivery_results';OUT.mkdir(exist_ok=True)
LEGACY=ROOT.parent/'awrls2/awrls2_package/results'
PRIMARY=['INPUT','V2','V3_dictionary','V3_full','V3_HYBRID','TCN_CONTROL','MCA','FBSE-EWT-LPATV','OVME-REG']
METRICS=['mae','rmse','rrmse_t','rrmse_s','cc','dbp_total','changed','normalized_modification','changed_gt_5pct_input_rms']

def read_complete(folder,expected):
    r=BASE/folder
    if not (r/'COMPLETENESS.json').exists():raise RuntimeError('Not complete: '+folder)
    d=pd.read_csv(r/'by_unit.csv');assert len(d)==expected,(folder,len(d),expected)
    d['supplement_source_folder']=folder
    return d

def verify_waves(folder,d):
    for i,row in d.iterrows():
        p=BASE/folder/row.waveform
        if sha(p)!=row.output_sha256:raise AssertionError(str(p))
        a=np.load(p,mmap_mode='r');assert a.shape==(int(row.output_samples),) and a.dtype==np.float32 and np.isfinite(a).all(),str(p)
        if 'samples' in row and pd.notna(row['samples']):assert len(a)==int(row['samples'])
    return len(d)

def run():
    frozen=json.loads((ROOT/'results/FULL_RUN_FREEZE.json').read_text())
    for p,h in frozen['files'].items():assert sha(ROOT/p)==h,p
    for name in ['native_validation.json','native_fullblock_validation.json','native_fastnorm_pilot.json','native_fastnorm_original_validation.json','legacy_baseline_reuse_verification.json']:
        assert json.loads((ROOT/'provenance'/name).read_text())['passed'],name
    execution=json.loads((ROOT/'provenance/NATIVE_FASTNORM_EXECUTION.json').read_text())
    for n,h in execution['files'].items():assert sha(ROOT/'scripts'/n)==h,n
    public=read_complete('public_full',11634)
    fast=read_complete('kaya_full_fast',876)
    slow=read_complete('kaya_full_ovme_native',438)
    extra_base=read_complete('missing_sessions_baselines',36)
    extra_models=read_complete('missing_sessions_existing_models',108)
    clean=read_complete('internal_clean',900)
    for d in [public,clean]:
        assert np.isfinite(d[['mae','rmse','rrmse_t','rrmse_s','cc','dbp_total','normalized_modification']]).all().all()
    wave_count=0
    for folder,d in [('public_full',public),('kaya_full_fast',fast),('kaya_full_ovme_native',slow),('missing_sessions_baselines',extra_base),('missing_sessions_existing_models',extra_models),('internal_clean',clean)]:
        wave_count+=verify_waves(folder,d);print('VERIFIED',folder,len(d),flush=True)
    # Legacy native and rescaled implanted-data comparisons already exist.
    old=pd.read_csv(LEGACY/'baselines/baselines_by_epoch.csv')
    assert len(old)==10500
    old['variant']=old.variant.replace({'OVME-REG_stated_mse':'OVME-REG','OVME-REG_printed_mean':'OVME-REG_mean'})
    old['condition']=old.snr_level_db.map(lambda v:'native' if pd.isna(v) else str(int(v)))
    split=json.loads((ROOT/'results/source_split.json').read_text());train=set(split['internal']['train'])
    old['split_role']=old.epoch.map(lambda i:'training_seen' if i in train else 'historical_development')
    old['reused_legacy_results']=True
    for k in ['f1','recall_percent','precision_percent','barr_percent','rmse_in_window','rmse_out_window']:
        old['removed_component_'+k]=old[k]
    old['elapsed_s']=old.elapsed_ms/1000
    internal_new=pd.concat([old,clean],ignore_index=True)
    internal_new.to_csv(OUT/'internal_baselines_by_epoch.csv',index=False)
    current=pd.read_csv(ROOT/'results/full_internal/by_epoch.csv');current['condition']=current.condition.astype(str)
    original_v1=pd.read_csv(LEGACY/'ablation_all/by_epoch.csv')
    original_v1=original_v1[original_v1.variant=='V1_reproduction'].copy()
    assert len(original_v1)==300
    original_v1['condition']='native';original_v1['reused_legacy_results']=True
    original_v1['split_role']=original_v1.epoch.map(lambda i:'training_seen' if i in train else 'historical_development')
    internal=pd.concat([current,internal_new,original_v1],ignore_index=True)
    assert not internal.duplicated(['condition','epoch','variant']).any()
    internal.to_csv(OUT/'internal_all_methods_by_epoch.csv',index=False)
    internal.groupby(['split_role','condition','variant'])[METRICS].mean().to_csv(OUT/'internal_summary_by_split.csv')
    internal.groupby(['condition','variant'])[METRICS].mean().to_csv(OUT/'internal_summary_all_retrospective.csv')
    legacy_native=old[old.condition=='native']
    n=internal[internal.condition=='native']
    # Table 2: training-free comparison is retrospective on all 300 epochs.
    main=n[n.variant.isin(['INPUT','V1_reproduction','V2','MCA','FBSE-EWT-LPATV','OVME-REG','OVME-REG_mean','ITMS'])]
    main.groupby('variant')[['mae','rmse','rrmse_t','cc','dbp_total']].agg(['mean','std']).to_csv(OUT/'table2_retrospective_300.csv')
    stats=[]
    stats.append(paired(main,'epoch','internal_native_300_retrospective'))
    for cond in ['native','clean']:
        d=internal[(internal.condition==cond)&(internal.split_role=='historical_development')&internal.variant.isin(PRIMARY)]
        stats.append(paired(d,'epoch','internal_'+cond+'_150_development'))
    frozen_public=pd.read_csv(ROOT/'results/test_models.csv')
    p=pd.concat([frozen_public,public],ignore_index=True)
    assert len(p)==554*7*13 and not p.duplicated(['condition','epoch','variant']).any()
    p.to_csv(OUT/'public_all_methods_by_epoch.csv',index=False)
    p.groupby(['condition','variant'])[METRICS].mean().to_csv(OUT/'public_by_snr_summary.csv')
    contaminated=p[p.condition.str.startswith('public_snr_')]
    contaminated.groupby('variant')[METRICS].mean().to_csv(OUT/'public_contaminated_summary.csv')
    p[p.condition=='public_clean'].groupby('variant')[METRICS].mean().to_csv(OUT/'public_clean_summary.csv')
    stats.append(paired(contaminated[contaminated.variant.isin(PRIMARY)],'epoch','public_554_snr_averaged_supplement'))
    stats.append(paired(p[(p.condition=='public_clean')&p.variant.isin(PRIMARY)],'epoch','public_554_clean_supplement'))
    previous=pd.read_csv(ROOT/'results/full_kaya/by_unit.csv')
    real73=pd.concat([previous,fast,slow],ignore_index=True)
    real75=pd.concat([real73,extra_base,extra_models],ignore_index=True)
    for name,d,records in [('kaya73',real73,73),('kaya75',real75,75)]:
        assert not d.duplicated(['kind','record','channel','variant']).any()
        assert d.groupby(['kind','variant']).size().eq(records*3).all()
        assert d.groupby(['kind','variant']).record.nunique().eq(records).all()
        assert d.groupby(['kind','variant']).subject.nunique().eq(13).all()
        assert d.groupby(['kind','record','channel']).samples.nunique().eq(1).all()
        assert d.groupby(['kind','record','channel']).input_sha256.nunique().eq(1).all()
        assert np.isfinite(d[['lf_energy_reduction_percent','outside_window_cc','outside_window_nrmse_percent','hf_13_40_cc']]).all().all()
        assert np.isfinite(d.loc[d.kind=='silver',['silver_rmse','silver_rrmse_t','silver_rrmse_s','silver_cc','silver_dbp_total']]).all().all()
        cols=[c for c in d if c.startswith('silver_') and pd.api.types.is_numeric_dtype(d[c])]
        cols+=['lf_energy_reduction_percent','outside_window_cc','outside_window_nrmse_percent','hf_13_40_cc','rms_modification']
        d.to_csv(OUT/f'{name}_by_unit.csv',index=False)
        subjects=d.groupby(['kind','subject','variant'])[cols].mean().reset_index()
        subjects.to_csv(OUT/f'{name}_by_subject.csv',index=False)
        subjects.groupby(['kind','variant'])[cols].agg(['mean','std']).to_csv(OUT/f'{name}_subject_summary.csv')
        silver=subjects[(subjects.kind=='silver')&subjects.variant.isin(PRIMARY)].copy()
        silver=silver.rename(columns={c:c.removeprefix('silver_') for c in silver if c.startswith('silver_')})
        stats.append(paired(silver,'subject',name+'_full_silver_13_subjects_supplement'))
    pd.concat(stats,ignore_index=True).to_csv(OUT/'paired_statistics.csv',index=False)
    # Include the evidence omitted from the preceding compact AWRLS3 upload.
    for name in ['ablation_all','sensitivity_dev','supplementary']:
        dest=OUT/'previous_v2_evidence'/name;dest.mkdir(parents=True,exist_ok=True)
        for p in (LEGACY/name).glob('*'):
            if p.suffix in ['.csv','.json']:shutil.copy2(p,dest/p.name)
    write_json(OUT/'legacy_reuse_provenance.json',{'baseline_csv':str(LEGACY/'baselines/baselines_by_epoch.csv'),'baseline_sha256':sha(LEGACY/'baselines/baselines_by_epoch.csv'),
       'legacy_baseline_protocol':json.loads((LEGACY/'baselines/protocol.json').read_text()),'source_identical':all(p.read_bytes()==(ROOT/'awrls2'/p.name).read_bytes() for p in (LEGACY.parent/'awrls2').glob('*.py')),
       'note':'Native and six SNR conditions reused; new clean controls computed. Rescaling arithmetic differs only by floating-point association; verify spot checks separately. No stored legacy baseline waveforms were available.'})
    durations=real75[real75.variant=='V2'].groupby('kind').seconds.sum()/3600
    audit={'status':'PASS','new_waveforms_verified':wave_count,'public_pairs':554,'public_baseline_rows':len(public),'kaya_records':75,'kaya_subjects':13,'kaya_channels':3,'kaya_recipes':2,'kaya_variants':12,'kaya75_rows':len(real75),'kaya73_rows':len(real73),'channel_hours_per_recipe':durations.to_dict(),'internal_legacy_baseline_rows_reused':len(old),'new_internal_clean_rows':len(clean),'frozen_previous_model_sources_unchanged':True,'finished':time.strftime('%Y-%m-%d %H:%M:%S')}
    write_json(OUT/'COMPLETENESS_AUDIT.json',audit);print(json.dumps(audit),flush=True)

if __name__=='__main__':run()
