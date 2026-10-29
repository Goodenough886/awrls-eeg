"""Descriptive coverage of ALL public EEG/EOG sources, labelled by split role.

The primary fresh test remains 554 frozen pairs. Additional EEG test sources
reuse EOG within the test partition and are DESCRIPTIVE ONLY. Training and
historically viewed sources must never be pooled into the primary test claim.
"""
from pathlib import Path
import os,sys,json,time,hashlib,gc
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor
from evaluate_models import proposal_worker
from evaluate_classical import metrics
from data_paths import paths


if __name__=='__main__':
    import torch
    from awrls3.neural import NeuralCleaner
    torch.set_num_threads(2)
    p=json.loads((ROOT/'results/source_split.json').read_text());pub,ip=paths(p)
    E=np.load(pub/'EEG_all_epochs.npy');O=np.load(pub/'EOG_all_epochs.npy')
    out=ROOT/'results/full_public_all';out.mkdir(exist_ok=True)
    primary=pd.read_csv(ROOT/'results/test_models.csv').set_index(['condition','variant','epoch'])
    models={}
    for seed in [20261028,20261029,20261030]:
        suffix='' if seed==20261028 else f'_seed{seed}'
        for name,file in [('V3_HYBRID','hybrid_tcn'),('TCN_CONTROL','standalone_tcn')]:models[name+suffix]=NeuralCleaner(ROOT/f'models/{file}{suffix}.pt','cuda')
    roles={}
    for role in ['train','validation','test']:
        roles[role]={d:p[d][role] for d in ['eeg','eog']}
    roles['historical_prior']={d:sorted(set(range(len(data)))-set(p[d]['train'])-set(p[d]['validation'])-set(p[d]['test'])) for d,data in [('eeg',E),('eog',O)]}
    protocol={'roles':roles,'primary_test':'First 554 test pairs, original frozen metrics retained',
      'other_rows':'Descriptive coverage only, not independent testing; EOG repeated within role when needed.',
      'mixing':'Physical power SNR; clean + eog * sqrt(Pclean / Peog / 10**(snr/10))',
      'conditions':['clean',-7,-5,-3,-1,1,3],
      'model_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'models').glob('*') if f.is_file()},
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:protocol_path.write_text(json.dumps(protocol,indent=2))
    for role,indices in roles.items():
        ei=indices['eeg'];oi=indices['eog'];num=max(len(ei),len(oi))
        pairs=[(ei[i%len(ei)],oi[i%len(oi)]) for i in range(num)]
        for condition in ['clean',-7,-5,-3,-1,1,3]:
            stem=f'{role}_{condition}';csv=out/f'{stem}.csv';wave=out/f'{stem}.npz';proof=out/f'{stem}_verification.json'
            if csv.exists() and wave.exists() and proof.exists():continue
            tasks=[]
            for i,(a,b) in enumerate(pairs):
                s=E[a];o=O[b]
                x=s.copy() if condition=='clean' else s+o*np.sqrt(np.sum(s*s)/np.sum(o*o)/10**(condition/10))
                cond='public_clean' if condition=='clean' else f'public_snr_{condition}'
                tasks.append((i,x,s,256,cond))
            rows=[];waves={'input':[],'clean':[]};maxdiff=0.
            with ProcessPoolExecutor(max_workers=3) as pool:
                for j,(task,ys,rr) in enumerate(pool.map(proposal_worker,tasks,chunksize=6)):
                    i,x,s,fs,cond=task;a=x-ys['V3_dictionary']
                    for name,model in models.items():
                        t=time.perf_counter();y=model.clean(x,fs,a);ys[name]=y
                        rr.append(dict(variant=name,epoch=i,condition=cond,elapsed_s=time.perf_counter()-t,**metrics(x,s,y,fs)))
                    for row in rr:
                        row.update(split_role=role,eeg_index=pairs[i][0],eog_index=pairs[i][1],primary_test_member=role=='test' and i<554)
                        if row['primary_test_member']:
                            ref=primary.loc[(cond,row['variant'],i)]
                            for k in ['rrmse_t','cc']:
                                diff=abs(row[k]-float(ref[k]));maxdiff=max(maxdiff,diff)
                                if diff>2e-6:raise AssertionError((role,condition,i,row['variant'],k,diff))
                    rows.extend(rr);waves['input'].append(x);waves['clean'].append(s)
                    for name,y in ys.items():waves.setdefault(name,[]).append(y)
                    if (j+1)%250==0:print(role,condition,j+1,'/',num,flush=True)
            pd.DataFrame(rows).to_csv(csv,index=False)
            np.savez_compressed(wave,**{k:np.asarray(v) for k,v in waves.items()},fs=256,
                                eeg_indices=np.array([a for a,b in pairs]),eog_indices=np.array([b for a,b in pairs]))
            proof.write_text(json.dumps({'role':role,'condition':str(condition),'pairs':num,'rows':len(rows),'primary_test_max_metric_difference':maxdiff},indent=2))
            del tasks,rows,waves;gc.collect()
            print('SAVED',stem,flush=True)
    frames=[pd.read_csv(out/f'{role}_{condition}.csv') for role in roles for condition in ['clean',-7,-5,-3,-1,1,3]]
    df=pd.concat(frames,ignore_index=True);df.to_csv(out/'by_epoch.csv',index=False)
    assert set(df.eeg_index)==set(range(len(E))) and set(df.eog_index)==set(range(len(O)))
    df.groupby(['split_role','condition','variant'])[['rrmse_t','rrmse_s','cc','dbp_total','changed','changed_gt_5pct_input_rms']].mean().to_csv(out/'summary_by_role.csv')
    proofs=[json.loads(f.read_text()) for f in out.glob('*_verification.json')];assert len(proofs)==28
    (out/'COMPLETENESS.json').write_text(json.dumps({'eeg_sources':len(E),'eog_sources':len(O),'conditions':7,'variants':10,
       'rows':len(df),'all_sources_covered':True,'waveform_npz_files':28,
       'primary_test_max_metric_difference':max(v['primary_test_max_metric_difference'] for v in proofs),
       'primary_test_unchanged':True},indent=2))
    print('FULL PUBLIC COVERAGE COMPLETE',flush=True)
