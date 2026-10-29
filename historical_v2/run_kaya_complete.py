"""Full Kaya execution; original Claude algorithm, audited I/O and ICA policy.

Each file is decoded once. Raw-channel evaluation follows wj strict preprocessing.
ICA is a separately labelled multi-channel reference construction, never an input
to AWRLS. ICA fitting is decimated to bound memory; application uses all samples.
"""
from pathlib import Path
import os,sys,time,json,gc,warnings,traceback
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[v]='1'
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat
from scipy.signal import resample_poly,butter,sosfiltfilt
from fractions import Fraction
ROOT=Path(__file__).resolve().parent;PKG=ROOT/'awrls2_package'
DATA=Path(r'C:\Users\13665\Desktop\eegqz\databasic');WJ=DATA/'wj'
sys.path[:0]=[str(PKG),str(PKG/'scripts'),str(WJ/'src')]
from common import Config,environment,evaluate,write_json,sha256
from awrls2 import ablation_variants,clean
from run_real_eeg import KAYA_PREPROCESSING
from blink_repro.strict_single_channel import preprocess_target
RESULT=PKG/'results'
CHANNELS=['Fp1','Fp2','Fz'];VARIANTS=['V1_reproduction','PROPOSED']
STAGE=ROOT/'kaya_preprocessed'

def execute_staged(task):
    global VARIANTS
    stem,ch,kind,name,fs,valid=task
    VARIANTS=[name]
    if kind=='kaya':
        x=np.load(STAGE/f'{stem}_{ch}.npy',mmap_mode='r');silver=None
    else:
        x=np.load(RESULT/'silver'/f'{stem}_{ch}_raw.npy',mmap_mode='r')
        silver=np.load(RESULT/'silver'/f'{stem}_{ch}_silver.npy',mmap_mode='r')
    save_unit(stem,ch,x,fs,kind,silver,valid)
    return stem,ch,kind,name

def save_unit(stem,ch,x,fs,kind,silver=None,valid=False,source_meta=None):
    for seconds,label in [(30,'30s'),(0,'full')]:
        t=time.perf_counter()
        out=RESULT/f'{kind}_{label}';(out/'cleaned').mkdir(parents=True,exist_ok=True);(out/'units').mkdir(exist_ok=True)
        a=np.asarray(x[:int(seconds*fs)] if seconds else x,float)
        s=None if silver is None else silver[:len(a)]
        for name in VARIANTS:
            key=f'{stem}_{ch}_{name}'
            dest=out/'units'/(key+'.json')
            if dest.exists():continue
            cfg=Config(fs=fs).with_updates(**ablation_variants()[name])
            t=time.perf_counter();y,diag=clean(a,fs,cfg,block_s=30 if len(a)>30*fs else None,overlap_s=4)
            elapsed=time.perf_counter()-t
            np.save(out/'cleaned'/(key+'.npy'),y)
            m=evaluate.real_metrics(a,y,fs)
            # Preserve original proxy, add a correctly named true bandpass metric.
            m['legacy_highpass_residual_cc']=m.pop('hf_13_40_cc')
            sos=butter(4,[13,40],btype='band',fs=fs,output='sos')
            bi,bo=sosfiltfilt(sos,a),sosfiltfilt(sos,y)
            m['hf_13_40_cc']=float(np.corrcoef(bi,bo)[0,1])
            m.update(record=stem,channel=ch,variant=name,seconds=len(a)/fs,elapsed_s=elapsed,detected=diag['detected_count'],processed=diag['processed_count'],fs=fs,silver_available=valid)
            if s is not None and valid:
                pm=evaluate.paired_metrics(a,s,y,fs)
                # ICA residual is continuous, not a sparse implanted artifact.
                # Event truth/BARR/window metrics are therefore not valid here.
                keys=['mae','rmse','snr_db','delta_snr_db','cc','rrmse_t','rrmse_s','dbp_delta','dbp_theta','dbp_alpha','dbp_beta','dbp_gamma','dbp_total']
                m.update({f'silver_{k}':pm[k] for k in keys})
            write_json(dest,m)
            del y,diag,bi,bo
        print(kind,label,stem,ch,'done',round(time.perf_counter()-t,1),flush=True)

def aggregate():
    for folder in ['kaya_30s','kaya_full','kaya_silver_30s','kaya_silver_full']:
        out=RESULT/folder
        rows=[json.loads(f.read_text()) for f in sorted((out/'units').glob('*.json'))]
        if not rows:continue
        df=pd.DataFrame(rows);df.to_csv(out/'real_by_unit.csv',index=False)
        numeric=df.select_dtypes('number').columns
        for name,fn in [('median',lambda g:g.median()),('q1',lambda g:g.quantile(.25)),('q3',lambda g:g.quantile(.75))]:
            fn(df.groupby('variant')[numeric]).to_csv(out/f'real_{name}.csv')

def one_file(path,pool):
    stem=path.stem
    done=RESULT/'kaya_audit'/(stem+'.json')
    if done.exists():return
    t=time.perf_counter()
    rec=loadmat(path,simplify_cells=True,variable_names=['o'])['o']
    names=[str(n) for n in np.asarray(rec['chnames']).ravel()];fs0=float(rec['sampFreq'])
    data=np.asarray(rec['data'],float)
    if data.shape[0]==len(names) and data.shape[1]!=len(names):data=data.T
    if data.shape[1]==len(names)+1:data=data[:,:len(names)]
    assert data.shape[1]==len(names)
    audit={'file':str(path),'sha256':sha256(path),'raw_shape':data.shape,'original_fs':fs0,'raw_duration_s':len(data)/fs0,'targets':{},'silver':{}}
    for ch in CHANNELS:
        x,fs,meta=preprocess_target(data[:,names.index(ch)],fs0,ch,'Kaya2018',KAYA_PREPROCESSING,target_fs=200.)
        STAGE.mkdir(exist_ok=True)
        np.save(STAGE/f'{stem}_{ch}.npy',x)
        audit['targets'][ch]=meta
        del x
    # Silver inputs follow Claude's full-record multichannel 0.5-45 Hz recipe,
    # with common zero edges trimmed before ICA and exported alignment recorded.
    scalp=[n for n in names if n.upper() not in ('X3','X5','A1','A2','TRIGGER','MARKER')]
    ix=[names.index(n) for n in scalp]
    x=data[:,ix].T.copy();del data,rec;gc.collect()
    finite=np.all(np.isfinite(x),axis=0)
    if not finite.all():
        for row in x:
            good=np.isfinite(row);row[~good]=np.interp(np.flatnonzero(~good),np.flatnonzero(good),row[good])
    active=np.flatnonzero(np.any(np.abs(x)>1e-9,axis=0));start,end=int(active[0]),int(active[-1]+1)
    x=x[:,start:end]
    ratio=Fraction(str(200./fs0)).limit_denominator(10000)
    if fs0!=200.:x=resample_poly(x,ratio.numerator,ratio.denominator,axis=1)
    fs=fs0*ratio.numerator/ratio.denominator
    x=x-np.median(x,axis=1,keepdims=True)
    x=sosfiltfilt(butter(4,[.5,45.],btype='band',fs=fs,output='sos'),x,axis=1)
    import mne
    raw=mne.io.RawArray(x*1e-6,mne.create_info(scalp,fs,ch_types='eeg'),verbose=False)
    raw.set_montage('standard_1020',on_missing='ignore',verbose=False)
    fit_warnings=[]
    # Picard is an explicitly supported method in Claude's CLI. Use all times
    # at regular decimation for fit, then apply the spatial solution at full fs.
    ica=mne.preprocessing.ICA(n_components=min(len(scalp)-1,18),method='picard',random_state=97,max_iter=500,fit_params={'ortho':False,'extended':True},verbose=False)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');ica.fit(raw,decim=5,verbose=False)
        fit_warnings=[str(w.message) for w in ws]
    # Correlations use full-rate sources; decimation is only used for ICA fit.
    src=ica.get_sources(raw).get_data()
    front=[i for i,n in enumerate(scalp) if n.upper() in ('FP1','FP2')]
    ref=sosfiltfilt(butter(4,12,btype='low',fs=fs,output='sos'),x[front].mean(axis=0))
    corrs=[float(abs(np.corrcoef(ref,v)[0,1])) for v in src];del src,ref
    exclude=[i for i,c in enumerate(corrs) if c>=.6]
    ica.exclude=exclude
    converged=int(ica.n_iter_)<500
    valid=bool(exclude) and converged
    silver=ica.apply(raw.copy(),verbose=False).get_data()*1e6
    out=RESULT/'silver';out.mkdir(exist_ok=True)
    # Store only the required target channels; full multichannel data stays local.
    for ch in CHANNELS:
        k=scalp.index(ch)
        np.save(out/f'{stem}_{ch}_raw.npy',x[k]);np.save(out/f'{stem}_{ch}_silver.npy',silver[k])
    audit['silver']={'excluded_ics':exclude,'ic_frontal_corr':corrs,'n_iter':int(ica.n_iter_),'converged':converged,'valid_for_agreement':valid,'fit_decim':5,'method':'picard','n_components':int(ica.n_components_),'fs':fs,'n_samples':x.shape[1],'raw_start_sample':start,'raw_end_sample':end,'warnings':fit_warnings,'forced_exclusion':False,'silver_is_ground_truth':False}
    write_json(RESULT/'kaya_audit'/(stem+'_preparation.json'),audit)
    del x,silver,raw,ica;gc.collect()
    tasks=[(stem,ch,kind,name,fs,valid if kind=='kaya_silver' else False) for kind in ['kaya','kaya_silver'] for ch in CHANNELS for name in VARIANTS]
    futures=[pool.submit(execute_staged,task) for task in tasks]
    return {'futures':futures,'audit':audit,'done':done,'start':t,'path':path}

def finish_file(job):
    for future in job['futures']:
        status=future.result()
        print('UNIT COMPLETE',*status,flush=True)
    audit=job['audit'];audit['elapsed_s']=time.perf_counter()-job['start']
    write_json(job['done'],audit);aggregate()
    print('FILE COMPLETE',job['path'].stem,round(audit['elapsed_s'],1),'seconds','silver_valid',audit['silver']['valid_for_agreement'],flush=True)
    gc.collect()

def finish_or_log(job):
    try:finish_file(job)
    except Exception:
        traceback.print_exc()
        write_json(RESULT/'kaya_audit'/(job['path'].stem+'_ERROR.json'),{'file':str(job['path']),'error':traceback.format_exc()})

if __name__=='__main__':
    (RESULT/'kaya_audit').mkdir(parents=True,exist_ok=True)
    files=sorted((DATA/'Kalya').glob('*.mat'))
    for kind in ['kaya_30s','kaya_full','kaya_silver_30s','kaya_silver_full','silver']:
        write_json(RESULT/kind/'protocol.json',{'environment':environment(),'files':[str(f) for f in files],'n_records':len(files),'channels':CHANNELS,'variants':{k:ablation_variants()[k] for k in VARIANTS},'default_config':Config().__dict__,'strict_preprocessing':KAYA_PREPROCESSING,'silver_recipe':'Separate multichannel common-zero-edge-trim, per-channel median, resample 200 Hz, 0.5-45 Hz zero-phase Butterworth; Picard extended non-orthogonal ICA rank <=18, seed 97, fit decim=5 max_iter=500, full-rate apply; threshold |corr(IC, lowpassed mean(Fp1,Fp2))|>=0.6; NO forced exclusion; no ICLabel. Exclude nonconverged/no-component records from agreement.','event_metrics_on_silver':'Not reported: ICA residual support is continuous and is not implanted event truth.','runtime':'Concurrent wall times descriptive only','block_s':30,'overlap_s':4,'seconds':30 if kind.endswith('30s') else 0,'hf_metric_correction':'hf_13_40_cc uses actual Butterworth 13-40 Hz; Claude original highpass residual is preserved as legacy_highpass_residual_cc.'})
    with ProcessPoolExecutor(max_workers=6) as pool:
        pending=[]
        for i,path in enumerate(files):
            print('START',i+1,'/',len(files),path.name,flush=True)
            try:
                job=one_file(path,pool)
                if job is not None:pending.append(job)
                if len(pending)>=2:finish_or_log(pending.pop(0))
            except Exception:
                traceback.print_exc();write_json(RESULT/'kaya_audit'/(path.stem+'_ERROR.json'),{'file':str(path),'error':traceback.format_exc()})
        for job in pending:finish_or_log(job)
    aggregate();print('COMPLETE KAYA',flush=True)
