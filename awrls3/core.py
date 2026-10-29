"""Offline event-based AWRLS-3. No clean signal is accepted by this API."""
from dataclasses import replace
import time
import numpy as np
from awrls2.config import Config as LegacyConfig
from awrls2.detect import detect, lowpass, robust_sigma
from awrls2.reference import (EventReference,precompute_bank,select_reference,
                              blink_subspace,event_regressors)
from awrls2.estimate import background_psd,taper_window,guard_check
from .config import Config
from .estimate import fit,background_basis


def clean(x, fs, config=None, dictionary=None):
    cfg=config or Config()
    cfg=replace(cfg,base=cfg.base.with_updates(fs=float(fs)))
    cfg.validate()
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all():
        raise ValueError("Expected finite 1D signal with >=32 samples")
    if cfg.dictionary and dictionary is None:
        raise ValueError("dictionary=True requires a frozen ShapeDictionary")
    start_time=time.perf_counter(); base=cfg.base
    dc=base.with_updates(surrogate_calibration=False) if cfg.rescue else base
    candidates,det=detect(x,dc)
    diag={"version":"3.0.0.dev1","events":[],"detected_count":len(candidates),
          "raw_intervals":[[c.start,c.end,c.peak] for c in candidates],"offline":True}
    if not candidates:
        diag.update(processed_count=0,elapsed_s=time.perf_counter()-start_time)
        return x.copy(),diag
    bank=precompute_bank(x,base); refs=[]
    for i,c in enumerate(candidates):
        r,info,_=select_reference(x,c.start,c.end,c.peak,bank,base)
        if r is not None: refs.append(EventReference(i,c.start,c.end,c.peak,r,info))
    mask=np.zeros(len(x),bool)
    for ev in refs: mask[max(0,ev.start-int(.15*fs)):min(len(x),ev.end+int(.15*fs))]=True
    psd=background_psd(x,mask,fs) if cfg.estimator=='aligned_gls' else None
    lp=det['lowpass']; sigma=robust_sigma(lp[~mask]) if (~mask).sum()>20 else robust_sigma(lp)
    correction=np.zeros(len(x)); overlap=np.zeros(len(x)); processed=0
    for ev in refs:
        d=x[ev.start:ev.end]; n=len(d)
        if n<20: continue
        local=blink_subspace(refs,ev.index,base.template_rank,int(.5*fs)) if len(refs)>=base.template_min_events else None
        A,info=event_regressors(ev,local,base)
        match=0.0; atom_info={}
        if cfg.dictionary:
            match,v,atom_info=dictionary.match(ev.reference,ev.peak-ev.start,fs)
            if match>=cfg.template_match_min:
                A=np.column_stack([A,v,np.gradient(v)])
        B,prior,bg=background_basis(x,ev.start,ev.end,mask,fs,cfg)
        a,fitinfo=fit(d,A,B,prior,fs,cfg,psd,info.get('self_column'))
        # Band-limited correction, not filtering the reconstructed EEG.
        a=lowpass(a,fs,cfg.correction_lowpass_hz,4)*taper_window(n,base.taper_alpha)
        rho=float(np.mean(lowpass(d,fs,12,4)**2)/(sigma*sigma+1e-12))
        shape_conf=max(match,float(info.get('template_match',0)))
        gain=1.0
        if cfg.confidence_gain:
            amplitude_conf=np.clip(1-1/max(rho,1e-12),0,1)
            shape_weight=np.clip((shape_conf-.35)/.5,0,1) if shape_conf else 0.5
            gain=float(amplitude_conf*(cfg.confidence_floor+(1-cfg.confidence_floor)*shape_weight))
        a*=gain
        accepted,reason=guard_check(d,a,base)
        diag['events'].append(dict(start=ev.start,end=ev.end,peak=ev.peak,accepted=accepted,
             reason=reason,gain=gain,rho=rho,dictionary_match=match,dictionary_used=bool(cfg.dictionary and match>=cfg.template_match_min),
             local_template=local is not None,atom=atom_info,background=bg,fit=fitinfo))
        if accepted:
            correction[ev.start:ev.end]+=a; overlap[ev.start:ev.end]+=1; processed+=1
    # Never double-subtract when independently fitted windows overlap.
    correction/=np.maximum(overlap,1)
    y=x-correction
    if not np.isfinite(y).all(): raise FloatingPointError('nonfinite output')
    diag.update(processed_count=processed,elapsed_s=time.perf_counter()-start_time)
    return y,diag
