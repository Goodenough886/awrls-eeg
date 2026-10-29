"""Throughput-only wrapper of the frozen NeuralCleaner window recipe.

The network, 10 s blocks, 5 s step, noisy-RMS normalisation, Hann overlap-add,
resampling, and final short-window rule are unchanged. Equal-length windows
are evaluated in batches. GPU float32 round-off is audited separately.
"""
import numpy as np
import torch
from scipy.signal import resample_poly
from fractions import Fraction


def clean_batched(cleaner,x,fs,proposal=None,block_s=10.0,batch_size=32):
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all():raise ValueError('invalid signal')
    if cleaner.model.hybrid and proposal is None:raise ValueError('Hybrid requires proposal')
    a=np.zeros_like(x) if proposal is None else np.asarray(proposal,float)
    if a.shape!=x.shape or not np.isfinite(a).all():raise ValueError('invalid proposal')
    ratio=Fraction(256/float(fs)).limit_denominator(10000)
    xx=resample_poly(x,ratio.numerator,ratio.denominator) if fs!=256 else x
    aa=resample_poly(a,ratio.numerator,ratio.denominator) if fs!=256 else a
    if len(xx)<65:raise ValueError('too short')
    block=int(block_s*256);step=max(1,block//2);out=np.zeros(len(xx));weights=np.zeros(len(xx));windows=[]
    for st in range(0,len(xx),step):
        en=min(st+block,len(xx))
        if en-st<65:st=max(0,en-65)
        windows.append((st,en))
        if en==len(xx):break
    def apply(group):
        rms=np.array([max(float(np.sqrt(np.mean(xx[st:en]**2))),1e-8) for st,en in group])
        tx=np.array([xx[st:en]/r for (st,en),r in zip(group,rms)],dtype=np.float32)
        ta=np.array([aa[st:en]/r for (st,en),r in zip(group,rms)],dtype=np.float32)
        with torch.inference_mode():
            y,_=cleaner.model(torch.tensor(tx,device=cleaner.device)[:,None],torch.tensor(ta,device=cleaner.device)[:,None])
            yy=y[:,0].cpu().numpy()
        for j,(st,en) in enumerate(group):
            # Preserve frozen wrapper's float32 multiply BEFORE float64 subtraction.
            corrected=xx[st:en]-yy[j]*float(rms[j])
            w=np.maximum(np.hanning(en-st),1e-3)
            out[st:en]+=corrected*w;weights[st:en]+=w
    group=[]
    for window in windows:
        if group and ((window[1]-window[0])!=(group[0][1]-group[0][0]) or len(group)>=batch_size):
            apply(group);group=[]
        group.append(window)
    if group:apply(group)
    corr=out/np.maximum(weights,1e-12)
    if fs!=256:corr=resample_poly(corr,ratio.denominator,ratio.numerator)[:len(x)]
    y=x-corr
    if not np.isfinite(y).all():raise FloatingPointError('nonfinite batched output')
    return y
