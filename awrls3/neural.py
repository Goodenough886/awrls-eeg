"""Optional CUDA-trained residual TCN, with frozen classical artifact proposal.

Hybrid predicts a gain on the classical proposal plus a low-pass residual.
Neural baseline uses the same TCN backbone without that proposal. Both are
offline. Normalisation uses ONLY noisy-input RMS and is inverted at output.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import firwin,resample_poly
from fractions import Fraction


class ResidualBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.conv=nn.Conv1d(channels,channels,5,padding=2*dilation,dilation=dilation)
        self.norm=nn.GroupNorm(4,channels)
        self.mix=nn.Conv1d(channels,channels,1)

    def forward(self,x):
        return x+.2*self.mix(F.silu(self.norm(self.conv(x))))


class Denoiser(nn.Module):
    def __init__(self,hybrid=True,channels=32):
        super().__init__();self.hybrid=hybrid;self.channels=channels
        self.stem=nn.Conv1d(2,channels,7,padding=3)
        self.blocks=nn.Sequential(*[ResidualBlock(channels,2**j) for j in range(7)])
        self.head=nn.Conv1d(channels,2,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if hybrid:self.head.bias.data[0]=2.0
        self.register_buffer('fir',torch.tensor(firwin(65,16,fs=256),dtype=torch.float32)[None,None,:])

    def lowpass(self,x):
        return F.conv1d(F.pad(x,(32,32),mode='reflect'),self.fir)

    def forward(self,x,proposal):
        proposal=proposal if self.hybrid else torch.zeros_like(proposal)
        raw=self.head(self.blocks(F.silu(self.stem(torch.cat([x,proposal],dim=1)))))
        if self.hybrid:
            gate=torch.sigmoid(raw[:,:1])
            correction=gate*proposal+self.lowpass(raw[:,1:2])
        else:
            gate=torch.zeros_like(x)
            correction=self.lowpass(raw[:,1:2])
        return x-correction,gate


class NeuralCleaner:
    def __init__(self,checkpoint,device='cpu'):
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=Denoiser(payload['hybrid'],payload['channels'])
        self.model.load_state_dict(payload['state_dict']);self.model.to(device).eval()
        self.device=device;self.metadata={k:v for k,v in payload.items() if k!='state_dict'}

    def clean(self,x,fs,proposal=None,block_s=10.0):
        x=np.asarray(x,float)
        if x.ndim!=1 or len(x)<32 or not np.isfinite(x).all():raise ValueError('invalid signal')
        if self.model.hybrid and proposal is None:raise ValueError('Hybrid requires classical artifact proposal')
        a=np.zeros_like(x) if proposal is None else np.asarray(proposal,float)
        if a.shape!=x.shape or not np.isfinite(a).all():raise ValueError('invalid proposal')
        ratio=Fraction(256/float(fs)).limit_denominator(10000)
        xx=resample_poly(x,ratio.numerator,ratio.denominator) if fs!=256 else x
        aa=resample_poly(a,ratio.numerator,ratio.denominator) if fs!=256 else a
        if len(xx)<65:raise ValueError('at least 65 samples at 256 Hz required')
        # Overlap-add bounded memory; avoids applying two-second crop normalisation
        # to an entire hours-long recording. Complete windows are offline.
        block=int(block_s*256);step=max(1,block//2);out=np.zeros(len(xx));weight=np.zeros(len(xx))
        with torch.inference_mode():
            for st in range(0,len(xx),step):
                en=min(st+block,len(xx))
                if en-st<65:
                    st=max(0,en-65)
                rms=max(float(np.sqrt(np.mean(xx[st:en]**2))),1e-8)
                tx=torch.tensor(xx[st:en]/rms,dtype=torch.float32,device=self.device)[None,None]
                ta=torch.tensor(aa[st:en]/rms,dtype=torch.float32,device=self.device)[None,None]
                y,_=self.model(tx,ta)
                corr=xx[st:en]-y[0,0].cpu().numpy()*rms
                w=np.maximum(np.hanning(en-st),1e-3)
                out[st:en]+=corr*w;weight[st:en]+=w
                if en==len(xx):break
        correction=out/np.maximum(weight,1e-12)
        if fs!=256:correction=resample_poly(correction,ratio.denominator,ratio.numerator)[:len(x)]
        y=x-correction
        if not np.isfinite(y).all():raise FloatingPointError('nonfinite neural output')
        return y
