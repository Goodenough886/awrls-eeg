"""Frozen, train-only shape dictionary. Inference never receives clean targets."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.signal import find_peaks
from awrls2.detect import lowpass


@dataclass
class ShapeDictionary:
    shapes: np.ndarray
    fs: float
    metadata: dict

    def save(self, path):
        np.savez_compressed(path, shapes=self.shapes, fs=self.fs,
                            metadata=json.dumps(self.metadata, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            shapes=z['shapes'].copy(); fs=float(z['fs']); meta=json.loads(str(z['metadata']))
        if shapes.ndim != 2 or not np.isfinite(shapes).all() or shapes.shape[1] % 2 != 1:
            raise ValueError("Invalid dictionary")
        return cls(shapes,fs,meta)

    @classmethod
    def train(cls, artifacts, fs, n_shapes=12, metadata=None):
        half=int(round(.6*fs)); rows=[]
        for a in artifacts:
            a=lowpass(np.asarray(a,float),fs,12,4)
            peaks,_=find_peaks(np.abs(a), distance=max(1,int(.4*fs)),
                               height=max(np.max(np.abs(a))*.35,1e-10))
            for p in peaks:
                if p-half<0 or p+half>=len(a): continue
                v=a[p-half:p+half+1].copy()
                v-=np.linspace(v[0],v[-1],len(v))
                if abs(v[half])<1e-10: continue
                v/=v[half]
                if np.linalg.norm(v)>0: rows.append(v/np.linalg.norm(v))
        if len(rows)<n_shapes: raise ValueError("Insufficient training shapes")
        X=np.asarray(rows)
        # Deterministic clustering; medoids retain observed morphology.
        centers,labels=kmeans2(X,n_shapes,minit='++',iter=30,seed=20261028)
        medoids=[]
        for j,c in enumerate(centers):
            group=X[labels==j]
            if len(group): medoids.append(group[np.argmin(np.sum((group-c)**2,axis=1))])
        return cls(np.array(medoids),fs,dict(metadata or {},n_training_shapes=len(X)))

    def match(self, reference, peak, fs):
        n=len(reference); tt=(np.arange(n)-peak)/fs
        native=(np.arange(self.shapes.shape[1])-self.shapes.shape[1]//2)/self.fs
        ref=reference-reference.mean(); ref_norm=np.linalg.norm(ref)+1e-12
        best=(-1.0,None,None)
        for k,shape in enumerate(self.shapes):
            for scale in (.8,1.0,1.25):
                for shift in (-.025,0,.025):
                    v=np.interp((tt-shift)/scale,native,shape,left=0,right=0)
                    vv=v-v.mean(); corr=float(vv@ref/(np.linalg.norm(vv)*ref_norm+1e-12))
                    if abs(corr)>best[0]: best=(abs(corr),v,dict(atom=k,width=scale,shift=shift))
        return best
