"""Optional FP64 compiled VME iteration backend, used only after equivalence QA.

HHO, RNG, tolerances, budgets, objective, FFT, detection and regression are
unchanged. Floating reduction order differs; record validation and backend.
"""
from pathlib import Path
import ctypes
import numpy as np
_dll=ctypes.CDLL(str(Path(__file__).with_suffix('.dll')))
_fn=_dll.vme_iterate
_ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
_fn.argtypes=[ctypes.c_int,_ptr,_ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,_ptr,_ptr,_ptr]
_fn.restype=ctypes.c_int

def extract(self,alpha,omega_hz,fs,*,tau=0.,tolerance=1e-7,max_iterations=300):
    if not np.all(np.isfinite([alpha,omega_hz,fs,tau,tolerance])) or alpha<=0 or fs<=0 or not 0<=omega_hz<fs/2 or tau<0 or tolerance<=0 or max_iterations<2:
        raise ValueError('Invalid VME parameters')
    mode=np.empty_like(self.spectrum);history=np.empty(max_iterations);meta=np.empty(2)
    step=_fn(len(mode),np.ascontiguousarray(self.frequency),np.ascontiguousarray(self.spectrum).view(float),alpha,omega_hz/fs,tau,tolerance,max_iterations,mode.view(float),history,meta)
    spectrum=np.zeros(self.nfft,dtype=complex);half=self.nfft//2;spectrum[half:]=mode
    spectrum[np.arange(half,0,-1)]=mode.conj();spectrum[0]=spectrum[-1].conj()
    restored=np.fft.ifft(np.fft.ifftshift(spectrum)).real
    restored=restored[self.nfft//4:3*self.nfft//4][:self.length].copy()
    if not np.isfinite(restored).all():raise FloatingPointError('Nonfinite VME')
    return restored,{'iterations':step,'converged':bool(meta[0]<=tolerance),'criterion':float(meta[0]),'omega_final_hz':float(meta[1]*fs),'omega_history_normalized':history[:step+1].tolist()}

def enable():
    from blink_repro.methods.ovme_reg import VMEWorkspace
    VMEWorkspace.extract=extract
