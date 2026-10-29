"""Bitwise-equivalent buffer/pointer reuse for the verified native kernel.
No changed arithmetic, FFT, stopping rule or parameter/search settings.
"""
from pathlib import Path
import ctypes
import numpy as np
_dll=ctypes.CDLL(str(Path(__file__).with_name('supplement_vme_native.dll')))
_fn=_dll.vme_iterate
_fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
_fn.restype=ctypes.c_int

def extract(self,alpha,omega_hz,fs,*,tau=0.,tolerance=1e-7,max_iterations=300):
    if not np.all(np.isfinite([alpha,omega_hz,fs,tau,tolerance])) or alpha<=0 or fs<=0 or not 0<=omega_hz<fs/2 or tau<0 or tolerance<=0 or max_iterations<2:raise ValueError('Invalid VME parameters')
    if not hasattr(self,'_native_cache') or len(self._native_cache['history'])!=max_iterations:
        freq=np.ascontiguousarray(self.frequency);spec=np.ascontiguousarray(self.spectrum)
        mode=np.empty_like(spec);history=np.empty(max_iterations);meta=np.empty(2)
        self._native_cache={'freq':freq,'spec':spec,'mode':mode,'history':history,'meta':meta,'fft':np.zeros(self.nfft,dtype=complex),'reverse':np.arange(self.nfft//2,0,-1),
          'ptrs':tuple(a.ctypes.data for a in [freq,spec,mode,history,meta])}
    c=self._native_cache;fp,sp,mp,hp,tp=c['ptrs']
    step=_fn(len(c['mode']),fp,sp,alpha,omega_hz/fs,tau,tolerance,max_iterations,mp,hp,tp)
    mode=c['mode'];spectrum=c['fft'];half=self.nfft//2;spectrum[half:]=mode;spectrum[c['reverse']]=mode.conj();spectrum[0]=spectrum[-1].conj()
    restored=np.fft.ifft(np.fft.ifftshift(spectrum)).real
    restored=restored[self.nfft//4:3*self.nfft//4][:self.length].copy()
    if not np.isfinite(restored).all():raise FloatingPointError('Nonfinite VME')
    meta=c['meta']
    return restored,{'iterations':step,'converged':bool(meta[0]<=tolerance),'criterion':float(meta[0]),'omega_final_hz':float(meta[1]*fs),'omega_history_normalized':c['history'][:step+1].tolist()}

def enable():
    from blink_repro.methods.ovme_reg import VMEWorkspace
    VMEWorkspace.extract=extract
