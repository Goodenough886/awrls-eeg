"""Real, scale-invariant ridge fit with the SAME operator on data and design.

LS uses a rectangular fit-band projection. GLS differs ONLY by PSD weights.
An orthonormal rFFT plus Parseval factors makes flat-PSD GLS equal to LS.
Nuisance priors are specified in waveform units before column scaling.
"""
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.signal.windows import tukey


def spectral_system(d, R, fs, band, psd=None, floor=0.05):
    n = len(d)
    # Same taper and padding on both sides; no asymmetric detrending.
    taper = tukey(n, alpha=0.12)
    nf = 1 << int(np.ceil(np.log2(max(n, 2))))
    f = np.fft.rfftfreq(nf, 1/fs)
    w = (f <= band).astype(float)
    if psd is not None:
        p = np.interp(f, *psd)
        inside = f <= band
        p = p + max(float(np.max(p[inside])) * floor, 1e-15)
        w[inside] /= p[inside]
        w[inside] /= np.mean(w[inside])
    multiplicity = np.full(len(f), 2.0)
    multiplicity[0] = 1.0
    if nf % 2 == 0:
        multiplicity[-1] = 1.0
    w *= multiplicity
    F = np.fft.rfft(R*taper[:, None], n=nf, axis=0, norm="ortho")
    D = np.fft.rfft(d*taper, n=nf, norm="ortho")
    # Real coefficients: differentiate the real least-squares objective.
    G = np.real(F.conj().T @ (w[:, None]*F))
    b = np.real(F.conj().T @ (w*D))
    return G, b


def fit(d, A, B, prior, fs, cfg, psd=None, self_column=None):
    R = np.column_stack([A, B])
    G, rhs = spectral_system(d, R, fs, cfg.base.fit_lowpass_hz or cfg.base.lowpass_hz,
                             psd if cfg.estimator == "aligned_gls" else None,
                             cfg.base.gls_floor)
    norms = np.sqrt(np.maximum(np.diag(G), 1e-20))
    G = G / norms[:, None] / norms[None, :]
    rhs = rhs / norms
    penalty = np.r_[np.full(A.shape[1], cfg.ridge),
                    np.full(B.shape[1], cfg.background_ridge)]
    if self_column is not None:
        penalty[self_column] *= cfg.self_penalty
    # The two trend coefficients are unpenalised; oscillatory priors penalised.
    penalty[A.shape[1]:A.shape[1]+min(2, B.shape[1])] = 0
    target = np.r_[np.zeros(A.shape[1]), prior] * norms
    H = G + np.diag(penalty)
    coef = np.linalg.lstsq(H, rhs+penalty*target, rcond=1e-10)[0] / norms
    return A @ coef[:A.shape[1]], {
        "condition": float(np.linalg.cond(H)), "rank": int(np.linalg.matrix_rank(G)),
        "n_columns": R.shape[1], "coefficients": coef.tolist(),
    }


def background_basis(x, start, end, mask, fs, cfg):
    """Fit a low-dimensional oscillatory prior ONLY on nearby non-event samples.

    This is a heuristic extrapolation, not a known-clean EEG estimate.
    If context is insufficient, return trend-only nuisance columns.
    """
    n = end-start
    t = np.linspace(-1, 1, n)
    B = np.column_stack([np.ones(n), t])
    prior = [0.0, 0.0]
    info = {"background_frequencies": [], "background_context_samples": 0}
    if not cfg.background or cfg.background_frequencies == 0:
        return B, np.asarray(prior), info
    lo, hi = max(0, start-int(2*fs)), min(len(x), end+int(2*fs))
    ix = np.arange(lo, hi)
    good = ~mask[lo:hi]
    info["background_context_samples"] = int(good.sum())
    if good.sum() < fs or good.sum() < 0.2*len(good):
        return B, np.asarray(prior), info
    # Spectral candidates from the longest contiguous usable flank, no gap joining.
    edges = np.flatnonzero(np.diff(np.r_[False, good, False].astype(int)))
    a,b = max(zip(edges[::2], edges[1::2]), key=lambda p:p[1]-p[0])
    if b-a < 0.5*fs:
        return B, np.asarray(prior), info
    freq,power = welch(x[lo+a:lo+b], fs, nperseg=min(b-a, int(2*fs)), nfft=int(4*fs))
    peaks,_ = find_peaks(power)
    peaks = [p for p in peaks if 2 <= freq[p] <= 12]
    chosen = []
    for p in sorted(peaks, key=lambda p:power[p], reverse=True):
        if all(abs(freq[p]-f)>1.0 for f in chosen): chosen.append(float(freq[p]))
        if len(chosen) >= cfg.background_frequencies: break
    tc = (ix[good]-start)/fs
    te = np.arange(n)/fs
    cols=[]; ccols=[]
    for f in chosen:
        cols.extend([np.cos(2*np.pi*f*te), np.sin(2*np.pi*f*te)])
        ccols.extend([np.cos(2*np.pi*f*tc), np.sin(2*np.pi*f*tc)])
    if cols:
        C = np.column_stack([np.ones(len(tc)), (ix[good]-start)*2/max(n-1,1)-1, *ccols])
        coef = np.linalg.lstsq(C, x[ix[good]], rcond=1e-8)[0]
        B = np.column_stack([B, *cols]); prior = coef.tolist()
    info["background_frequencies"] = chosen
    return B, np.asarray(prior), info
