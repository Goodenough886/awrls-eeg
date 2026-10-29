"""Wavelet filter bank with a PyWavelets front-end and a self-contained fallback.

The production path is PyWavelets (``pywt``) - every discrete wavelet that
PyWavelets knows is available and results match the published AWRLS v1 code.

When ``pywt`` cannot be imported (e.g. an offline sandbox) a numerically
generated fallback bank is used: Haar, Daubechies dbN (N = 2..10, spectral
factorisation of the Daubechies half-band polynomial) and Symlets symN
(N = 4..8, minimum phase-nonlinearity root selection).  Coiflets,
biorthogonal and discrete-Meyer filters are NOT available in the fallback and
are silently dropped from candidate lists.  ``BACKEND`` tells which path is
active so experiment logs can record it.

Two reconstruction operators are provided:

* ``dwt_approximation``  - critically sampled DWT, periodisation mode,
  approximation-only reconstruction (as in AWRLS v1).
* ``swt_approximation``  - shift-invariant (undecimated / a-trous) approximation
  obtained from the zero-phase cascade of the upsampled low-pass filters.
  This is the translation-invariant counterpart of the DWT projection and
  removes the 2^j alignment dependence of the periodised DWT.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import comb

import numpy as np
from scipy.signal import fftconvolve

import os as _os

if _os.environ.get("AWRLS2_FORCE_FALLBACK", "0") == "1":  # testing / sandbox switch
    _pywt = None
    BACKEND = "fallback"
else:
    try:  # pragma: no cover - exercised only where pywt is installed
        import pywt as _pywt
        BACKEND = "pywt"
    except ImportError:  # pragma: no cover
        _pywt = None
        BACKEND = "fallback"


# ----------------------------------------------------------------------------
# Fallback filter generation
# ----------------------------------------------------------------------------
def _daubechies_roots(n_vanishing: int) -> np.ndarray:
    """Roots (inside unit circle) of the Daubechies polynomial for dbN."""
    n = int(n_vanishing)
    if n < 1:
        raise ValueError("Daubechies order must be >= 1")
    if n == 1:
        return np.array([], dtype=complex)
    # P(y) = sum_k C(n-1+k, k) y^k, y = (1-z)(1-1/z)/4 ; multiply by z^(n-1)
    poly = np.zeros(2 * n - 1)
    for k in range(n):
        c_k = comb(n - 1 + k, k)
        term = np.array([1.0])
        for _ in range(k):
            term = np.convolve(term, np.array([-0.25, 0.5, -0.25]))  # -(z-1)^2/4 (ascending powers in z)
        # multiply by z^(n-1-k): shift
        shifted = np.zeros(2 * n - 1)
        shifted[n - 1 - k:n - 1 - k + len(term)] = term
        poly += c_k * shifted
    roots = np.roots(poly[::-1])  # np.roots expects descending powers
    return roots


def _lowpass_from_roots(n_vanishing: int, chosen: np.ndarray) -> np.ndarray:
    """Build the low-pass filter h from ((1+z^-1)/2)^N and chosen roots."""
    h = np.array([1.0])
    for _ in range(n_vanishing):
        h = np.convolve(h, np.array([0.5, 0.5]))
    q = np.array([1.0 + 0j])
    for r in chosen:
        q = np.convolve(q, np.array([1.0, -r]))
    q = np.real_if_close(q, tol=1e6)
    q = np.real(q)
    h = np.convolve(h, q)
    h = h / np.sum(h) * np.sqrt(2.0)
    return h


def _phase_nonlinearity(h: np.ndarray) -> float:
    n_fft = 4096
    spectrum = np.fft.rfft(h, n_fft)
    omega = np.linspace(0.0, np.pi, len(spectrum))
    keep = np.abs(spectrum) > 1e-3 * np.max(np.abs(spectrum))
    phase = np.unwrap(np.angle(spectrum[keep]))
    w = omega[keep]
    slope, intercept = np.polyfit(w, phase, 1)
    return float(np.sum((phase - (slope * w + intercept)) ** 2))


@lru_cache(maxsize=None)
def _fallback_lowpass(name: str) -> np.ndarray:
    name = name.lower()
    if name == "haar" or name == "db1":
        return np.array([1.0, 1.0]) / np.sqrt(2.0)
    if name.startswith("db"):
        n = int(name[2:])
        roots = _daubechies_roots(n)
        inside = roots[np.abs(roots) < 1.0]
        return _lowpass_from_roots(n, inside)
    if name.startswith("sym"):
        n = int(name[3:])
        if n <= 3:
            return _fallback_lowpass(f"db{n}")
        roots = _daubechies_roots(n)
        inside = roots[np.abs(roots) < 1.0]
        # group inside roots into conjugate pairs / real singles
        groups: list[np.ndarray] = []
        used = np.zeros(len(inside), bool)
        for i, r in enumerate(inside):
            if used[i]:
                continue
            if abs(r.imag) < 1e-9:
                groups.append(np.array([r]))
                used[i] = True
            else:
                j = int(np.argmin(np.abs(inside - np.conj(r)) + used * 1e9))
                groups.append(np.array([r, inside[j]]))
                used[i] = used[j] = True
        best = None
        for flips in product([False, True], repeat=len(groups)):
            chosen = np.concatenate([1.0 / np.conj(g) if f else g for g, f in zip(groups, flips)])
            h = _lowpass_from_roots(n, chosen)
            score = _phase_nonlinearity(h)
            if best is None or score < best[0]:
                best = (score, h)
        return best[1]
    raise KeyError(name)


FALLBACK_NAMES = tuple(["haar"] + [f"db{n}" for n in range(2, 11)] + [f"sym{n}" for n in range(4, 9)])


class _FallbackWavelet:
    def __init__(self, name: str):
        self.name = name
        lo = _fallback_lowpass(name)
        self.rec_lo = lo
        self.rec_hi = np.array([((-1) ** k) * lo[len(lo) - 1 - k] for k in range(len(lo))])
        self.dec_lo = lo[::-1]
        self.dec_hi = self.rec_hi[::-1]
        self.dec_len = len(lo)
        self.rec_len = len(lo)
        self.orthogonal = True


# ----------------------------------------------------------------------------
# Public interface
# ----------------------------------------------------------------------------
def available_wavelets() -> tuple[str, ...]:
    if _pywt is not None:
        return tuple(_pywt.wavelist(kind="discrete"))
    return FALLBACK_NAMES


def filter_candidates(names) -> tuple[str, ...]:
    """Keep the candidate wavelets the active backend can provide (order kept)."""
    pool = set(available_wavelets())
    return tuple(n for n in names if n in pool)


def wavelet(name: str):
    if _pywt is not None:
        return _pywt.Wavelet(name)
    return _FallbackWavelet(name)


def max_level(n_samples: int, name: str) -> int:
    w = wavelet(name)
    if _pywt is not None:
        return int(_pywt.dwt_max_level(int(n_samples), w.dec_len))
    return int(np.floor(np.log2(max(n_samples, 1) / (w.dec_len - 1)))) if n_samples >= w.dec_len else 0


def _circular_correlate(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x)
    hp = np.zeros(n)
    hp[: len(h)] = h
    return np.real(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(hp))))


def _circular_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x)
    hp = np.zeros(n)
    hp[: len(h)] = h
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(hp)))


def _fallback_wavedec(x: np.ndarray, w: _FallbackWavelet, level: int):
    coeffs = []
    a = np.asarray(x, dtype=float)
    lengths = []
    for _ in range(level):
        lengths.append(len(a))
        if len(a) % 2:
            a = np.append(a, a[-1])
        ca = _circular_correlate(a, w.rec_lo)[0::2]
        cd = _circular_correlate(a, w.rec_hi)[0::2]
        coeffs.append(cd)
        a = ca
    return [a] + coeffs[::-1], lengths


def _fallback_waverec(coeffs, w: _FallbackWavelet, lengths):
    a = np.asarray(coeffs[0], dtype=float)
    for cd, n_out in zip(coeffs[1:], lengths[::-1]):
        n_even = n_out + (n_out % 2)
        up_a = np.zeros(n_even)
        up_a[0::2] = a[: n_even // 2]
        up_d = np.zeros(n_even)
        up_d[0::2] = np.asarray(cd, dtype=float)[: n_even // 2]
        a = _circular_convolve(up_a, w.rec_lo) + _circular_convolve(up_d, w.rec_hi)
        a = a[:n_out]
    return a


def dwt_approximation(x: np.ndarray, name: str, level: int, mode: str = "periodization") -> np.ndarray:
    """Critically sampled DWT, keep A_level only, reconstruct to len(x)."""
    x = np.asarray(x, dtype=float)
    if _pywt is not None:
        w = _pywt.Wavelet(name)
        coeffs = _pywt.wavedec(x, wavelet=w, mode=mode, level=int(level))
        approx_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec = _pywt.waverec(approx_only, wavelet=w, mode=mode)
        if len(rec) < len(x):
            rec = np.pad(rec, (0, len(x) - len(rec)), mode="edge")
        return np.asarray(rec[: len(x)], dtype=float)
    w = _FallbackWavelet(name)
    coeffs, lengths = _fallback_wavedec(x, w, int(level))
    approx_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    return _fallback_waverec(approx_only, w, lengths)


def _cascade(lo: np.ndarray, level: int) -> np.ndarray:
    g = np.array([1.0])
    for j in range(int(level)):
        up = np.zeros((len(lo) - 1) * (2 ** j) + 1)
        up[:: 2 ** j] = lo
        g = np.convolve(g, up)
    return g


@lru_cache(maxsize=None)
def _atrous_kernel(name: str, level: int) -> np.ndarray:
    """Zero-phase equivalent kernel of the undecimated approximation at `level`:
    analysis cascade (dec_lo) followed by synthesis cascade (rec_lo).  For orthogonal
    wavelets dec_lo is the time reverse of rec_lo and the kernel is g * g[::-1]; for
    biorthogonal (linear-phase) pairs the product is symmetric as well."""
    w = wavelet(name)
    dec = np.asarray(w.dec_lo, dtype=float)
    rec = np.asarray(w.rec_lo, dtype=float)
    dec = dec / np.sum(dec)  # DC gain 1 per stage
    rec = rec / np.sum(rec)
    kernel = np.convolve(_cascade(dec, level), _cascade(rec, level))
    kernel /= np.sum(kernel)
    # enforce exact symmetry (numerical) so the filter is zero phase
    kernel = 0.5 * (kernel + kernel[::-1])
    return kernel


def swt_approximation(x: np.ndarray, name: str, level: int) -> np.ndarray:
    """Shift-invariant approximation (undecimated cascade, symmetric extension)."""
    x = np.asarray(x, dtype=float)
    kernel = _atrous_kernel(name, int(level))
    half = (len(kernel) - 1) // 2
    if half == 0:
        return x.copy()
    pad = min(half, len(x) - 1)
    xp = np.pad(x, (half, half), mode="reflect") if pad == half else np.pad(x, (half, half), mode="edge")
    y = fftconvolve(xp, kernel, mode="same")
    return y[half: half + len(x)]


def approximation(x: np.ndarray, name: str, level: int, transform: str = "dwt") -> np.ndarray:
    if transform == "dwt":
        return dwt_approximation(x, name, level)
    if transform == "swt":
        return swt_approximation(x, name, level)
    raise ValueError(f"Unknown transform {transform!r}")


def self_test() -> dict:
    """Perfect-reconstruction and DC-gain checks for the active backend."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    report = {"backend": BACKEND}
    for name in ("haar", "db4", "sym5"):
        if name not in available_wavelets():
            continue
        if _pywt is None:
            w = _FallbackWavelet(name)
            coeffs, lengths = _fallback_wavedec(x, w, 3)
            err = float(np.max(np.abs(_fallback_waverec(coeffs, w, lengths) - x)))
            report[f"{name}_pr_error"] = err
            report[f"{name}_sum_lo"] = float(np.sum(w.rec_lo))
        report[f"{name}_swt_dc_gain"] = float(np.mean(swt_approximation(np.ones(256), name, 4)))
    return report
