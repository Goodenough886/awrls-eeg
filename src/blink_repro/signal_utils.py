from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from scipy.signal.windows import tukey


EPS = np.finfo(float).eps


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)) + 1e-12)


def robust_sigma(x: np.ndarray) -> float:
    return robust_mad(x) / 0.6744897501960817


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    aa = a[mask] - np.mean(a[mask])
    bb = b[mask] - np.mean(b[mask])
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else float("nan")


def safe_filter(x: np.ndarray, fs: float, cutoff: float | tuple[float, float], btype: str, order: int = 4) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    nyq = fs / 2.0
    if btype == "bandpass":
        lo, hi = cutoff
        lo = max(float(lo), 0.01)
        hi = min(float(hi), nyq * 0.995)
        if not lo < hi:
            return np.zeros_like(x)
        wn: float | tuple[float, float] = (lo, hi)
    else:
        wn = min(float(cutoff), nyq * 0.995)
    sos = butter(order, wn, fs=fs, btype=btype, output="sos")
    pad_need = 3 * (2 * len(sos) + 1)
    if len(x) <= pad_need:
        return x.copy()
    return sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, fs: float, cutoff: float = 12.0, order: int = 4) -> np.ndarray:
    return safe_filter(x, fs, cutoff, "lowpass", order)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    return safe_filter(x, fs, (lo, hi), "bandpass", order)


def resample_signal(x: np.ndarray, original_fs: float, target_fs: float, axis: int = -1) -> np.ndarray:
    if abs(original_fs - target_fs) < 1e-9:
        return np.asarray(x, dtype=float)
    from math import gcd
    a, b = int(round(target_fs)), int(round(original_fs))
    g = gcd(a, b)
    return resample_poly(np.asarray(x, dtype=float), a // g, b // g, axis=axis)


def band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return float("nan")
    f, p = welch(x, fs=fs, nperseg=min(len(x), max(64, int(round(4 * fs)))))
    mask = (f >= lo) & (f < min(hi, fs / 2.0 + EPS))
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(trapezoid(p[mask], f[mask]))


def overlap_add(
    x: np.ndarray,
    fs: float,
    worker: Callable[[np.ndarray], np.ndarray],
    block_seconds: float,
    overlap_seconds: float,
) -> np.ndarray:
    """Apply a one-dimensional worker blockwise with tapered overlap-add."""
    x = np.asarray(x, dtype=float)
    block = max(32, int(round(block_seconds * fs)))
    overlap = max(0, min(block - 1, int(round(overlap_seconds * fs))))
    if len(x) <= block:
        return np.asarray(worker(x.copy()), dtype=float)
    step = block - overlap
    out = np.zeros_like(x, dtype=float)
    weights = np.zeros_like(x, dtype=float)
    starts = list(range(0, max(1, len(x) - overlap), step))
    for start in starts:
        end = min(len(x), start + block)
        segment = x[start:end]
        cleaned = np.asarray(worker(segment.copy()), dtype=float)
        if cleaned.shape != segment.shape:
            raise ValueError("Block worker changed the signal shape.")
        win = tukey(len(segment), alpha=min(1.0, 2.0 * overlap / max(len(segment), 1)))
        if start == 0:
            win[: min(overlap, len(win))] = 1.0
        if end == len(x):
            win[max(0, len(win) - overlap):] = 1.0
        out[start:end] += cleaned * win
        weights[start:end] += win
        if end == len(x):
            break
    return out / np.maximum(weights, 1e-12)
