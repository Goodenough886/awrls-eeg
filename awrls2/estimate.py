"""Stage 4/5 - artifact estimation, gain, taper, projection and guards.

Estimators
----------
``rls``  recursive least squares over the regressor rows (for ``self_tdl`` this
         is exactly the AWRLS v1 tapped-delay-line RLS with ``rls_passes``
         sweeps; for any other regressor it is a sequential least-squares fit).
``ls``   batch ridge least squares; with ``fit_lowpass_hz`` set, the window
         and the regressors are low-passed before the fit so that EEG above
         the blink band cannot bias the coefficients (the subtracted component
         is still built from the unfiltered regressors).
``gls``  background-weighted (generalised) least squares in the frequency
         domain: the Welch spectrum S_bg(f) of the artifact-free background
         (event windows masked) weights the residual by 1/S_bg(f) inside the
         fit band and by zero above it.  Frequencies where the EEG background
         is strong are down-weighted, so the blink basis is not fitted to
         delta/theta/alpha rhythms, while high-frequency EEG is excluded
         altogether.  With a flat spectrum this reduces to band-limited ``ls``.

The nuisance columns ([1, n]) are estimated jointly but never subtracted.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.signal.windows import tukey

from .config import Config
from .detect import lowpass


# ----------------------------------------------------------------------------
# Background spectrum
# ----------------------------------------------------------------------------
def background_psd(x: np.ndarray, mask_events: np.ndarray, fs: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Welch PSD of the target channel outside event windows (segments >= 1 s)."""
    x = np.asarray(x, dtype=float)
    keep = ~np.asarray(mask_events, bool)
    edges = np.flatnonzero(np.diff(np.r_[False, keep, False].astype(np.int8)))
    min_len = int(round(1.0 * fs))
    segments = [x[s:e] - np.mean(x[s:e]) for s, e in zip(edges[0::2], edges[1::2]) if e - s >= min_len]
    if not segments:
        return None
    nper = int(round(min(2.0 * fs, max(len(s) for s in segments))))
    acc = None
    weight = 0.0
    for seg in segments:
        f, p = welch(seg, fs=fs, nperseg=min(nper, len(seg)), nfft=nper)
        acc = p * len(seg) if acc is None else acc + p * len(seg)
        weight += len(seg)
    return f, acc / weight


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------
def _ridge_solve(G: np.ndarray, b: np.ndarray, ridge: float, penalise: np.ndarray,
                 ridge_scale: float = 1.0, shape_cols: Optional[np.ndarray] = None) -> np.ndarray:
    scale = np.trace(G).real / max(G.shape[0], 1)
    weights = penalise.astype(float) * ridge * scale
    if shape_cols is not None and ridge_scale != 1.0:
        weights = weights * np.where(shape_cols, ridge_scale, 1.0)
    P = np.diag(weights)
    try:
        beta = np.linalg.solve(G + P + 1e-12 * np.eye(G.shape[0]), b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(G + P, b, rcond=None)[0]
    return np.real(beta)


def rls_fit(R: np.ndarray, d: np.ndarray, cfg: Config) -> np.ndarray:
    """Sequential RLS over rows of R; returns the final weight vector."""
    m = R.shape[1]
    w = np.zeros(m)
    P = np.eye(m) / cfg.rls_delta
    lam = cfg.rls_forgetting_factor
    for _ in range(max(1, cfg.rls_passes)):
        for n in range(len(d)):
            u = R[n]
            Pu = P @ u
            denom = lam + u @ Pu
            if abs(denom) < 1e-12:
                continue
            k = Pu / denom
            e = d[n] - w @ u
            w = w + k * e
            P = (P - np.outer(k, Pu)) / lam
            P = (P + P.T) * 0.5
    return w


def tapped_delay_matrix(reference: np.ndarray, order: int) -> np.ndarray:
    r = np.asarray(reference, dtype=float)
    r = r / (np.std(r) + 1e-12)
    n = len(r)
    order = min(order, max(1, n // 5))
    R = np.zeros((n, order))
    for k in range(order):
        R[k:, k] = r[: n - k]
    return R


def _spectral_weights(n_fft: int, fs: float, psd: Optional[Tuple[np.ndarray, np.ndarray]],
                      band_hz: float, floor: float) -> np.ndarray:
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    w = np.zeros(len(f))
    inband = f <= band_hz
    if psd is None:
        w[inband] = 1.0
        return w
    fb, sb = psd
    s = np.interp(f, fb, sb)
    s_in = s[inband]
    s_floor = floor * np.max(s_in) if np.max(s_in) > 0 else 1.0
    w[inband] = 1.0 / (s_in + s_floor)
    w /= np.mean(w[inband]) + 1e-300
    return w


def estimate_artifact(d: np.ndarray, R: np.ndarray, subtract_cols: np.ndarray, cfg: Config,
                      psd: Optional[Tuple[np.ndarray, np.ndarray]] = None, ridge_scale: float = 1.0,
                      shape_cols: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """Fit d ~ R beta and return the artifact estimate R[:, subtract] beta[subtract]."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if cfg.estimator == "rls":
        beta = rls_fit(R, d, cfg)
        info = {"estimator": "rls"}
    elif cfg.estimator == "ls":
        if cfg.fit_lowpass_hz:
            df = lowpass(d, cfg.fs, cfg.fit_lowpass_hz, 4)
            Rf = np.column_stack([lowpass(R[:, j], cfg.fs, cfg.fit_lowpass_hz, 4) for j in range(R.shape[1])])
        else:
            df, Rf = d, R
        beta = _ridge_solve(Rf.T @ Rf, Rf.T @ df, cfg.ridge, subtract_cols, ridge_scale, shape_cols)
        info = {"estimator": "ls"}
    else:  # gls in the frequency domain
        n_fft = int(2 ** np.ceil(np.log2(2 * n)))
        band = cfg.fit_lowpass_hz or cfg.lowpass_hz
        w = _spectral_weights(n_fft, cfg.fs, psd, band, cfg.gls_floor)
        # endpoint detrend + taper before zero padding: avoids the broadband
        # leakage of the padding step that would otherwise dominate the fit
        taper = tukey(n, alpha=0.25)
        dd = (d - np.linspace(d[0], d[-1], n)) * taper
        D = np.fft.rfft(dd, n_fft)
        RF = np.fft.rfft(R * taper[:, None], n_fft, axis=0)
        G = (RF.conj().T * w) @ RF
        b = (RF.conj().T * w) @ D
        beta = _ridge_solve(G, b, cfg.ridge, subtract_cols, ridge_scale, shape_cols)
        info = {"estimator": "gls", "psd_available": psd is not None}
    artifact = R[:, subtract_cols] @ beta[subtract_cols]
    info["beta"] = np.asarray(beta).tolist()
    return artifact, info


# ----------------------------------------------------------------------------
# Gain, taper, projection, guard
# ----------------------------------------------------------------------------
def apply_gain(d: np.ndarray, artifact: np.ndarray, cfg: Config, sigma_lp_bg: float) -> Tuple[np.ndarray, float]:
    if cfg.gain_mode == "unit":
        return artifact, 1.0
    d_lp = lowpass(d, cfg.fs, cfg.lowpass_hz, cfg.lowpass_order)
    a_lp = lowpass(artifact, cfg.fs, cfg.lowpass_hz, cfg.lowpass_order)
    aa = float(np.dot(a_lp, a_lp)) + 1e-12
    da = float(np.dot(d_lp, a_lp))
    if cfg.gain_mode == "bounded_ls":
        g = float(np.clip(da / aa, 0.0, 1.1))
        return artifact * g, g
    # energy_match: choose g in [0, 1] so that residual LF energy ~ background LF energy
    target = len(d) * sigma_lp_bg ** 2
    dd = float(np.dot(d_lp, d_lp))
    disc = da ** 2 - aa * (dd - target)
    g = (da - np.sqrt(disc)) / aa if disc >= 0 else da / aa
    g = float(np.clip(g, 0.0, 1.0))
    return artifact * g, g


def taper_window(n: int, alpha: float) -> np.ndarray:
    return tukey(n, alpha=alpha) if n > 2 else np.ones(n)


def guard_check(d: np.ndarray, correction: np.ndarray, cfg: Config) -> Tuple[bool, str]:
    """Accept only corrections that reduce in-window LF energy and are LF-dominated."""
    if not cfg.guard:
        return True, "guard_off"
    if not np.all(np.isfinite(correction)) or np.linalg.norm(correction) < 1e-12:
        return False, "degenerate_correction"
    d_lp = lowpass(d, cfg.fs, cfg.lowpass_hz, cfg.lowpass_order)
    c_lp = lowpass(correction, cfg.fs, cfg.lowpass_hz, cfg.lowpass_order)
    before = float(np.sum(d_lp ** 2))
    after = float(np.sum((d_lp - c_lp) ** 2))
    if after >= before:
        return False, "no_lf_energy_reduction"
    hf = float(np.sum((correction - c_lp) ** 2)) / (float(np.sum(correction ** 2)) + 1e-12)
    if hf > cfg.guard_hf_fraction:
        return False, "correction_not_lowfrequency"
    return True, "accepted"
