"""FBSE-EWT rhythm decomposition plus enhanced LPATV reimplementation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dct, idct
from scipy.special import j0, j1, jn_zeros
from scipy.signal.windows import tukey

from .base import MethodOutput


def _soft(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


@lru_cache(maxsize=32)
def _fused_system_denominator(n_samples: int, rho: float) -> np.ndarray:
    """Eigenvalues of I + rho*D.T@D for first-difference D.

    The path-graph Laplacian D.T@D is diagonalized by an orthonormal DCT-II.
    Caching this denominator makes every ADMM x-update an O(n log n) transform
    and avoids SciPy's platform-dependent LAPACK ``gtsv`` wrapper.
    """
    modes = np.arange(int(n_samples), dtype=float)
    laplacian_eigenvalues = 2.0 - 2.0 * np.cos(np.pi * modes / float(n_samples))
    return 1.0 + float(rho) * laplacian_eigenvalues


def _solve_fused_system(rhs: np.ndarray, rho: float) -> np.ndarray:
    rhs = np.asarray(rhs, dtype=float)
    denominator = _fused_system_denominator(len(rhs), float(rho))
    coefficients = dct(rhs, type=2, norm="ortho", workers=1)
    return idct(coefficients / denominator, type=2, norm="ortho", workers=1)


@lru_cache(maxsize=8)
def _fbse_basis(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return roots, analysis weights, and J0 synthesis basis (paper eqs. 1-3)."""
    roots = jn_zeros(0, n_samples)
    n = np.arange(n_samples, dtype=float)
    basis = j0(np.outer(n / float(n_samples), roots))
    analysis_weights = (2.0 / (n_samples ** 2 * (j1(roots) ** 2)))
    return roots, analysis_weights, basis


def fbse_analysis(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    centered = x - mean
    roots, weights, basis = _fbse_basis(len(x))
    n = np.arange(len(x), dtype=float)
    coefficients = weights * (basis.T @ (n * centered))
    frequencies = roots * fs / (2.0 * np.pi * len(x))
    return coefficients, frequencies, basis, mean


def _local_polynomial(x: np.ndarray, order: int, block: int, overlap: int) -> np.ndarray:
    n = len(x)
    block = min(max(order + 3, block), n)
    overlap = min(max(0, overlap), block - 1)
    step = block - overlap
    out = np.zeros(n, dtype=float)
    weight = np.zeros(n, dtype=float)
    for start in range(0, n, step):
        end = min(n, start + block)
        y = x[start:end]
        t = np.linspace(-1.0, 1.0, len(y))
        coeff = np.polyfit(t, y, min(order, len(y) - 1))
        fitted = np.polyval(coeff, t)
        taper = tukey(len(y), alpha=min(1.0, 2.0 * overlap / max(len(y), 1)))
        if start == 0:
            taper[: min(overlap, len(taper))] = 1.0
        if end == n:
            taper[max(0, len(taper) - overlap):] = 1.0
        out[start:end] += fitted * taper
        weight[start:end] += taper
        if end == n:
            break
    return out / np.maximum(weight, 1e-12)


def _fused_lasso_admm(y: np.ndarray, lam: float, rho: float, iterations: int, tolerance: float) -> tuple[np.ndarray, int]:
    """Solve .5||y-x||2^2 + lam||Dx||1 by ADMM using a DCT solve."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy(), 0
    x = y.copy()
    z = np.diff(x)
    u = np.zeros(n - 1, dtype=float)
    used = iterations
    for k in range(iterations):
        q = z - u
        dtq = np.empty(n, dtype=float)
        dtq[0] = -q[0]
        dtq[-1] = q[-1]
        dtq[1:-1] = q[:-1] - q[1:]
        x = _solve_fused_system(y + rho * dtq, rho)
        dx = np.diff(x)
        z_old = z
        z = _soft(dx + u, lam / rho)
        u = u + dx - z
        primal = np.linalg.norm(dx - z)
        dual = rho * np.linalg.norm(z - z_old)
        if max(primal, dual) <= tolerance * np.sqrt(n):
            used = k + 1
            break
    return x, used


def enhanced_lpatv(delta: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    order = int(cfg.get("polynomial_order", 3))
    block = max(order + 3, int(round(float(cfg.get("polynomial_block_s", 1.0)) * fs)))
    overlap = int(round(float(cfg.get("polynomial_overlap_s", 0.5)) * fs))
    lam = float(cfg.get("tv_lambda", 0.35))
    rho = float(cfg.get("admm_rho", 1.0))
    iterations = int(cfg.get("admm_iterations", 120))
    outer = int(cfg.get("outer_iterations", 3))
    tolerance = float(cfg.get("admm_tolerance", 1e-5))

    tv = np.zeros_like(delta)
    pa = np.zeros_like(delta)
    inner_iterations: list[int] = []
    for _ in range(max(1, outer)):
        pa = _local_polynomial(delta - tv, order, block, overlap)
        tv, used = _fused_lasso_admm(delta - pa, lam, rho, iterations, tolerance)
        inner_iterations.append(used)
    return pa, tv, {
        "polynomial_order": order,
        "polynomial_block_samples": block,
        "polynomial_overlap_samples": overlap,
        "tv_lambda": lam,
        "admm_rho": rho,
        "admm_iterations_used": inner_iterations,
        "outer_iterations": outer,
    }


def _clean_block(x: np.ndarray, fs: float, cfg: dict) -> MethodOutput:
    coefficients, frequencies, basis, mean = fbse_analysis(x, fs)
    upper = min(float(cfg.get("gamma_upper_hz", 75.0)), fs / 2.0)
    masks = {
        "delta": (frequencies >= 0.0) & (frequencies < 4.0),
        "theta": (frequencies >= 4.0) & (frequencies < 8.0),
        "alpha": (frequencies >= 8.0) & (frequencies < 13.0),
        "beta": (frequencies >= 13.0) & (frequencies < 30.0),
        "gamma": (frequencies >= 30.0) & (frequencies <= upper),
    }
    rhythms = {name: basis[:, mask] @ coefficients[mask] for name, mask in masks.items()}
    rhythms["delta"] = rhythms["delta"] + mean
    pa, tv, lpatv_diag = enhanced_lpatv(rhythms["delta"], fs, cfg)
    artifact = pa + tv
    cleaned_delta = rhythms["delta"] - artifact
    cleaned = cleaned_delta + rhythms["theta"] + rhythms["alpha"] + rhythms["beta"] + rhythms["gamma"]
    uncovered = x - sum(rhythms.values())
    if bool(cfg.get("preserve_uncovered_residual", True)):
        cleaned = cleaned + uncovered
    diagnostics = {
        "implementation": "paper-and-author-toolkit reimplementation",
        "frequency_boundaries_hz": [0.0, 4.0, 8.0, 13.0, 30.0, upper],
        "orders_per_band": {name: int(np.count_nonzero(mask)) for name, mask in masks.items()},
        "preserve_uncovered_residual": bool(cfg.get("preserve_uncovered_residual", True)),
        **lpatv_diag,
    }
    return MethodOutput(np.asarray(cleaned), np.asarray(artifact), diagnostics)


def fbse_ewt_lpatv_clean(x: np.ndarray, fs: float, cfg: dict | None = None) -> MethodOutput:
    cfg = cfg or {}
    x = np.asarray(x, dtype=float).ravel()
    block_samples = max(64, int(round(float(cfg.get("fbse_block_s", 5.0)) * fs)))
    overlap_samples = max(0, int(round(float(cfg.get("fbse_overlap_s", 1.0)) * fs)))
    if len(x) <= block_samples:
        return _clean_block(x, fs, cfg)

    step = max(1, block_samples - overlap_samples)
    cleaned = np.zeros_like(x)
    artifact = np.zeros_like(x)
    weight = np.zeros_like(x)
    diagnostics: list[dict] = []
    for start in range(0, len(x), step):
        end = min(len(x), start + block_samples)
        result = _clean_block(x[start:end], fs, cfg)
        win = tukey(end - start, alpha=min(1.0, 2.0 * overlap_samples / max(end - start, 1)))
        if start == 0:
            win[: min(overlap_samples, len(win))] = 1.0
        if end == len(x):
            win[max(0, len(win) - overlap_samples):] = 1.0
        cleaned[start:end] += result.cleaned * win
        artifact[start:end] += result.estimated_artifact * win
        weight[start:end] += win
        diagnostics.append({"start": start, "end": end, **result.diagnostics})
        if end == len(x):
            break
    weight = np.maximum(weight, 1e-12)
    return MethodOutput(cleaned / weight, artifact / weight, {"blocks": diagnostics})
