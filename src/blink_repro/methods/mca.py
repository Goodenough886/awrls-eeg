"""MCA with equally scaled UDWT, DST and DIRAC analysis dictionaries."""

from __future__ import annotations

import numpy as np
import pywt
from scipy.fft import dst, idst

from .base import MethodOutput


def _soft(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _pad_for_swt(x: np.ndarray, level: int) -> tuple[np.ndarray, int]:
    multiple = 2 ** level
    extra = (-len(x)) % multiple
    if extra == 0:
        return x.copy(), 0
    return np.pad(x, (0, extra), mode="reflect"), extra


def _udwt_threshold_reconstruct(x: np.ndarray, wavelet: str, level: int, threshold: float) -> np.ndarray:
    padded, extra = _pad_for_swt(x, level)
    # Unit-L2 wavelet filters match the orthonormal DST and identity atoms.
    # norm=True attenuates level-j coefficients by 2**(-j/2), making a
    # shared threshold unfairly suppress the ocular dictionary.
    coeffs = pywt.swt(padded, wavelet, level=level, trim_approx=False, norm=False)
    shrunk = [(_soft(ca, threshold), _soft(cd, threshold)) for ca, cd in coeffs]
    reconstructed = pywt.iswt(shrunk, wavelet, norm=False)
    return reconstructed[:-extra] if extra else reconstructed


def _dst_threshold_reconstruct(x: np.ndarray, threshold: float) -> np.ndarray:
    coeff = dst(x, type=2, norm="ortho")
    return idst(_soft(coeff, threshold), type=2, norm="ortho")


def mca_clean(x: np.ndarray, fs: float, cfg: dict | None = None) -> MethodOutput:
    """
    Signal-domain block relaxation based on Singh & Wagatsuma (2017).

    The paper identifies the UDWT component as ocular morphology, the DST
    component as oscillatory EEG and DIRAC as transient/spike EEG.  Therefore
    only the UDWT component is subtracted. Boundary extension, noise-scaled
    lambda and the finite iteration budget are explicit reimplementation
    choices, not a claim of numerical equivalence to the authors' code.
    """
    cfg = cfg or {}
    x = np.asarray(x, dtype=float).ravel()
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive.")
    if not np.all(np.isfinite(x)):
        raise ValueError("MCA input must contain only finite samples.")
    level = int(cfg.get("level", 4))
    wavelet = str(cfg.get("wavelet", "db4"))
    iterations = int(cfg.get("iterations", 14))
    inner_iterations = int(cfg.get("inner_iterations", 5))
    lambda_scale = float(cfg.get("lambda", 4.0))
    schedule_kind = str(cfg.get("threshold_schedule", "linear")).lower()
    if not 1 <= level <= 12:
        raise ValueError("level must be between 1 and 12.")
    if iterations < 2 or inner_iterations < 1:
        raise ValueError("iterations must be >= 2 and inner_iterations >= 1.")
    if not np.isfinite(lambda_scale) or lambda_scale <= 0:
        raise ValueError("lambda must be finite and positive.")
    if schedule_kind not in {"linear", "geometric"}:
        raise ValueError("threshold_schedule must be linear or geometric.")
    if not pywt.Wavelet(wavelet).orthogonal:
        raise ValueError("MCA requires an orthogonal wavelet for equal atom norms.")
    threshold_kind = str(cfg.get("threshold", "soft")).lower()
    if threshold_kind != "soft":
        raise ValueError("This reproducible profile implements the paper's stable soft-threshold setting only.")

    diagnostics = {
        "implementation": "scale-corrected MCA reimplementation v2",
        "dictionaries": ["UDWT", "DST", "DIRAC"],
        "wavelet": wavelet,
        "level": level,
        "threshold": threshold_kind,
        "lambda": lambda_scale,
        "lambda_units": "multiple of robust difference noise sigma",
        "iterations": iterations,
        "inner_iterations": inner_iterations,
        "threshold_schedule": schedule_kind,
        "swt_norm": False,
    }
    if x.size < 2 or np.ptp(x) == 0:
        return MethodOutput(x.copy(), np.zeros_like(x), {**diagnostics, "status": "constant_or_empty"})

    difference = np.diff(x)
    sigma = float(np.median(np.abs(difference - np.median(difference)))) / (0.6744897501960817 * np.sqrt(2.0))
    original_length = len(x)
    x, _ = _pad_for_swt(x, level)
    initial_coeffs = pywt.swt(x, wavelet, level=level, norm=False)
    lambda_min = max(lambda_scale * sigma, np.finfo(float).eps * float(np.max(np.abs(x))))
    lambda_max = max(
        max(float(np.max(np.abs(c))) for pair in initial_coeffs for c in pair),
        float(np.max(np.abs(dst(x, type=2, norm="ortho")))),
        float(np.max(np.abs(x))),
        lambda_min,
    )
    schedule_fn = np.linspace if schedule_kind == "linear" else np.geomspace
    schedule = schedule_fn(lambda_max, lambda_min, iterations)

    slow = np.zeros_like(x)
    oscillatory = np.zeros_like(x)
    spikes = np.zeros_like(x)
    residual_norm: list[float] = []

    for threshold in schedule:
        for _ in range(inner_iterations):
            residual = x - oscillatory - spikes
            slow = _udwt_threshold_reconstruct(residual, wavelet, level, float(threshold))

            residual = x - slow - spikes
            oscillatory = _dst_threshold_reconstruct(residual, float(threshold))

            residual = x - slow - oscillatory
            spikes = _soft(residual, float(threshold))
        residual_norm.append(float(np.linalg.norm(x - slow - oscillatory - spikes)))

    cleaned = (x - slow)[:original_length]
    return MethodOutput(
        cleaned=cleaned,
        estimated_artifact=slow[:original_length],
        diagnostics={
            **diagnostics,
            "noise_sigma": sigma,
            "threshold_start": lambda_max,
            "threshold_end": lambda_min,
            "residual_norm": residual_norm,
        },
    )
