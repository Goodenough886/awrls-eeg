"""Paper-level MCA reimplementation using UDWT, DST and DIRAC dictionaries."""

from __future__ import annotations

import numpy as np
import pywt
from scipy.fft import dst, idst

from ..signal_utils import robust_sigma
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
    coeffs = pywt.swt(padded, wavelet, level=level, trim_approx=False, norm=True)
    shrunk = [(_soft(ca, threshold), _soft(cd, threshold)) for ca, cd in coeffs]
    reconstructed = pywt.iswt(shrunk, wavelet, norm=True)
    return reconstructed[:-extra] if extra else reconstructed


def _dst_threshold_reconstruct(x: np.ndarray, threshold: float) -> np.ndarray:
    coeff = dst(x, type=2, norm="ortho")
    return idst(_soft(coeff, threshold), type=2, norm="ortho")


def mca_clean(x: np.ndarray, fs: float, cfg: dict | None = None) -> MethodOutput:
    """
    Block-coordinate relaxation described by Singh & Wagatsuma (2017).

    The paper identifies the UDWT component as ocular morphology, the DST
    component as oscillatory EEG and DIRAC as transient/spike EEG.  Therefore
    only the UDWT component is subtracted.
    """
    cfg = cfg or {}
    x = np.asarray(x, dtype=float).ravel()
    level = max(1, int(cfg.get("level", 4)))
    wavelet = str(cfg.get("wavelet", "db4"))
    iterations = max(2, int(cfg.get("iterations", 14)))
    lambda_scale = float(cfg.get("lambda", 4.0))
    threshold_kind = str(cfg.get("threshold", "soft")).lower()
    if threshold_kind != "soft":
        raise ValueError("This reproducible profile implements the paper's stable soft-threshold setting only.")

    sigma = robust_sigma(np.diff(x)) / np.sqrt(2.0)
    lambda_min = max(lambda_scale * sigma, 1e-10)
    lambda_max = max(float(np.max(np.abs(x))), lambda_min)
    schedule = np.geomspace(lambda_max, lambda_min, iterations)

    slow = np.zeros_like(x)
    oscillatory = np.zeros_like(x)
    spikes = np.zeros_like(x)
    residual_norm: list[float] = []

    for threshold in schedule:
        residual = x - oscillatory - spikes
        slow = _udwt_threshold_reconstruct(residual, wavelet, level, float(threshold))

        residual = x - slow - spikes
        oscillatory = _dst_threshold_reconstruct(residual, float(threshold))

        residual = x - slow - oscillatory
        spikes = _soft(residual, float(threshold))
        residual_norm.append(float(np.linalg.norm(x - slow - oscillatory - spikes)))

    cleaned = x - slow
    return MethodOutput(
        cleaned=cleaned,
        estimated_artifact=slow,
        diagnostics={
            "implementation": "paper-level reimplementation",
            "dictionaries": ["UDWT", "DST", "DIRAC"],
            "wavelet": wavelet,
            "level": level,
            "threshold": threshold_kind,
            "lambda": lambda_scale,
            "iterations": iterations,
            "residual_norm": residual_norm,
        },
    )

