"""Stage 1 - candidate blink localisation on the target channel only.

The first pass is the AWRLS v1 detector (12 Hz zero-phase low-pass, moving
energy, robust universal threshold, prominence, refined peak, amplitude gate).
Two optional improvements are added:

* ``boundary_mode="envelope"`` replaces the "first left / third right zero
  crossing" rule with an envelope-return rule (the low-frequency magnitude
  returns to the background level), padded and bounded relative to the peak.
  This removes the dependence on an arbitrary number of zero crossings and lets
  events close to the record end be processed.
* ``second_pass=True`` builds a data-driven matched filter from the first-pass
  events (median of aligned, normalised low-frequency windows) and rescues
  weaker blinks whose normalised cross-correlation with that template exceeds
  ``second_pass_ncc`` and whose amplitude exceeds ``second_pass_amp_z`` robust
  sigmas.  Only the target channel is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

from .config import Config

MAD_GAUSSIAN_NORMALIZER = 0.6744897501960817


@dataclass
class Candidate:
    start: int
    end: int
    peak: int
    source: str  # "pass1" | "pass2"


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float((np.median(np.abs(x - med)) + 1e-12) / MAD_GAUSSIAN_NORMALIZER)


def universal_lambda(n_eff: int) -> float:
    return float(np.sqrt(2.0 * np.log(max(int(n_eff), 2))))


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=float)
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    if len(x) <= 3 * (2 * len(sos) + 1):
        return x.copy()
    return sosfiltfilt(sos, x)


def zero_crossings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    signs = np.sign(x).copy()
    if np.all(signs == 0):
        return np.array([], dtype=int)
    nonzero = np.flatnonzero(signs)
    first = int(nonzero[0])
    signs[:first] = signs[first]
    for i in range(first + 1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    return np.flatnonzero(signs[:-1] * signs[1:] < 0) + 1


def _envelope_bounds(centered: np.ndarray, peak: int, sigma_bg: float, cfg: Config) -> Tuple[int, int]:
    fs = cfg.fs
    n = len(centered)
    baseline = max(cfg.boundary_kappa * sigma_bg, cfg.boundary_peak_fraction * abs(centered[peak]), 1e-12)
    max_left = int(round(cfg.boundary_max_left_s * fs))
    max_right = int(round(cfg.boundary_max_right_s * fs))
    hold = max(2, int(round(0.04 * fs)))
    left = peak
    quiet = 0
    while left > max(0, peak - max_left):
        if abs(centered[left - 1]) <= baseline:
            quiet += 1
            if quiet >= hold:
                break
        else:
            quiet = 0
        left -= 1
    right = peak
    quiet = 0
    while right < min(n - 1, peak + max_right):
        if abs(centered[right + 1]) <= baseline:
            quiet += 1
            if quiet >= hold:
                break
        else:
            quiet = 0
        right += 1
    pad = int(round(cfg.boundary_pad_s * fs))
    return max(0, left - pad), min(n, right + 1 + pad)


def _zero_crossing_bounds(crossings: np.ndarray, peak: int):
    left_c = crossings[crossings < peak]
    right_c = crossings[crossings > peak]
    if len(left_c) < 1 or len(right_c) < 3:
        return None
    return int(left_c[-1]), int(right_c[2])


def _snap_to_crossings(start: int, end: int, crossings: np.ndarray, peak: int, n: int, cfg: Config):
    """Move the envelope bounds outward to the nearest zero crossing of the LF signal
    (bounded by the maximum extent) so that every window starts and ends on the baseline."""
    fs = cfg.fs
    max_left = int(round(cfg.boundary_max_left_s * fs))
    max_right = int(round(cfg.boundary_max_right_s * fs))
    left_c = crossings[(crossings <= start) & (crossings >= peak - max_left)]
    right_c = crossings[(crossings >= end) & (crossings <= peak + max_right)]
    s = int(left_c[-1]) if len(left_c) else start
    e = int(right_c[0]) if len(right_c) else end
    return max(0, s), min(n, e + 1)


def _bounds(centered: np.ndarray, crossings: np.ndarray, peak: int, sigma_bg: float, cfg: Config):
    if cfg.boundary_mode == "zero_crossing":
        return _zero_crossing_bounds(crossings, peak)
    env = _envelope_bounds(centered, peak, sigma_bg, cfg)
    if cfg.boundary_mode == "envelope":
        return env
    if cfg.boundary_mode == "envelope_zc":
        # envelope return (no padding) then snap outward to the baseline crossings
        pad = int(round(cfg.boundary_pad_s * cfg.fs))
        core_start, core_end = env[0] + pad, env[1] - pad
        return _snap_to_crossings(core_start, max(core_start + 1, core_end), crossings, peak, len(centered), cfg)
    if cfg.boundary_mode == "union":
        zc = _zero_crossing_bounds(crossings, peak)
        if zc is None:
            return env
        return min(zc[0], env[0]), max(zc[1], env[1])
    raise ValueError(cfg.boundary_mode)


def _merge(candidates: List[Candidate], centered: np.ndarray) -> List[Candidate]:
    candidates = sorted(candidates, key=lambda c: c.start)
    merged: List[Candidate] = []
    for c in candidates:
        if not merged or c.start > merged[-1].end:
            merged.append(Candidate(c.start, c.end, c.peak, c.source))
            continue
        old = merged[-1]
        peak = c.peak if abs(centered[c.peak]) > abs(centered[old.peak]) else old.peak
        merged[-1] = Candidate(min(old.start, c.start), max(old.end, c.end), peak,
                               old.source if peak == old.peak else c.source)
    return merged


def _normalized_xcorr(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sliding zero-mean normalised cross-correlation, centred alignment."""
    t = template - np.mean(template)
    tn = np.linalg.norm(t)
    if tn < 1e-12:
        return np.zeros_like(signal)
    t = t / tn
    L = len(t)
    half = L // 2
    xp = np.pad(signal, (half, L - half - 1), mode="reflect")
    # local mean and norm via moving sums
    ones = np.ones(L)
    s1 = np.convolve(xp, ones, mode="valid")
    s2 = np.convolve(xp ** 2, ones, mode="valid")
    local_var = np.maximum(s2 - s1 ** 2 / L, 1e-12)
    corr = np.convolve(xp, t[::-1], mode="valid")  # sum x[n+m] t[m]
    return corr / np.sqrt(local_var)


def surrogate_thresholds(centered: np.ndarray, win: int, cfg: Config, seed: int = 0) -> Tuple[float, float]:
    """Null distribution of the maximum energy / amplitude from phase-randomised
    surrogates of the clipped background (same amplitude spectrum, random phases).

    Blinks are clipped away before the spectrum is taken so that the surrogates
    describe the artifact-free background; the returned quantiles bound the
    energy-peak height and the peak amplitude that background EEG alone reaches
    in a record of this length and spectrum (false-alarm control)."""
    sigma = robust_sigma(centered)
    clipped = np.clip(centered, -cfg.surrogate_clip_sigma * sigma, cfg.surrogate_clip_sigma * sigma)
    spectrum = np.abs(np.fft.rfft(clipped - np.mean(clipped)))
    rng = np.random.default_rng(seed)
    n = len(clipped)
    max_energy, max_amp = [], []
    for _ in range(max(1, cfg.surrogate_count)):
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, len(spectrum)))
        phase[0] = 1.0
        if n % 2 == 0:
            phase[-1] = 1.0
        surrogate = np.fft.irfft(spectrum * phase, n)
        max_energy.append(float(np.max(uniform_filter1d(surrogate ** 2, size=win, mode="nearest"))))
        max_amp.append(float(np.max(np.abs(surrogate))))
    q = cfg.surrogate_quantile
    return float(np.quantile(max_energy, q)), float(np.quantile(max_amp, q))


def detect(x: np.ndarray, cfg: Config) -> Tuple[List[Candidate], dict]:
    x = np.asarray(x, dtype=float)
    lp = lowpass(x, cfg.fs, cfg.lowpass_hz, cfg.lowpass_order)
    centered = lp - np.median(lp)
    win = max(3, int(round(cfg.energy_window_s * cfg.fs)))
    energy = uniform_filter1d(centered ** 2, size=win, mode="nearest")
    energy_median = float(np.median(energy))
    energy_sigma = robust_sigma(energy)
    n_eff = max(2, int(np.ceil(len(energy) / win)))
    threshold = energy_median + energy_sigma * universal_lambda(n_eff)
    signal_sigma = robust_sigma(centered)
    amplitude_threshold = signal_sigma * universal_lambda(len(centered))
    surrogate_info = {}
    if cfg.surrogate_calibration:
        sur_energy, sur_amp = surrogate_thresholds(centered, win, cfg)
        surrogate_info = {"surrogate_energy_threshold": sur_energy, "surrogate_amplitude_threshold": sur_amp,
                          "universal_energy_threshold": float(threshold), "universal_amplitude_threshold": float(amplitude_threshold)}
        threshold = max(threshold, sur_energy)
        amplitude_threshold = max(amplitude_threshold, sur_amp)
    prominence = max(energy_sigma, np.finfo(float).eps)
    min_distance = max(1, int(round(cfg.min_peak_distance_s * cfg.fs)))
    raw_peaks, _ = find_peaks(energy, height=threshold, prominence=prominence, distance=min_distance)

    crossings = zero_crossings(centered)
    refine = max(1, int(round(cfg.peak_refine_window_s * cfg.fs)))

    def build(peak: int, source: str, amp_thr: float):
        left = max(0, peak - refine)
        right = min(len(centered), peak + refine + 1)
        refined = left + int(np.argmax(np.abs(centered[left:right])))
        if abs(centered[refined]) < amp_thr:
            return None
        bounds = _bounds(centered, crossings, refined, signal_sigma, cfg)
        if bounds is None:
            return None
        start, end = bounds
        duration = (end - start) / cfg.fs
        if not (cfg.min_artifact_duration_s <= duration <= cfg.max_artifact_duration_s):
            return None
        return Candidate(start, end, refined, source)

    candidates: List[Candidate] = []
    for p in raw_peaks:
        c = build(int(p), "pass1", amplitude_threshold)
        if c is not None:
            candidates.append(c)
    candidates = _merge(candidates, centered)
    details = {
        "energy_threshold": float(threshold), "energy_sigma": float(energy_sigma),
        "signal_sigma": float(signal_sigma), "amplitude_threshold": float(amplitude_threshold),
        "pass1_count": len(candidates), "pass2_count": 0, "template_length": 0, **surrogate_info,
    }

    # ------------------------------------------------------------------
    # Second pass: data-driven matched filter built from first-pass events
    # ------------------------------------------------------------------
    if cfg.second_pass and len(candidates) >= cfg.second_pass_min_seed_events:
        half = int(round(0.25 * cfg.fs))
        rows = []
        for c in candidates:
            if c.peak - half < 0 or c.peak + half + 1 > len(centered):
                continue
            seg = centered[c.peak - half: c.peak + half + 1]
            rows.append(seg / (abs(centered[c.peak]) + 1e-12))
        if len(rows) >= cfg.second_pass_min_seed_events:
            template = np.median(np.vstack(rows), axis=0)
            ncc = _normalized_xcorr(centered, template)
            amp_thr2 = cfg.second_pass_amp_z * signal_sigma
            occupied = np.zeros(len(centered), bool)
            for c in candidates:
                occupied[max(0, c.start - min_distance): min(len(centered), c.end + min_distance)] = True
            peaks2, _ = find_peaks(ncc, height=cfg.second_pass_ncc, distance=min_distance)
            added = []
            polarity = np.sign(np.median([centered[c.peak] for c in candidates]))
            for p in peaks2:
                p = int(p)
                if occupied[p]:
                    continue
                if polarity != 0 and np.sign(centered[p]) != polarity:
                    continue
                c = build(p, "pass2", amp_thr2)
                if c is not None and not occupied[c.peak]:
                    added.append(c)
            if added:
                candidates = _merge(candidates + added, centered)
            details["pass2_count"] = len(added)
            details["template_length"] = len(template)
    details["detected_count"] = len(candidates)
    return candidates, {**details, "lowpass": lp, "centered": centered, "energy": energy}
