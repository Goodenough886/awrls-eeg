"""Stage 2/3 - adaptive wavelet pseudo-reference and blink-shape subspace.

Stage 2 keeps the AWRLS v1 selection rule (absolute Pearson similarity gate
P >= P_min, score = P * F_E * F_f over every eligible mother wavelet x level)
but computes the approximation components with either the periodised DWT (v1)
or the shift-invariant undecimated transform (``transform="swt"``).

Stage 3 is new.  For every accepted event the selected approximation is a
*self* reference that still contains the low-frequency EEG of that very
window.  To purify it we build a leave-one-out blink-shape subspace from the
other events of the same record: aligned, polarity/amplitude normalised
references are stacked and a rank-r SVD basis is extracted.  Event-specific
regressors are obtained by searching a small time-shift / width-scale grid and
adding the derivative of the leading component as a first-order correction.
Low-frequency EEG is uncorrelated across events, so it averages out of the
subspace, while the stereotyped blink shape is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import wavelet_bank as bank
from .config import Config


def centered_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def starting_level(fs: float, target_hz: float) -> int:
    return max(1, int(np.floor(np.log2(fs / (2.0 * target_hz)))))


def approximation_upper_hz(fs: float, level: int) -> float:
    return float(fs / (2.0 ** (level + 1)))


@dataclass
class WaveletBank:
    cache: Dict[Tuple[str, int], np.ndarray]
    j_start: int
    availability: Dict[str, dict]


def precompute_bank(x: np.ndarray, cfg: Config) -> WaveletBank:
    x = np.asarray(x, dtype=float)
    j_start = starting_level(cfg.fs, cfg.wavelet_target_hz)
    names = (cfg.fixed_wavelet,) if cfg.fixed_wavelet else cfg.wavelet_candidates
    names = bank.filter_candidates(names)
    cache: Dict[Tuple[str, int], np.ndarray] = {}
    availability: Dict[str, dict] = {}
    for name in names:
        lib_max = bank.max_level(len(x), name)
        j_max = min(cfg.max_wavelet_level, lib_max)
        levels = range(j_start, j_max + 1)
        if cfg.fixed_level is not None:
            levels = [cfg.fixed_level] if j_start <= cfg.fixed_level <= max(j_max, cfg.fixed_level) else []
        availability[name] = {"library_max_level": lib_max, "effective_max_level": j_max, "eligible": len(list(levels)) > 0}
        for level in levels:
            cache[(name, int(level))] = bank.approximation(x, name, int(level), cfg.transform)
    return WaveletBank(cache, j_start, availability)


def _score(similarity: float, retention: float, upper_hz: float, cfg: Config) -> Tuple[float, float, float]:
    f_e = float(np.exp(-abs(np.log(max(retention, 1e-12)))))
    f_f = float(np.exp(-abs(np.log(max(upper_hz, 1e-12) / cfg.wavelet_target_hz))))
    if cfg.score_mode == "similarity":
        score = similarity
    elif cfg.score_mode == "no_energy":
        score = similarity * f_f
    elif cfg.score_mode == "no_frequency":
        score = similarity * f_e
    else:
        score = similarity * f_e * f_f
    return float(score), f_e, f_f


def select_reference(x: np.ndarray, start: int, end: int, peak: int, wb: WaveletBank, cfg: Config):
    """Return (reference, info, candidate_table) or (None, None, table)."""
    segment = np.asarray(x[start:end], dtype=float)
    if len(segment) < 4:
        return None, None, []
    rel_peak = peak - start
    half = max(4, len(segment) // 4)
    w0, w1 = max(0, rel_peak - half), min(len(segment), rel_peak + half + 1)
    e0 = float(np.sum(segment[w0:w1] ** 2) + 1e-12)
    table = []
    best = None
    for (name, level), full in wb.cache.items():
        local = full[start:end]
        sim = abs(centered_correlation(segment, local))
        retention = float(np.sum(local[w0:w1] ** 2) / e0)
        upper = approximation_upper_hz(cfg.fs, level)
        score, f_e, f_f = _score(sim, retention, upper, cfg)
        row = {"wavelet": name, "level": level, "similarity": sim, "energy_retention": retention,
               "energy_fidelity": f_e, "frequency_fidelity": f_f, "selection_score": score,
               "eligible": sim >= cfg.minimum_similarity}
        table.append(row)
        if sim < cfg.minimum_similarity:
            continue
        key = (score, f_f, f_e, sim)
        if best is None or key > best[0]:
            best = (key, row, local)
    if best is None:
        return None, None, table
    _, row, local = best
    reference = np.array(local, dtype=float)
    reference -= np.linspace(reference[0], reference[-1], len(reference))  # endpoint baseline
    info = {k: v for k, v in row.items() if k != "eligible"}
    info["j_start"] = wb.j_start
    return reference, info, table


# ----------------------------------------------------------------------------
# Stage 3 - leave-one-out blink-shape subspace
# ----------------------------------------------------------------------------
@dataclass
class EventReference:
    index: int
    start: int
    end: int
    peak: int
    reference: np.ndarray   # endpoint-corrected wavelet approximation on [start, end)
    info: dict


def _aligned_rows(events: List[EventReference], half: int) -> Tuple[np.ndarray, np.ndarray]:
    rows, amps = [], []
    for ev in events:
        rel = ev.peak - ev.start
        seg = np.zeros(2 * half + 1)
        lo, hi = rel - half, rel + half + 1
        src_lo, src_hi = max(0, lo), min(len(ev.reference), hi)
        seg[src_lo - lo: src_lo - lo + (src_hi - src_lo)] = ev.reference[src_lo:src_hi]
        amp = ev.reference[rel] if 0 <= rel < len(ev.reference) else (np.max(np.abs(seg)) + 1e-12)
        if abs(amp) < 1e-12:
            amp = np.max(np.abs(seg)) + 1e-12
        rows.append(seg / amp)
        amps.append(amp)
    return np.vstack(rows), np.asarray(amps)


def blink_subspace(events: List[EventReference], exclude: int, rank: int, half: int) -> Optional[np.ndarray]:
    others = [ev for ev in events if ev.index != exclude]
    if len(others) < 2:
        return None
    rows, _ = _aligned_rows(others, half)
    # robust centre: use the median template as first component, SVD of residual for the rest
    median = np.median(rows, axis=0)
    comps = [median / (np.linalg.norm(median) + 1e-12)]
    if rank > 1 and len(others) >= 3:
        resid = rows - median
        try:
            _, s, vt = np.linalg.svd(resid, full_matrices=False)
        except np.linalg.LinAlgError:
            vt, s = np.zeros((0, rows.shape[1])), np.zeros(0)
        for k in range(min(rank - 1, len(s))):
            if s[k] <= 1e-9 * (s[0] + 1e-12):
                break
            v = vt[k]
            v = v - comps[0] * np.dot(v, comps[0])  # orthogonalise against leading component
            n = np.linalg.norm(v)
            if n > 1e-9:
                comps.append(v / n)
    return np.vstack(comps)


def _resample_shape(shape: np.ndarray, scale: float) -> np.ndarray:
    """Stretch a centred shape by `scale` (width) keeping the centre sample fixed."""
    half = (len(shape) - 1) // 2
    grid = np.arange(-half, half + 1)
    src = grid / scale
    return np.interp(src, grid, shape, left=0.0, right=0.0)


def event_regressors(ev: EventReference, basis: Optional[np.ndarray], cfg: Config) -> Tuple[np.ndarray, dict]:
    """Build the regressor matrix (columns to be subtracted) for one event."""
    n = ev.end - ev.start
    rel = ev.peak - ev.start
    cols: List[np.ndarray] = []
    info: dict = {"basis": "self", "shift": 0, "width_scale": 1.0}
    if basis is not None and cfg.regressor == "template":
        half = (basis.shape[1] - 1) // 2
        max_shift = int(round(cfg.shift_search_s * cfg.fs))
        lead = basis[0]
        best = None
        for scale in cfg.width_scales:
            stretched = _resample_shape(lead, scale)
            for shift in range(-max_shift, max_shift + 1):
                col = np.zeros(n)
                for i in range(n):
                    k = i - rel - shift + half
                    if 0 <= k < len(stretched):
                        col[i] = stretched[k]
                c = abs(centered_correlation(col, ev.reference)) if np.any(col) else 0.0
                if best is None or c > best[0]:
                    best = (c, scale, shift)
        _, scale, shift = best
        info.update(basis="template", shift=int(shift), width_scale=float(scale), template_match=float(best[0]))

        def place(shape: np.ndarray) -> np.ndarray:
            s = _resample_shape(shape, scale)
            col = np.zeros(n)
            idx = np.arange(n) - rel - shift + half
            ok = (idx >= 0) & (idx < len(s))
            col[ok] = s[idx[ok]]
            return col

        for comp in basis:
            cols.append(place(comp))
        if cfg.derivative_basis:
            d = np.gradient(place(basis[0]))
            cols.append(d / (np.linalg.norm(d) + 1e-12) * (np.linalg.norm(cols[0]) + 1e-12))
        if cfg.include_self_reference:
            info["self_column"] = len(cols)
            cols.append(ev.reference.copy())
    else:
        info["self_column"] = 0
        cols.append(ev.reference.copy())
        if cfg.derivative_basis and cfg.regressor != "self_tdl":
            d = np.gradient(ev.reference)
            cols.append(d / (np.linalg.norm(d) + 1e-12) * (np.linalg.norm(ev.reference) + 1e-12))
    R = np.column_stack(cols)
    # scale columns to unit norm for conditioning (scale is absorbed by the coefficients)
    norms = np.linalg.norm(R, axis=0) + 1e-12
    R = R / norms
    info["n_columns"] = int(R.shape[1])
    return R, info
