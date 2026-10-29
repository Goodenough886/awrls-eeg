"""AWRLS-2 pipeline: detection -> adaptive wavelet reference -> blink-shape
subspace -> background-whitened estimation -> gain/taper/guard -> subtraction.

Strictly single channel: every quantity is computed from the target vector.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal.windows import tukey

from . import wavelet_bank as bank
from .config import Config
from .detect import Candidate, detect, lowpass, robust_sigma
from .estimate import (apply_gain, background_psd, estimate_artifact, guard_check,
                       tapped_delay_matrix, taper_window)
from .reference import EventReference, blink_subspace, event_regressors, precompute_bank, select_reference


def _event_mask(n: int, events, pad: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for ev in events:
        m[max(0, ev.start - pad): min(n, ev.end + pad)] = True
    return m


def process_epoch(x: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg.validate()
    t0 = time.perf_counter()
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("process_epoch expects a one-dimensional target channel")
    output = x.copy()
    n = len(x)
    diagnostics: Dict[str, Any] = {"backend": bank.BACKEND, "n_samples": n, "events": [], "skipped": []}

    candidates, det = detect(x, cfg)
    diagnostics["detection"] = {k: v for k, v in det.items() if k not in ("lowpass", "centered", "energy")}
    diagnostics["detected_intervals_raw"] = [(c.start, c.end, c.peak) for c in candidates]
    if not candidates:
        diagnostics.update(processed_count=0, detected_count=0, skipped_count=0,
                           elapsed_s=time.perf_counter() - t0)
        return output, diagnostics

    # ---------------- Stage 2: per-event adaptive wavelet reference ----------
    wb = precompute_bank(x, cfg)
    refs: List[EventReference] = []
    for i, c in enumerate(candidates):
        if c.end - c.start < 20:
            diagnostics["skipped"].append({"start": c.start, "end": c.end, "peak": c.peak, "reason": "segment_too_short"})
            continue
        reference, info, _ = select_reference(x, c.start, c.end, c.peak, wb, cfg)
        if reference is None:
            diagnostics["skipped"].append({"start": c.start, "end": c.end, "peak": c.peak,
                                           "reason": "no_wavelet_level_satisfies_Pmin"})
            continue
        refs.append(EventReference(i, c.start, c.end, c.peak, reference, {**info, "source": c.source}))

    # ---------------- Stage 3/4 preparation -----------------------------------
    pad = int(round(0.2 * cfg.fs))
    mask = _event_mask(n, refs, pad)
    psd = background_psd(x, mask, cfg.fs) if cfg.estimator == "gls" else None
    lp = det["lowpass"]
    sigma_lp_bg = robust_sigma((lp - np.median(lp))[~mask]) if np.any(~mask) else robust_sigma(lp - np.median(lp))
    half = int(round(0.5 * cfg.fs))
    use_template = cfg.regressor == "template" and len(refs) >= cfg.template_min_events
    diagnostics["template_active"] = bool(use_template)
    diagnostics["background_psd_available"] = psd is not None

    processed = 0
    correction_total = np.zeros(n)
    for ev in refs:
        d = x[ev.start:ev.end].copy()
        basis = blink_subspace(refs, ev.index, cfg.template_rank, half) if use_template else None
        if cfg.regressor == "self_tdl":
            R = tapped_delay_matrix(ev.reference, cfg.rls_order)
            reg_info = {"basis": "self_tdl", "n_columns": int(R.shape[1])}
        else:
            R, reg_info = event_regressors(ev, basis, cfg)
        subtract = np.ones(R.shape[1], bool)
        # columns other than the event's own reference are "shape" columns (adaptive ridge target)
        shape_cols = np.ones(R.shape[1], bool)
        if reg_info.get("self_column") is not None:
            shape_cols[reg_info["self_column"]] = False
        if cfg.nuisance_trend:
            t = np.linspace(-1.0, 1.0, len(d))
            poly = np.column_stack([t ** k for k in range(max(0, cfg.nuisance_order) + 1)])
            R = np.column_stack([R, poly])
            subtract = np.r_[subtract, np.zeros(poly.shape[1], bool)]
            shape_cols = np.r_[shape_cols, np.zeros(poly.shape[1], bool)]
        ridge_scale = 1.0
        if cfg.ridge_adaptive and cfg.ridge_adaptive_kappa > 0:
            d_lp = lowpass(d, cfg.fs, cfg.lowpass_hz, cfg.lowpass_order)
            rho = float(np.sum(d_lp ** 2)) / (len(d) * sigma_lp_bg ** 2 + 1e-12)
            # effective shape penalty = ridge + kappa / rho  (expressed as a multiple of ridge)
            ridge_scale = 1.0 + (cfg.ridge_adaptive_kappa / max(rho, 1e-6)) / max(cfg.ridge, 1e-12)
        artifact, est_info = estimate_artifact(d, R, subtract, cfg, psd, ridge_scale=ridge_scale,
                                               shape_cols=shape_cols)
        artifact, gain = apply_gain(d, artifact, cfg, sigma_lp_bg)
        correction = taper_window(len(d), cfg.taper_alpha) * artifact
        if cfg.correction_lowpass_hz:
            correction = lowpass(correction, cfg.fs, cfg.correction_lowpass_hz, 4) * taper_window(len(d), cfg.taper_alpha)
        ok, reason = guard_check(d, correction, cfg)
        row = {"start": ev.start, "end": ev.end, "peak": ev.peak, "wavelet": ev.info["wavelet"],
               "level": ev.info["level"], "similarity": ev.info["similarity"],
               "selection_score": ev.info["selection_score"], "source": ev.info["source"],
               "gain": gain, "accepted": ok, "reason": reason, "ridge_scale": ridge_scale, **reg_info,
               "estimator": est_info.get("estimator")}
        if not ok:
            diagnostics["skipped"].append(row)
            continue
        correction_total[ev.start:ev.end] += correction
        processed += 1
        diagnostics["events"].append(row)

    output = x - correction_total
    diagnostics.update(detected_count=len(candidates), processed_count=processed,
                       skipped_count=len(diagnostics["skipped"]), elapsed_s=time.perf_counter() - t0,
                       config=asdict(cfg))
    return output, diagnostics


def _overlap_add(x: np.ndarray, fs: float, worker, block_s: float, overlap_s: float) -> np.ndarray:
    block = max(32, int(round(block_s * fs)))
    overlap = max(0, min(block - 1, int(round(overlap_s * fs))))
    if len(x) <= block:
        return worker(x.copy())
    step = block - overlap
    out = np.zeros_like(x)
    weights = np.zeros_like(x)
    for start in range(0, max(1, len(x) - overlap), step):
        end = min(len(x), start + block)
        seg = x[start:end]
        cleaned = worker(seg.copy())
        win = tukey(len(seg), alpha=min(1.0, 2.0 * overlap / max(len(seg), 1)))
        if start == 0:
            win[: min(overlap, len(win))] = 1.0
        if end == len(x):
            win[max(0, len(win) - overlap):] = 1.0
        out[start:end] += cleaned * win
        weights[start:end] += win
        if end == len(x):
            break
    return out / np.maximum(weights, 1e-12)


def clean(x: np.ndarray, fs: float, config: Optional[Config | dict] = None, *,
          block_s: Optional[float] = None, overlap_s: float = 4.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clean one target channel.  Long records are processed in overlapping blocks."""
    if config is None:
        cfg = Config(fs=float(fs))
    elif isinstance(config, Config):
        cfg = config.with_updates(fs=float(fs))
    else:
        cfg = Config(fs=float(fs)).with_updates(**config)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("clean() requires a finite one-dimensional target channel")
    blocks: List[dict] = []

    def worker(segment):
        cleaned, diag = process_epoch(segment, cfg)
        blocks.append({k: v for k, v in diag.items() if k != "config"})
        return cleaned

    if block_s is None:
        cleaned = worker(x.copy())
    else:
        cleaned = _overlap_add(x, fs, worker, block_s, overlap_s)
    if cleaned.shape != x.shape or not np.all(np.isfinite(cleaned)):
        raise FloatingPointError("AWRLS-2 produced a nonfinite or mis-shaped output")
    return cleaned, {"mode": "awrls2_strict_single_channel", "backend": bank.BACKEND, "config": asdict(cfg),
                     "block_s": block_s, "overlap_s": overlap_s if block_s else 0.0,
                     "detected_count": sum(b["detected_count"] for b in blocks),
                     "processed_count": sum(b["processed_count"] for b in blocks),
                     "blocks": blocks}
