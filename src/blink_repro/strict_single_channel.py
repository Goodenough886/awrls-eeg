"""Target-only I/O, preprocessing and the existing single-input AWRLS core.

Multichannel MAT containers are decoded only at the I/O boundary. No numerical
data from unselected channels reach resampling, preprocessing or any method.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from .methods.base import MethodOutput
from .methods.fg_awpr_rls import _load
from .preprocessing import preprocess_real_eeg
from .signal_utils import overlap_add


MODE = "strict_single_channel_v1"


def single_vector(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("Strict single-channel processing requires one nonempty 1-D vector.")
    return values


def select_target(data: np.ndarray, names: list[str], target: str) -> np.ndarray:
    """Copy one column; other channel amplitudes never determine the selection."""
    matrix = np.asarray(data)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Expected samples x labelled channels at the I/O boundary.")
    matches = [i for i, name in enumerate(names) if name.casefold() == target.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one target channel {target!r}, found {len(matches)}.")
    return np.array(matrix[:, matches[0]], dtype=float, copy=True)


def read_target(path: Path, dataset: str, target: str):
    """Read raw target only; do NOT call the multichannel preprocessing adapters."""
    if dataset == "Kaya2018":
        recording = loadmat(Path(path), simplify_cells=True, variable_names=["o"])["o"]
        if not isinstance(recording, dict):
            raise ValueError("Kaya MAT variable o must be a struct.")
        names = [str(name) for name in np.asarray(recording["chnames"]).ravel()]
        data = np.asarray(recording["data"])
        if data.ndim == 1 and len(names) == 1:
            data = data[:, None]
        elif data.ndim != 2:
            raise ValueError(f"Unsupported Kaya signal shape: {data.shape}")
        if data.shape[1] == len(names):
            pass
        elif data.shape[0] == len(names):
            data = data.T
        elif data.shape[1] == len(names) + 1:
            # Same explicit trailing-unlabelled-column policy as the source reader.
            data = data[:, :len(names)]
        else:
            raise ValueError("Cannot align Kaya channel labels and signal columns.")
        selected = select_target(data, names, target)
        fs = float(recording["sampFreq"])
    else:
        raise ValueError(f"Unknown real dataset: {dataset}")
    return selected, float(fs), {
        "source_file": str(Path(path).resolve()), "source_channel": target,
        "original_fs": float(fs), "raw_target_samples": len(selected),
        "numeric_channels_after_selection": 1, "selection_before_preprocessing": True,
        "mat_container_may_contain_other_channels": True,
        "recorded_hardware_reference": "unchanged; not inferred from channel names",
    }


def preprocess_target(x, fs, target, dataset, config, target_fs=None):
    values = single_vector(x).copy()
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be positive and finite.")
    cfg = dict(config)
    allowed = {"trim_zero_edges", "minimum_zero_edge_s", "robust_center", "reference_mode",
               "highpass_hz", "lowpass_hz", "filter_order", "filter_segment_s",
               "final_robust_center", "zero_edge_absolute_tolerance", "zero_edge_relative_tolerance"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unsupported strict preprocessing settings: {sorted(unknown)}")
    if str(cfg.get("reference_mode", "none")).casefold() != "none":
        raise ValueError("Strict mode forbids common or auxiliary-channel re-referencing.")
    cfg.update(reference_mode="none", edge_detection_channels=[target])
    good = np.isfinite(values)
    if np.count_nonzero(good) < 2:
        raise ValueError("Target channel has fewer than two finite samples.")
    replaced = int(np.count_nonzero(~good))
    if replaced:
        grid = np.arange(len(values))
        values[~good] = np.interp(grid[~good], grid[good], values[good])
    original_fs = float(fs)
    if target_fs is not None:
        if not np.isfinite(target_fs) or target_fs <= 0:
            raise ValueError("Target sampling rate must be positive and finite.")
        if fs > target_fs:
            ratio = (Fraction(str(target_fs)) / Fraction(str(fs))).limit_denominator(10000)
            values = resample_poly(values, ratio.numerator, ratio.denominator)
            fs = original_fs * ratio.numerator / ratio.denominator
    result = preprocess_real_eeg(values[:, None], fs, [target], dataset, cfg)
    metadata = {**result.metadata, "mode": MODE, "source_channel": target,
                "edge_detection_channels": [target], "numeric_input_channels": 1,
                "original_fs": original_fs, "replaced_nonfinite_before_resampling": replaced,
                "resampled": bool(fs != original_fs),
                "time_origin_seconds": result.active_start / fs}
    return np.array(result.data[:, 0], copy=True), float(fs), metadata


def awrls_settings(fs, config=None):
    core = _load("fg_awpr_rls_simulated_reference.py", "fg_awpr_rls_simulated_reference")
    overrides = dict(config or {})
    allowed = {f.name for f in fields(core.Config)} - {"fs"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unsupported strict AWRLS settings: {sorted(unknown)}")
    cfg = core.Config(fs=float(fs), **overrides)
    if not np.isfinite(fs) or not 0 < cfg.lowpass_hz < fs / 2:
        raise ValueError("Invalid sampling rate or detection lowpass.")
    if not 0 < cfg.wavelet_target_hz < fs / 2 or not 0 <= cfg.minimum_similarity <= 1:
        raise ValueError("Invalid pseudo-reference frequency or similarity.")
    if not 0 < cfg.rls_forgetting_factor <= 1 or cfg.rls_delta <= 0:
        raise ValueError("Invalid RLS forgetting factor or initialization.")
    for name in ("rls_order", "rls_passes", "max_wavelet_level", "lowpass_order"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not cfg.wavelet_candidates or not 0 <= cfg.taper_alpha <= 1:
        raise ValueError("Invalid wavelet candidates or taper.")
    if not 0 < cfg.min_artifact_duration_s <= cfg.max_artifact_duration_s:
        raise ValueError("Invalid artifact duration limits.")
    return core, cfg


def strict_awrls_clean(x, fs, config=None, *, block_s=None, overlap_s=2.0):
    """Every detector, DWT candidate and RLS input derives from x alone.

    This intentionally uses the original single-channel simulation core, NOT
    the Kaya/Cho consensus, leave-one-out references or rescue heuristics.
    """
    values = single_vector(x)
    if not np.all(np.isfinite(values)):
        raise ValueError("AWRLS input must be finite; preprocess the target channel first.")
    core, cfg = awrls_settings(fs, config)
    if block_s is not None:
        if not np.isfinite(block_s) or block_s <= 2 * cfg.max_artifact_duration_s:
            raise ValueError("Block duration must exceed twice the maximum artifact duration.")
        if not np.isfinite(overlap_s) or not cfg.max_artifact_duration_s <= overlap_s < block_s:
            raise ValueError("Overlap must cover the maximum artifact duration and be smaller than the block.")
    blocks = []
    cursor = 0
    step = None if block_s is None else max(32, round(block_s * fs)) - round(overlap_s * fs)

    def worker(segment):
        nonlocal cursor
        offset = cursor
        if len(segment) <= 3 * (cfg.lowpass_order + 1):
            cleaned = segment.copy()
            info = {"detected_count": 0, "processed_count": 0, "skipped_count": 0,
                    "detected_intervals_raw": [], "intervals": [], "skipped_intervals": [],
                    "reason": "short_signal_passthrough"}
        else:
            cleaned, info = core.process_epoch(segment, cfg)
        events = []
        for event in info["intervals"]:
            row = event.serializable(len(blocks))
            for key in ("start", "end", "peak"):
                row[key] += offset
            row.update(reference_source="target_only", rls_desired_source="target_only",
                       subtraction_gain=1.0)
            events.append(row)
        blocks.append({"start": offset, "end": offset + len(segment),
                       "detected_count": info["detected_count"],
                       "processed_count": info["processed_count"],
                       "skipped_count": info["skipped_count"],
                       "threshold": info.get("threshold"),
                       "detected_intervals": [[s + offset, e + offset, p + offset]
                                              for s, e, p in info["detected_intervals_raw"]],
                       "corrections": events,
                       "skipped_intervals": [dict(row, **{k: row[k] + offset for k in ("start", "end", "peak")})
                                             for row in info["skipped_intervals"]],
                       "reason": info.get("reason")})
        cursor += step or len(segment)
        return cleaned

    cleaned = worker(values.copy()) if block_s is None else overlap_add(values, fs, worker, block_s, overlap_s)
    if cleaned.shape != values.shape or not np.all(np.isfinite(cleaned)):
        raise FloatingPointError("AWRLS returned a nonfinite or incorrectly shaped signal.")
    diagnostics = {"mode": MODE, "numeric_input_channels": 1, "auxiliary_channels": [],
                   "core": "fg_awpr_rls_simulated_reference.process_epoch", "config": asdict(cfg),
                   "block_s": block_s, "overlap_s": overlap_s if block_s is not None else 0,
                   "event_counts_include_overlap_duplicates": block_s is not None,
                   "detected_count": sum(b["detected_count"] for b in blocks),
                   "processed_count": sum(b["processed_count"] for b in blocks),
                   "skipped_count": sum(b["skipped_count"] for b in blocks), "blocks": blocks}
    return MethodOutput(cleaned, values - cleaned, diagnostics)
