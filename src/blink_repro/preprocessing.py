from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .signal_utils import safe_filter


@dataclass(frozen=True)
class PreprocessResult:
    """Uniform real-EEG preprocessing output shared by all four methods."""

    data: np.ndarray
    active_start: int
    active_end: int
    metadata: dict[str, Any]


def _finite_interpolate(data: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.asarray(data, dtype=float).copy()
    replaced = 0
    grid = np.arange(len(out), dtype=float)
    for column in range(out.shape[1]):
        signal = out[:, column]
        good = np.isfinite(signal)
        bad_count = int(np.count_nonzero(~good))
        if bad_count == 0:
            continue
        replaced += bad_count
        if np.count_nonzero(good) < 2:
            signal[:] = 0.0
        else:
            signal[~good] = np.interp(grid[~good], grid[good], signal[good])
    return out, replaced


def _channel_indices(channel_names: Sequence[str], requested: Sequence[str]) -> list[int]:
    lookup = {str(name).casefold(): index for index, name in enumerate(channel_names)}
    return [lookup[name.casefold()] for name in requested if name.casefold() in lookup]


def _active_bounds(
    data: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    cfg: dict[str, Any],
) -> tuple[int, int, int, int]:
    n_samples = len(data)
    if not bool(cfg.get("trim_zero_edges", True)) or n_samples == 0:
        return 0, n_samples, 0, 0

    requested = tuple(cfg.get("edge_detection_channels", ("Fp1", "Fp2", "Fpz", "Fz")))
    indices = _channel_indices(channel_names, requested)
    if not indices:
        indices = list(range(data.shape[1]))

    selected = np.asarray(data[:, indices], dtype=float)
    scale = float(np.nanpercentile(np.abs(selected), 95)) if selected.size else 0.0
    tolerance = max(
        float(cfg.get("zero_edge_absolute_tolerance", 1e-12)),
        float(cfg.get("zero_edge_relative_tolerance", 1e-10)) * max(scale, 1.0),
    )
    active = np.any(np.abs(selected) > tolerance, axis=1)
    active_indices = np.flatnonzero(active)
    if len(active_indices) == 0:
        return 0, n_samples, 0, 0

    proposed_start = int(active_indices[0])
    proposed_end = int(active_indices[-1] + 1)
    minimum = max(1, int(round(float(cfg.get("minimum_zero_edge_s", 0.25)) * fs)))
    start = proposed_start if proposed_start >= minimum else 0
    end = proposed_end if n_samples - proposed_end >= minimum else n_samples
    return start, end, start, n_samples - end


def _filter_segmented(
    data: np.ndarray,
    fs: float,
    highpass_hz: float,
    lowpass_hz: float,
    order: int,
    segment_seconds: float,
) -> np.ndarray:
    out = np.asarray(data, dtype=float).copy()
    if len(out) < 4:
        return out
    segment_samples = len(out)
    if segment_seconds > 0:
        segment_samples = max(4, int(round(segment_seconds * fs)))

    for start in range(0, len(out), segment_samples):
        end = min(len(out), start + segment_samples)
        block = out[start:end]
        for column in range(block.shape[1]):
            signal = block[:, column]
            if highpass_hz > 0:
                signal = safe_filter(signal, fs, highpass_hz, "highpass", order)
            if lowpass_hz > 0 and lowpass_hz < 0.495 * fs:
                signal = safe_filter(signal, fs, lowpass_hz, "lowpass", order)
            out[start:end, column] = signal
    return out


def preprocess_real_eeg(
    data: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    dataset: str,
    cfg: dict[str, Any] | None = None,
) -> PreprocessResult:
    """
    Apply one auditable preprocessing chain before any of the four methods.

    The function deliberately does not guess an ADC-to-microvolt factor. Values
    stay in their source scale and are labelled a.u. unless calibration metadata
    is explicitly available outside this routine.
    """
    settings = dict(cfg or {})
    source = np.asarray(data, dtype=float)
    if source.ndim != 2:
        raise ValueError(f"Real EEG preprocessing expects samples x channels, got {source.shape}.")
    if source.shape[1] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: data has {source.shape[1]} columns but "
            f"{len(channel_names)} channel names were supplied."
        )

    finite, replaced_nonfinite = _finite_interpolate(source)
    start, end, trimmed_prefix, trimmed_suffix = _active_bounds(
        finite, fs, channel_names, settings
    )
    active = finite[start:end].copy()
    if len(active) < max(16, int(round(fs * 0.25))):
        raise ValueError("After invalid-edge trimming, too few EEG samples remain.")

    before_medians = np.median(active, axis=0)
    if bool(settings.get("robust_center", True)):
        active -= before_medians[None, :]

    reference_mode = str(settings.get("reference_mode", "none")).casefold()
    reference_channel_count = 0
    if reference_mode not in {"none", "off", "false"}:
        requested_reference_channels = settings.get("reference_channels")
        if requested_reference_channels:
            reference_indices = _channel_indices(channel_names, requested_reference_channels)
        else:
            reference_indices = list(range(active.shape[1]))
        if len(reference_indices) < 2:
            raise ValueError("Common referencing requires at least two available channels.")
        reference_channel_count = len(reference_indices)
        if reference_mode in {"common_average", "car", "average"}:
            reference = np.mean(active[:, reference_indices], axis=1)
        elif reference_mode in {"robust_median", "median"}:
            reference = np.median(active[:, reference_indices], axis=1)
        else:
            raise ValueError(f"Unsupported reference_mode: {reference_mode}")
        active -= reference[:, None]

    highpass_hz = float(settings.get("highpass_hz", 0.5))
    lowpass_hz = float(settings.get("lowpass_hz", 45.0))
    filter_order = int(settings.get("filter_order", 4))
    filter_segment_s = float(settings.get("filter_segment_s", 0.0))
    active = _filter_segmented(
        active,
        fs,
        highpass_hz,
        lowpass_hz,
        filter_order,
        filter_segment_s,
    )

    residual_medians = np.median(active, axis=0)
    if bool(settings.get("final_robust_center", True)):
        active -= residual_medians[None, :]

    metadata: dict[str, Any] = {
        "dataset": str(dataset),
        "sampling_rate_hz": float(fs),
        "source_n_samples": int(len(source)),
        "processed_n_samples": int(len(active)),
        "active_start_sample": int(start),
        "active_end_sample": int(end),
        "trimmed_prefix_samples": int(trimmed_prefix),
        "trimmed_suffix_samples": int(trimmed_suffix),
        "replaced_nonfinite_values": int(replaced_nonfinite),
        "robust_center": bool(settings.get("robust_center", True)),
        "reference_mode": reference_mode,
        "reference_channel_count": int(reference_channel_count),
        "highpass_hz": float(highpass_hz),
        "lowpass_hz": float(lowpass_hz),
        "filter_order": int(filter_order),
        "filter_segment_s": float(filter_segment_s),
        "source_global_min": float(np.min(finite)),
        "source_global_max": float(np.max(finite)),
        "source_channel_median_min": float(np.min(before_medians)),
        "source_channel_median_max": float(np.max(before_medians)),
        "processed_global_min": float(np.min(active)),
        "processed_global_max": float(np.max(active)),
        "processed_channel_median_abs_max": float(np.max(np.abs(np.median(active, axis=0)))),
        "amplitude_unit": "a.u. (source scale; no guessed ADC conversion)",
    }
    return PreprocessResult(
        data=np.asarray(active, dtype=float),
        active_start=int(start),
        active_end=int(end),
        metadata=metadata,
    )
