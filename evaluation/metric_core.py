from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import friedmanchisquare, wilcoxon


CODE_VERSION = "paper-metrics-v1.0"
METHODS = ("FG-AWPR-RLS", "MCA", "FBSE-EWT-LPATV", "ITMS")
METHOD_FILE_NAMES = {method: method.replace("-", "_") for method in METHODS}
METHOD_COLORS = {
    "FG-AWPR-RLS": "#D62728",
    "MCA": "#6BAED6",
    "FBSE-EWT-LPATV": "#5AAE87",
    "ITMS": "#9746E8",
}
EPS = 1e-12

SIMULATED_METRICS = (
    "barr_percent",
    "recall_percent",
    "precision_percent",
    "f1_percent",
    "mae_after",
    "rmse_after",
    "snr_after_db",
    "delta_snr_db",
    "pearson_after",
)
REAL_METRICS = (
    "candidate_window_low_frequency_energy_reduction_percent",
    "outside_window_pearson_r",
    "outside_window_nrmse_percent",
)


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    peak: int
    score: float


@dataclass(frozen=True)
class RealReference:
    centered_original: np.ndarray
    low_frequency_original: np.ndarray
    events: tuple[Event, ...]
    candidate_mask: np.ndarray
    outside_mask: np.ndarray


def load_yaml(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPS:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > EPS:
        return float(mad / 0.6744897501960817)
    return float(np.std(values))


def moving_mean_nearest(values: np.ndarray, size: int) -> np.ndarray:
    """NumPy equivalent of uniform_filter1d(mode='nearest')."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    size = max(1, int(size))
    if size == 1 or len(values) == 0:
        return values.copy()
    left = size // 2
    right = size - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    cumulative = np.empty(len(padded) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(padded, dtype=np.float64, out=cumulative[1:])
    return (cumulative[size:] - cumulative[:-size]) / float(size)


def _filter_one_block(values: np.ndarray, sos: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float64)
    if len(values) < 16:
        return values.copy()
    default_pad = 3 * (2 * len(sos) + 1)
    padlen = min(default_pad, len(values) - 1)
    if padlen < 1:
        return values.copy()
    return np.asarray(
        sosfiltfilt(sos, values, padlen=padlen), dtype=np.float64
    )


def safe_lowpass(
    values: np.ndarray,
    fs: float,
    cutoff_hz: float = 12.0,
    order: int = 4,
    block_s: float = 120.0,
    overlap_s: float = 2.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or cutoff_hz <= 0 or cutoff_hz >= fs / 2.0:
        return values.copy()
    sos = butter(order, cutoff_hz / (fs / 2.0), btype="lowpass", output="sos")
    block = max(32, int(round(block_s * fs)))
    overlap = min(max(0, int(round(overlap_s * fs))), block // 3)
    if len(values) <= block or overlap == 0:
        return _filter_one_block(values, sos)

    step = block - overlap
    output = np.zeros(len(values), dtype=np.float64)
    weights = np.zeros(len(values), dtype=np.float64)
    start = 0
    while True:
        end = min(len(values), start + block)
        filtered = _filter_one_block(values[start:end], sos)
        window = np.ones(end - start, dtype=np.float64)
        fade = min(overlap, len(window))
        if fade > 1 and start > 0:
            phase = np.linspace(0.0, np.pi / 2.0, fade)
            window[:fade] = np.sin(phase) ** 2
        if fade > 1 and end < len(values):
            phase = np.linspace(np.pi / 2.0, 0.0, fade)
            window[-fade:] = np.sin(phase) ** 2
        output[start:end] += filtered * window
        weights[start:end] += window
        if end >= len(values):
            break
        start += step
    return output / np.maximum(weights, EPS)


def merge_events(events: Iterable[Event]) -> list[Event]:
    ordered = sorted(events, key=lambda event: (event.start, event.end))
    if not ordered:
        return []
    merged = [ordered[0]]
    for event in ordered[1:]:
        previous = merged[-1]
        if event.start <= previous.end:
            best = event if event.score > previous.score else previous
            merged[-1] = Event(
                previous.start,
                max(previous.end, event.end),
                best.peak,
                max(previous.score, event.score),
            )
        else:
            merged.append(event)
    return merged


def detect_blink_events(
    values: np.ndarray,
    fs: float,
    cfg: dict | None = None,
) -> list[Event]:
    cfg = cfg or {}
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low_frequency = safe_lowpass(
        values,
        fs,
        float(cfg.get("lowpass_hz", 12.0)),
        int(cfg.get("order", 4)),
    )
    centered = low_frequency - np.median(low_frequency)
    window_size = max(3, int(round(float(cfg.get("energy_window_s", 0.08)) * fs)))
    energy = moving_mean_nearest(centered * centered, window_size)
    n_effective = max(2, int(np.ceil(len(values) / window_size)))
    energy_sigma = robust_sigma(energy)
    threshold = np.median(energy) + energy_sigma * np.sqrt(2.0 * np.log(n_effective))
    distance = max(1, int(round(float(cfg.get("min_distance_s", 0.25)) * fs)))
    candidates, properties = find_peaks(
        energy,
        height=threshold,
        distance=distance,
        prominence=max(energy_sigma, EPS),
    )
    half_width = max(2, int(round(float(cfg.get("half_width_s", 0.45)) * fs)))
    minimum_duration = max(
        2, int(round(float(cfg.get("min_duration_s", 0.10)) * fs))
    )
    events: list[Event] = []
    heights = properties.get("peak_heights", np.zeros(len(candidates)))
    signal_sigma = robust_sigma(centered)
    for peak, height in zip(candidates, heights):
        peak = int(peak)
        local_start = max(0, peak - half_width)
        local_end = min(len(values), peak + half_width + 1)
        baseline = max(signal_sigma, 0.08 * abs(centered[peak]), EPS)
        left = peak
        while left > local_start and abs(centered[left]) > baseline:
            left -= 1
        right = peak
        while right < local_end - 1 and abs(centered[right]) > baseline:
            right += 1
        if right - left < minimum_duration:
            left = max(0, peak - minimum_duration // 2)
            right = min(len(values), left + minimum_duration)
        events.append(Event(left, right, peak, float(height)))
    return merge_events(events)


def event_mask(length: int, events: Iterable[Event], padding: int = 0) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for event in events:
        start = max(0, event.start - padding)
        end = min(length, event.end + padding)
        mask[start:end] = True
    return mask


def match_events(
    reference: list[Event],
    candidate: list[Event],
    fs: float,
    tolerance_s: float = 0.15,
) -> tuple[int, int, int]:
    tolerance = int(round(tolerance_s * fs))
    used: set[int] = set()
    true_positive = 0
    for truth in reference:
        possible: list[tuple[int, int]] = []
        for index, found in enumerate(candidate):
            if index in used:
                continue
            if (
                found.start <= truth.peak <= found.end
                or truth.start <= found.peak <= truth.end
                or abs(found.peak - truth.peak) <= tolerance
            ):
                possible.append((abs(found.peak - truth.peak), index))
        if possible:
            _, selected = min(possible)
            used.add(selected)
            true_positive += 1
    return true_positive, len(reference) - true_positive, len(candidate) - len(used)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    signal_energy = float(np.sum(reference * reference))
    noise_energy = float(np.sum((estimate - reference) ** 2))
    return float(10.0 * np.log10((signal_energy + EPS) / (noise_energy + EPS)))


def simulated_metrics(
    mixed: np.ndarray,
    clean: np.ndarray,
    cleaned: np.ndarray,
    fs: float,
    detector_cfg: dict | None = None,
) -> dict[str, float | int]:
    mixed = np.asarray(mixed, dtype=np.float64).reshape(-1)
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    cleaned = np.asarray(cleaned, dtype=np.float64).reshape(-1)
    if not (len(mixed) == len(clean) == len(cleaned)):
        raise ValueError("Simulated mixed, clean and cleaned signals must have equal lengths.")

    true_artifact = mixed - clean
    estimated_artifact = mixed - cleaned
    residual = cleaned - clean
    truth_events = detect_blink_events(true_artifact, fs, detector_cfg)
    removed_events = detect_blink_events(estimated_artifact, fs, detector_cfg)
    true_positive, false_negative, false_positive = match_events(
        truth_events, removed_events, fs
    )
    if truth_events:
        mask = event_mask(len(mixed), truth_events, int(round(0.05 * fs)))
        before_energy = float(np.sum(true_artifact[mask] ** 2))
        after_energy = float(np.sum(residual[mask] ** 2))
        barr = 100.0 * (1.0 - after_energy / (before_energy + EPS))
        recall = 100.0 * true_positive / len(truth_events)
    else:
        barr = float("nan")
        recall = float("nan")
    precision = (
        100.0 * true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else float("nan")
    )
    f1 = (
        200.0 * true_positive / (2 * true_positive + false_positive + false_negative)
        if 2 * true_positive + false_positive + false_negative > 0
        else float("nan")
    )
    before_snr = snr_db(clean, mixed)
    after_snr = snr_db(clean, cleaned)
    return {
        "true_blink_event_count": len(truth_events),
        "removed_event_tp": true_positive,
        "removed_event_fn": false_negative,
        "removed_event_fp": false_positive,
        "barr_percent": barr,
        "recall_percent": recall,
        "precision_percent": precision,
        "f1_percent": f1,
        "mae_before": float(np.mean(np.abs(mixed - clean))),
        "mae_after": float(np.mean(np.abs(cleaned - clean))),
        "rmse_before": float(np.sqrt(np.mean((mixed - clean) ** 2))),
        "rmse_after": float(np.sqrt(np.mean((cleaned - clean) ** 2))),
        "snr_before_db": before_snr,
        "snr_after_db": after_snr,
        "delta_snr_db": after_snr - before_snr,
        "pearson_before": correlation(clean, mixed),
        "pearson_after": correlation(clean, cleaned),
    }


def prepare_real_reference(
    original: np.ndarray,
    fs: float,
    detector_cfg: dict | None = None,
) -> RealReference:
    original = np.asarray(original, dtype=np.float64).reshape(-1)
    centered = original - np.median(original)
    events = tuple(detect_blink_events(centered, fs, detector_cfg))
    candidate = event_mask(len(original), events, int(round(0.05 * fs)))
    outside = ~event_mask(len(original), events, int(round(0.20 * fs)))
    if np.count_nonzero(outside) < 8:
        outside = np.ones(len(original), dtype=bool)
    low_frequency = safe_lowpass(centered, fs, 12.0, 4)
    return RealReference(centered, low_frequency, events, candidate, outside)


def real_metrics(
    original: np.ndarray,
    cleaned: np.ndarray,
    fs: float,
    reference: RealReference,
) -> dict[str, float | int | bool]:
    original = np.asarray(original, dtype=np.float64).reshape(-1)
    cleaned = np.asarray(cleaned, dtype=np.float64).reshape(-1)
    if len(original) != len(cleaned) or len(original) != len(reference.centered_original):
        raise ValueError("Real input and cleaned output must have equal lengths.")
    cleaned_centered = cleaned - np.median(cleaned)
    if reference.events:
        cleaned_low_frequency = safe_lowpass(cleaned_centered, fs, 12.0, 4)
        before_energy = float(
            np.sum(reference.low_frequency_original[reference.candidate_mask] ** 2)
        )
        after_energy = float(
            np.sum(cleaned_low_frequency[reference.candidate_mask] ** 2)
        )
        reduction = 100.0 * (1.0 - after_energy / (before_energy + EPS))
    else:
        reduction = float("nan")

    outside = reference.outside_mask
    original_outside = original[outside]
    cleaned_outside = cleaned[outside]
    denominator = float(np.sqrt(np.mean(original_outside**2)))
    nrmse = 100.0 * float(
        np.sqrt(np.mean((cleaned_outside - original_outside) ** 2))
    ) / (denominator + EPS)
    return {
        "candidate_event_count": len(reference.events),
        "candidate_energy_metric_valid": bool(reference.events),
        "candidate_window_low_frequency_energy_reduction_percent": reduction,
        "outside_window_sample_count": int(np.count_nonzero(outside)),
        "outside_window_pearson_r": correlation(original_outside, cleaned_outside),
        "outside_window_nrmse_percent": nrmse,
    }


def summary_table(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    group_columns: tuple[str, ...] = ("dataset", "method"),
) -> pd.DataFrame:
    rows: list[dict] = []
    if frame.empty:
        return pd.DataFrame()
    for group_key, group in frame.groupby(list(group_columns), dropna=False, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_columns, group_key))
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            row = {**base, "metric": metric, "n": int(len(values))}
            if len(values):
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                row.update(
                    {
                        "median": float(median),
                        "q25": float(q25),
                        "q75": float(q75),
                        "iqr": float(q75 - q25),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )
            else:
                row.update(
                    {
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "iqr": np.nan,
                        "mean": np.nan,
                        "std": np.nan,
                        "minimum": np.nan,
                        "maximum": np.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _holm_adjust(rows: list[dict]) -> list[dict]:
    valid = sorted(
        ((index, row["p_raw"]) for index, row in enumerate(rows) if np.isfinite(row["p_raw"])),
        key=lambda item: item[1],
    )
    running = 0.0
    total = len(valid)
    for rank, (index, probability) in enumerate(valid):
        adjusted = min(1.0, (total - rank) * probability)
        running = max(running, adjusted)
        rows[index]["p_holm"] = running
    return rows


def repeated_method_tests(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    unit_column: str = "unit_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    for metric in metrics:
        if metric not in frame:
            continue
        subset = frame[[unit_column, "method", metric]].copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        pivot = subset.pivot_table(
            index=unit_column, columns="method", values=metric, aggfunc="mean"
        )
        available = [method for method in METHODS if method in pivot.columns]
        complete = pivot[available].dropna() if available else pd.DataFrame()
        if len(available) == len(METHODS) and len(complete) >= 3:
            statistic, probability = friedmanchisquare(
                *[complete[method].to_numpy(float) for method in METHODS]
            )
            note = ""
        else:
            statistic, probability = np.nan, np.nan
            note = "insufficient complete four-method units"
        omnibus_rows.append(
            {
                "metric": metric,
                "n_complete": int(len(complete)),
                "test": "Friedman",
                "statistic": statistic,
                "p_value": probability,
                "note": note,
            }
        )
        if len(available) != len(METHODS) or len(complete) < 3:
            continue
        metric_pairs: list[dict] = []
        for left, right in combinations(METHODS, 2):
            difference = complete[left] - complete[right]
            try:
                statistic, probability = wilcoxon(
                    complete[left],
                    complete[right],
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            except ValueError:
                statistic, probability = np.nan, 1.0
            metric_pairs.append(
                {
                    "metric": metric,
                    "method_a": left,
                    "method_b": right,
                    "n": int(len(complete)),
                    "median_difference_a_minus_b": float(np.median(difference)),
                    "statistic": statistic,
                    "p_raw": probability,
                    "p_holm": np.nan,
                }
            )
        pairwise_rows.extend(_holm_adjust(metric_pairs))
    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)
