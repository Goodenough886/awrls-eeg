#!/usr/bin/env python3
"""Recompute manuscript D_BP from five paired simulated EEG NPY files.

Read-only inputs. No artifact-removal rerun, no network access, no raw-data ZIP.
NumPy and SciPy are imported only after an output/diagnostics directory exists.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import traceback
import warnings
import zipfile

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
METHODS = {
    "AWRLS": ("FG-AWPR-RLS", "simulated_FG_AWPR_RLS.npy"),
    "MCA": ("MCA", "simulated_MCA.npy"),
    "FBSE-EWT-LPATV": ("FBSE-EWT-LPATV", "simulated_FBSE_EWT_LPATV.npy"),
    "ITMS": ("ITMS", "simulated_ITMS.npy"),
}
BANDS = (
    ("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0),
    ("beta", 12.0, 30.0), ("gamma", 30.0, 40.0),
)
DEFAULTS = {
    "clean_path": "data/blinkeeg/clean.npy",
    "run_dir": "results",
    "output_root": "results",
    "sampling_rate_hz": 200.0,
    "window_seconds": 4.0,
    "overlap_fraction": 0.5,
    "nfft": None,
    "expected_epochs": 300,
    "method_files": {},
}
np = scipy = periodogram = get_window = trapezoid = friedmanchisquare = wilcoxon = None


def load_dependencies():
    global np, scipy, periodogram, get_window, trapezoid, friedmanchisquare, wilcoxon
    import numpy as np
    import scipy
    from scipy.signal import periodogram, get_window
    from scipy.integrate import trapezoid
    from scipy.stats import friedmanchisquare, wilcoxon


def say(message):
    print(message, flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_zip(folder, path):
    # Deliberate allowlist: never add any NPY/raw signal or recursively zip a run.
    allowed = {".csv", ".json", ".txt", ".py", ".yaml", ".md"}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix.lower() in allowed:
                archive.write(file, arcname=file.name)
    return path


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def select_missing_paths(config):
    """A file dialog is used only if the configured input path is missing."""
    if Path(config["clean_path"]).is_file() and Path(config["run_dir"]).is_dir():
        return
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if not Path(config["clean_path"]).is_file():
                selected = filedialog.askopenfilename(
                    title="选择模拟数据的 clean.npy（配对干净背景）",
                    filetypes=[("NumPy signal", "*.npy")], parent=root,
                )
                if selected:
                    config["clean_path"] = str(Path(selected).resolve())
            if not Path(config["run_dir"]).is_dir():
                selected = filedialog.askdirectory(
                    title="选择 run_20260903_011237 运行目录（其中有 cleaned 文件夹）",
                    parent=root,
                )
                if selected:
                    config["run_dir"] = str(Path(selected).resolve())
        finally:
            root.destroy()
    except Exception as exc:
        say(f"无法显示文件选择窗口，请在 dbp_config.json 中修改路径。{exc}")


def prepare_paths(config, gui):
    if gui:
        select_missing_paths(config)
    run_dir = Path(config["run_dir"])
    if run_dir.name.lower() == "cleaned":
        run_dir = run_dir.parent
        config["run_dir"] = str(run_dir)
    overrides = config.get("method_files") or {}
    unknown = set(overrides) - set(METHODS)
    if unknown:
        raise ValueError(f"method_files 中有未知方法名：{sorted(unknown)}")
    files = {"clean_reference": Path(config["clean_path"])}
    for method, (_, filename) in METHODS.items():
        files[method] = resolve_path(overrides[method], run_dir) if method in overrides else run_dir / "cleaned" / filename
    missing = [f"{label}: {path}" for label, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "以下文件没有找到。请检查 dbp_config.json 中的 clean_path 和 run_dir。\n"
            "run_dir 应指向运行目录；只需上述运行的模拟数据文件。\n" + "\n".join(missing)
        )
    if any(path.suffix.lower() != ".npy" for path in files.values()):
        raise ValueError("本工具只接受 .npy 数值数组。")
    if len({path.resolve() for path in files.values()}) != 5:
        raise ValueError("参考和四种方法必须分别指向五个不同文件，请检查路径。")
    return files


def epoch_view(array, label):
    # This is exactly engine.run_simulated's convention, with no transposition.
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(
            f"{label}: 形状 {array.shape} 不符合原代码。需要 (epochs, samples)，"
            "或一个片段的一维数组；不自动转置、展平或截取。"
        )
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{label}: 需要实数数值数组，实际类型为 {array.dtype}。")
    return array


def build_settings(config, n_samples):
    fs = float(config["sampling_rate_hz"])
    seconds = float(config["window_seconds"])
    overlap = float(config["overlap_fraction"])
    if not math.isfinite(fs) or fs <= 0 or fs / 2 < 40:
        raise ValueError("sampling_rate_hz 必须为正数，且 Nyquist 频率至少为 40 Hz。")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("window_seconds 必须为正数。")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap_fraction 必须在 [0, 1) 范围内。")
    window = int(math.floor(seconds * fs + 0.5))
    if window < 8 or window > n_samples:
        raise ValueError(f"窗长 {window} 点不适用于每个片段的 {n_samples} 点。请明确修改 window_seconds。")
    overlap_samples = int(math.floor(window * overlap + 0.5))
    step = window - overlap_samples
    if step <= 0:
        raise ValueError("取整后的重叠点数达到窗长，请减小 overlap_fraction。")
    nfft_value = config.get("nfft")
    nfft = window if nfft_value is None else int(nfft_value)
    if nfft_value is not None and float(nfft_value) != nfft:
        raise ValueError("nfft 必须为整数或 null。")
    if nfft < window:
        raise ValueError("nfft 不能小于分析窗长。")
    starts = list(range(0, n_samples - window + 1, step))
    return {
        "tool_version": VERSION,
        "calculation_status": "new_explicit_recomputation_not_verified_historical_settings",
        "metric": "D_BP", "unit": "dB",
        "formula": "L_b(z)=mean_k[10*log10(integral_Bb P_z,k(f) df)]; D_BP=sum_b abs(L_b(output)-L_b(clean))",
        "band_aggregation": "sum over five bands; NOT divide by five",
        "window_aggregation": "mean log power first, then subtract and take absolute values",
        "bands_hz": [{"band": b, "lower_hz": lo, "upper_hz": hi} for b, lo, hi in BANDS],
        "sampling_rate_hz": fs,
        "requested_window_seconds": seconds,
        "window_samples": window, "actual_window_seconds": window / fs,
        "requested_overlap_fraction": overlap,
        "overlap_samples": overlap_samples, "actual_overlap_fraction": overlap_samples / window,
        "step_samples": step, "step_seconds": step / fs,
        "window_starts_zero_based": starts,
        "n_complete_windows_per_epoch": len(starts),
        "n_samples_per_epoch": n_samples,
        "epoch_seconds": n_samples / fs,
        "unused_tail_samples": n_samples - (starts[-1] + window),
        "incomplete_window_policy": "discard final incomplete window; no signal padding",
        "spectral_estimator": "scipy.signal.periodogram separately in each outer analysis window",
        "window_function": "periodic Hann: scipy.signal.get_window('hann', W, fftbins=True)",
        "detrend": "constant independently in each window BEFORE tapering",
        "one_sided": True, "scaling": "density", "nfft": nfft,
        "frequency_bin_spacing_hz": fs / nfft,
        "internal_welch_averaging": False,
        "band_integration": "trapezoidal integral of piecewise-linear PSD over exact band edges; interpolate boundary values",
        "band_edge_policy": "adjacent integrals share their boundary point, not a finite-width bin",
        "power_units": "native signal amplitude squared; no absolute microvolt conversion assumed",
        "log_power_reference": "one native squared-amplitude unit; cancels in paired dB differences",
        "zero_power_policy": "nonpositive or nonfinite band power invalidates this signal-epoch; no epsilon or floor",
        "invalid_sample_policy": "any nonfinite sample invalidates the whole signal-epoch, including unused tail",
        "invalid_window_policy": "do not average a smaller subset of windows; any invalid window invalidates the signal-epoch",
        "primary_summary_policy": "same complete epochs across clean reference and all four methods",
        "secondary_summary_policy": "each method's valid pairs with clean reference; separate CSV",
        "summary_sd_ddof": 1, "epoch_index_base": 0,
        "statistical_unit": "paired epoch, not overlapping time window",
        "statistical_assumption": "independence between epochs not verified; no subject/session IDs supplied",
        "friedman": {"reference": "chi-square asymptotic", "df": 3, "minimum_complete_epochs": 10},
        "wilcoxon": {
            "alternative": "two-sided", "zero_method": "wilcox", "method": "approx",
            "correction": False, "difference_rounding_decimals_dB": 12,
            "minimum_nonzero_pairs_for_approximation": 10,
            "all_zero_case": "statistic=0, p=1 by explicit convention",
        },
        "multiplicity": "Holm correction across all six D_BP method pairs",
        "expected_epochs_from_manuscript": config.get("expected_epochs"),
    }


def band_integrals(frequencies, psd):
    powers = []
    for _, low, high in BANDS:
        inside = (frequencies > low) & (frequencies < high)
        grid = np.r_[low, frequencies[inside], high]
        values = np.r_[np.interp(low, frequencies, psd), psd[inside],
                       np.interp(high, frequencies, psd)]
        powers.append(float(trapezoid(values, grid)))
    return np.asarray(powers, dtype=np.float64)


def mean_log_powers(window_powers):
    if not np.isfinite(window_powers).all() or np.any(window_powers <= 0):
        raise ValueError("All window-band powers must be finite and positive.")
    return np.mean(10.0 * np.log10(window_powers), axis=0)


def analyze_epoch(signal, settings, taper):
    starts = settings["window_starts_zero_based"]
    width = settings["window_samples"]
    powers = np.full((len(starts), len(BANDS)), np.nan)
    if not np.isfinite(signal).all():
        return powers, None, "nonfinite_input_samples"
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index, start in enumerate(starts):
                f, psd = periodogram(
                    np.asarray(signal[start:start + width], dtype=np.float64),
                    fs=settings["sampling_rate_hz"], window=taper,
                    nfft=settings["nfft"], detrend="constant",
                    return_onesided=True, scaling="density",
                )
                powers[index] = band_integrals(f, psd)
            if not np.isfinite(powers).all():
                return powers, None, "nonfinite_band_power"
            if np.any(powers <= 0):
                return powers, None, "nonpositive_band_power"
            return powers, mean_log_powers(powers), ""
    except FloatingPointError:
        return powers, None, "numerical_overflow_or_invalid_operation"


def optional_context(run_dir, files, arrays, settings, out_dir, notes):
    """Audit small metadata only; no loading any other signal arrays."""
    result = {"run_dir": str(run_dir), "config_found": False, "raw_metric_csv_found": False}
    config_path = run_dir / "config_used.yaml"
    if config_path.is_file():
        if config_path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("config_used.yaml 异常大，请检查运行目录。")
        text = config_path.read_text(encoding="utf-8-sig")
        shutil.copyfile(config_path, out_dir / "original_run_config.yaml")
        block = re.search(r"(?ms)^simulated:\s*\n((?:[ \t]+[^\n]*\n|[ \t]*\n)*)", text + "\n")
        fields = {}
        if block:
            for name in ("sampling_rate_hz", "max_epochs", "clean_path"):
                match = re.search(rf"(?m)^  {name}:\s*([^\r\n]+)", block.group(1))
                if match:
                    fields[name] = match.group(1).strip().strip("'\"")
        result.update(config_found=True, simulated_fields=fields, config_sha256=hash_file(config_path))
        if "sampling_rate_hz" in fields:
            recorded_fs = float(fields["sampling_rate_hz"].split("#")[0].strip())
            if not math.isclose(recorded_fs, settings["sampling_rate_hz"], rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"原运行采样率 {recorded_fs} Hz 与本工具 {settings['sampling_rate_hz']} Hz 不一致，已停止。")
        else:
            notes.append("未从原配置识别模拟采样率；本次采用 dbp_config.json 中的明确设置。")
    else:
        notes.append("运行目录中未找到 config_used.yaml；采样率采用本工具配置，需与原运行核对。")
    metadata = run_dir / "raw" / "simulated_metrics_by_epoch.csv"
    if metadata.is_file():
        seen = {label: set() for label in METHODS}
        aliases = {legacy: label for label, (legacy, _) in METHODS.items()}
        aliases.update({label: label for label in METHODS})
        with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if {"method", "epoch"} <= set(reader.fieldnames or []):
                for row in reader:
                    label = aliases.get(row["method"])
                    if label is None:
                        continue
                    value = float(row["epoch"])
                    if not value.is_integer():
                        raise ValueError("原始指标明细中的 epoch 编号不是整数，已停止以防错配。")
                    epoch = int(value)
                    if epoch in seen[label]:
                        raise ValueError(f"原始指标明细中 {label} 的 epoch {epoch} 重复，已停止。")
                    seen[label].add(epoch)
                    if row.get("sampling_rate_hz") and not math.isclose(float(row["sampling_rate_hz"]), settings["sampling_rate_hz"], abs_tol=1e-9):
                        raise ValueError("原始指标明细中的采样率与本工具设置不一致，已停止。")
                    if row.get("n_samples") and float(row["n_samples"]) != settings["n_samples_per_epoch"]:
                        raise ValueError("原始指标明细中的每片段采样点数与 NPY 数组不一致，已停止。")
                expected = set(range(len(arrays["clean_reference"])))
                for label, ids in seen.items():
                    if ids and ids != expected:
                        raise ValueError(f"{label} 的原始指标 epoch 编号与 NPY 行数不一致；本工具不会自动截取。")
                    if not ids:
                        notes.append(f"原始指标明细中没有 {label} 的可核对 epoch 编号。")
                result["raw_metric_csv_found"] = True
                result["raw_metric_unique_epoch_counts"] = {key: len(value) for key, value in seen.items()}
                result["raw_metric_csv_sha256"] = hash_file(metadata)
            else:
                notes.append("原指标 CSV 缺少 method/epoch 列，未用其核对行号。")
    result["pairing_rule"] = "row i pairs with row i, following uploaded engine.run_simulated"
    result["pairing_limit"] = "shape and metadata checks cannot independently prove source identity if arrays were reordered or replaced"
    return result


def describe(values):
    n = len(values)
    return {
        "n_epochs": n,
        "mean_dbp_db": float(np.mean(values)) if n else None,
        "sd_dbp_db": float(np.std(values, ddof=1)) if n > 1 else None,
        "median_dbp_db": float(np.median(values)) if n else None,
        "q25_dbp_db": float(np.quantile(values, 0.25)) if n else None,
        "q75_dbp_db": float(np.quantile(values, 0.75)) if n else None,
        "min_dbp_db": float(np.min(values)) if n else None,
        "max_dbp_db": float(np.max(values)) if n else None,
    }


def statistical_tests(matrix, common):
    values = matrix[common]
    n = len(values)
    omnibus = {"metric": "D_BP", "test": "Friedman", "n_complete_epochs": n, "df": 3,
               "statistic": None, "p_asymptotic": None,
               "status": "not_run_n_less_than_10", "warning": "epoch independence not verified"}
    if n >= 10:
        if np.all(values == values[:, [0]]):
            omnibus.update(statistic=0.0, p_asymptotic=1.0, status="all_methods_equal_convention")
        else:
            result = friedmanchisquare(*values.T)
            omnibus.update(statistic=float(result.statistic), p_asymptotic=float(result.pvalue), status="computed")
    pairs = []
    for a, b in itertools.combinations(range(len(METHODS)), 2):
        difference = values[:, a] - values[:, b]
        rounded = np.round(difference, decimals=12)
        n_nonzero = int(np.count_nonzero(rounded))
        row = {"metric": "D_BP", "method_a": list(METHODS)[a], "method_b": list(METHODS)[b],
               "n_complete_epochs": n, "n_nonzero_rounded_differences": n_nonzero,
               "median_difference_a_minus_b_db": float(np.median(difference)) if n else None,
               "statistic": None, "p_raw": None, "p_holm": None,
               "status": "not_run_n_less_than_10", "calculation_warning": ""}
        if n >= 10:
            if n_nonzero == 0:
                row.update(statistic=0.0, p_raw=1.0, status="all_rounded_differences_zero_convention")
            elif n_nonzero < 10:
                row["status"] = "not_run_fewer_than_10_nonzero_differences"
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = wilcoxon(rounded, zero_method="wilcox", correction=False,
                                      alternative="two-sided", method="approx")
                row.update(statistic=float(result.statistic), p_raw=float(result.pvalue),
                           status="computed", calculation_warning="; ".join(str(w.message) for w in caught))
        pairs.append(row)
    # Keep the full predeclared six-comparison family even if a test is unavailable.
    order = sorted(range(6), key=lambda i: 1.0 if pairs[i]["p_raw"] is None else pairs[i]["p_raw"])
    running = 0.0
    for rank, index in enumerate(order):
        raw = pairs[index]["p_raw"]
        running = max(running, min(1.0, (6 - rank) * (1.0 if raw is None else raw)))
        if raw is not None:
            pairs[index]["p_holm"] = running
    return omnibus, pairs


def calculate(config, files, out_dir, manifest):
    arrays = {}
    input_info = {}
    for label, path in files.items():
        say(f"读取 {label}: {path.name}")
        original = np.load(path, mmap_mode="r", allow_pickle=False)
        arrays[label] = epoch_view(original, label)
        input_info[label] = {
            "path": str(path.resolve()), "original_shape": list(original.shape),
            "epoch_shape": list(arrays[label].shape), "dtype": str(original.dtype),
            "file_bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "sha256": hash_file(path),
        }
        manifest["inputs"] = input_info
    target_shape = arrays["clean_reference"].shape
    mismatches = {key: list(value.shape) for key, value in arrays.items() if value.shape != target_shape}
    if mismatches:
        raise ValueError(f"配对形状不一致：clean_reference={target_shape}，其他={mismatches}。不会转置或截取；请核对五个文件。")
    n_epochs, n_samples = target_shape
    settings = build_settings(config, n_samples)
    write_json(out_dir / "settings.json", settings)
    notes = []
    if config.get("expected_epochs") is not None and n_epochs != int(config["expected_epochs"]):
        notes.append(f"论文预期 {config['expected_epochs']} 个片段，实际找到 {n_epochs} 个；本次计算全部严格配对片段，请核对样本量。")
    context = optional_context(Path(config["run_dir"]), files, arrays, settings, out_dir, notes)
    write_json(out_dir / "source_context.json", context)
    labels = list(arrays)
    avg_logs = np.full((5, n_epochs, 5), np.nan)
    reasons = [[""] * n_epochs for _ in range(5)]
    window_fields = ["dataset", "signal", "epoch", "unit_id", "window_index", "start_sample",
                     "end_sample_exclusive", "start_seconds", "end_seconds", "band",
                     "lower_hz", "upper_hz", "power_native_squared", "log_power_db",
                     "window_band_valid", "signal_epoch_valid", "invalid_reason"]
    taper = get_window("hann", settings["window_samples"], fftbins=True)
    with (out_dir / "window_band_powers.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=window_fields)
        writer.writeheader()
        for epoch in range(n_epochs):
            for signal_index, label in enumerate(labels):
                powers, log_mean, reason = analyze_epoch(arrays[label][epoch], settings, taper)
                reasons[signal_index][epoch] = reason
                if log_mean is not None:
                    avg_logs[signal_index, epoch] = log_mean
                for wi, start in enumerate(settings["window_starts_zero_based"]):
                    for bi, (band, low, high) in enumerate(BANDS):
                        power = float(powers[wi, bi])
                        finite = math.isfinite(power)
                        valid = finite and power > 0
                        writer.writerow({
                            "dataset": "simulated_paired", "signal": label, "epoch": epoch,
                            "unit_id": f"epoch_{epoch:04d}", "window_index": wi, "start_sample": start,
                            "end_sample_exclusive": start + settings["window_samples"],
                            "start_seconds": start / settings["sampling_rate_hz"],
                            "end_seconds": (start + settings["window_samples"]) / settings["sampling_rate_hz"],
                            "band": band, "lower_hz": low, "upper_hz": high,
                            "power_native_squared": power if finite else None,
                            "log_power_db": 10 * math.log10(power) if valid else None,
                            "window_band_valid": int(valid), "signal_epoch_valid": int(not reason),
                            "invalid_reason": reason,
                        })
            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == n_epochs:
                say(f"已计算 {epoch + 1}/{n_epochs} 个配对片段")
    valid_signals = np.isfinite(avg_logs).all(axis=2)
    common = valid_signals.all(axis=0)
    matrix = np.full((n_epochs, 4), np.nan)
    epoch_rows, band_rows, qc_rows = [], [], []
    for epoch in range(n_epochs):
        for si, label in enumerate(labels):
            qc_rows.append({"signal": label, "epoch": epoch, "unit_id": f"epoch_{epoch:04d}",
                            "valid": int(valid_signals[si, epoch]), "reason": reasons[si][epoch]})
        for mi, method in enumerate(METHODS):
            valid = bool(valid_signals[0, epoch] and valid_signals[mi + 1, epoch])
            delta = avg_logs[mi + 1, epoch] - avg_logs[0, epoch]
            if valid:
                matrix[epoch, mi] = np.abs(delta).sum()
            row = {"dataset": "simulated_paired", "epoch": epoch, "unit_id": f"epoch_{epoch:04d}",
                   "method": method, "legacy_method": METHODS[method][0],
                   "n_windows_expected": settings["n_complete_windows_per_epoch"],
                   "valid_pair": int(valid), "complete_for_all_methods": int(common[epoch]),
                   "invalid_reason": "; ".join(
                       f"{labels[si]}:{reasons[si][epoch]}" for si in (0, mi + 1) if reasons[si][epoch]),
                   "D_BP_db": float(matrix[epoch, mi]) if valid else None}
            for bi, (band, low, high) in enumerate(BANDS):
                row[f"abs_delta_{band}_db"] = float(abs(delta[bi])) if valid else None
                band_rows.append({
                    "epoch": epoch, "method": method, "band": band, "lower_hz": low, "upper_hz": high,
                    "n_windows_expected": settings["n_complete_windows_per_epoch"],
                    "clean_mean_log_power_db": float(avg_logs[0, epoch, bi]) if valid_signals[0, epoch] else None,
                    "output_mean_log_power_db": float(avg_logs[mi + 1, epoch, bi]) if valid_signals[mi + 1, epoch] else None,
                    "signed_difference_db": float(delta[bi]) if valid else None,
                    "absolute_difference_db": float(abs(delta[bi])) if valid else None, "valid_pair": int(valid),
                })
            epoch_rows.append(row)
    write_csv(out_dir / "dbp_by_epoch.csv", epoch_rows, list(epoch_rows[0]))
    write_csv(out_dir / "epoch_band_log_powers.csv", band_rows, list(band_rows[0]))
    write_csv(out_dir / "signal_epoch_quality.csv", qc_rows, list(qc_rows[0]))
    primary, secondary = [], []
    for mi, method in enumerate(METHODS):
        primary.append({"method": method, "population": "common_complete_epochs", **describe(matrix[common, mi])})
        secondary.append({"method": method, "population": "method_specific_valid_pairs",
                          **describe(matrix[np.isfinite(matrix[:, mi]), mi])})
    write_csv(out_dir / "method_summary.csv", primary, list(primary[0]))
    write_csv(out_dir / "method_summary_all_valid.csv", secondary, list(secondary[0]))
    omnibus, pairs = statistical_tests(matrix, common)
    write_csv(out_dir / "friedman_test.csv", [omnibus], list(omnibus))
    write_csv(out_dir / "wilcoxon_holm_pairwise.csv", pairs, list(pairs[0]))
    counts = {"input_epochs": n_epochs, "common_complete_epochs": int(common.sum()),
              "complete_windows_per_signal_epoch": settings["n_complete_windows_per_epoch"],
              "invalid_signal_epochs": int((~valid_signals).sum()),
              "per_method_valid_pairs": {method: int(np.isfinite(matrix[:, mi]).sum()) for mi, method in enumerate(METHODS)},
              "excluded_common_epoch_ids": np.flatnonzero(~common).tolist()}
    manifest.update(status="completed" if common.all() else "completed_with_exclusions",
                    counts=counts, notes=notes, runtime={
                        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform(), "python_executable": sys.executable,
                    })
    lines = [
        "D_BP 本地重算结果", "",
        "本次计算使用明确的新设置；不声称复现原表未记录的窗长和谱估计。",
        f"输入：{n_epochs} 个配对片段，每片段 {n_samples} 点，采样率 {settings['sampling_rate_hz']:g} Hz。",
        f"设置：{settings['actual_window_seconds']:g} 秒窗，重叠 {settings['actual_overlap_fraction']:.0%}，"
        f"每片段 {settings['n_complete_windows_per_epoch']} 个完整窗；周期 Hann 窗、逐窗去均值、单边密度功率谱。",
        "先计算每窗五频带功率的 10log10，再在窗间平均，最后求五频带绝对差的总和。",
        f"主表和统计使用同一组 {int(common.sum())} 个四方法共同有效片段；窗口数不计入统计样本量。",
        "", "主表：均值 ± 样本标准差，单位 dB（较小表示平均频带功率更接近配对背景）：",
    ]
    for row in primary:
        if row["mean_dbp_db"] is not None:
            sd = f"{row['sd_dbp_db']:.6f}" if row["sd_dbp_db"] is not None else "不可计算（n<2）"
            lines.append(f"  {row['method']}: {row['mean_dbp_db']:.6f} ± {sd}; n={row['n_epochs']}")
        else:
            lines.append(f"  {row['method']}: 无有效共同配对片段")
    lines += [
        "", "文件对应：",
        "method_summary.csv：四方法共同有效样本的汇总，供论文表格复核。",
        "dbp_by_epoch.csv：逐 epoch D_BP、各频带贡献及有效性。",
        "window_band_powers.csv：逐信号、逐 epoch、逐时间窗、逐频带的功率和对数功率。",
        "epoch_band_log_powers.csv：各频带的窗间平均对数功率及输入—输出差。",
        "signal_epoch_quality.csv：全部信号片段的有效性与排除原因。",
        "settings.json / manifest.json / source_context.json：设置、样本量、输入文件哈希和来源核对。",
        "friedman_test.csv / wilcoxon_holm_pairwise.csv：以 epoch 为配对单位的检验，六组比较作 Holm 校正。",
        "method_summary_all_valid.csv：各方法自身有效配对的补充汇总，样本集可能不同。",
        "",
        "检验中的 epoch 间独立性尚未由受试者/录次编号核实，P 值供复核，不自动作为论文显著性结论。",
        "少于 10 个共同有效 epoch 时不作渐近检验；Wilcoxon 非零差值少于 10 时不作正态近似。",
        "输入任一非有限值，或任一窗频带功率非正/非有限，均使对应信号片段无效；不补零或加 epsilon。",
        "同一行按原引擎输出顺序配对；形状与元数据核对不能独立证明文件未被换序或替换。",
    ]
    if notes:
        lines += ["", "本次需核对："] + [f"  - {note}" for note in notes]
    (out_dir / "RESULT_README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    write_json(out_dir / "manifest.json", manifest)
    return counts


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="读取五个模拟 EEG NPY，重算 D_BP 并生成小型结果 ZIP。")
    parser.add_argument("--config", type=Path, default=HERE / "dbp_config.json")
    parser.add_argument("--clean", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, help="结果根目录，不覆盖原始文件")
    parser.add_argument("--fs", type=float)
    parser.add_argument("--window-s", type=float)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--nfft", type=int)
    parser.add_argument("--gui-on-missing", action="store_true")
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = HERE / "results" / f"DBP_results_{stamp}"
    manifest = {"tool_version": VERSION, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "started"}
    try:
        config_path = args.config.resolve()
        config = dict(DEFAULTS)
        if config_path.is_file():
            custom = json.loads(config_path.read_text(encoding="utf-8-sig"))
            unknown = set(custom) - set(DEFAULTS)
            if unknown:
                raise ValueError(f"配置中有未知键名：{sorted(unknown)}")
            config.update(custom)
        elif args.config != HERE / "dbp_config.json":
            raise FileNotFoundError(f"指定配置不存在：{args.config}")
        for field, value in (("clean_path", args.clean), ("run_dir", args.run_dir),
                             ("output_root", args.out_dir), ("sampling_rate_hz", args.fs),
                             ("window_seconds", args.window_s), ("overlap_fraction", args.overlap),
                             ("nfft", args.nfft)):
            if value is not None:
                config[field] = str(value) if isinstance(value, Path) else value
        for field in ("clean_path", "run_dir", "output_root"):
            # Paths supplied on CLI are interpreted relative to the current directory.
            cli_path = {"clean_path": args.clean, "run_dir": args.run_dir, "output_root": args.out_dir}[field]
            config[field] = str(resolve_path(config[field], Path.cwd() if cli_path is not None else config_path.parent))
        out_dir = Path(config["output_root"]) / f"DBP_results_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=False)
        say("D_BP 本地计算工具：只读配对背景和四种方法的模拟输出。")
        say("结果中包含指标和设置，不包含原始 NPY 信号。")
        write_json(out_dir / "resolved_config.json", config)
        load_dependencies()
        files = prepare_paths(config, args.gui_on_missing)
        write_json(out_dir / "resolved_config.json", config)
        shutil.copyfile(Path(__file__), out_dir / "compute_dbp.py")
        manifest["tool_sha256"] = hash_file(Path(__file__))
        counts = calculate(config, files, out_dir, manifest)
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out_dir / "manifest.json", manifest)
        final_zip = create_zip(out_dir, out_dir.with_suffix(".zip"))
        (HERE / "LAST_RESULT_PATH.txt").write_text(str(final_zip) + "\n", encoding="utf-8-sig")
        say(f"\n计算完成：共同有效片段 {counts['common_complete_epochs']}/{counts['input_epochs']}。")
        say(f"请上传这个 ZIP：\n{final_zip}\n文件大小：{final_zip.stat().st_size / 1024 / 1024:.2f} MB")
        return 0
    except Exception as exc:
        error = traceback.format_exc()
        say(f"\n计算停止：{exc}")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest.update(status="failed", error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(out_dir / "manifest.json", manifest)
            (out_dir / "ERROR.txt").write_text(
                "本次计算未完成，请上传诊断 ZIP 以检查问题，不要使用部分结果填论文。\n\n" + error,
                encoding="utf-8-sig",
            )
            diag = create_zip(out_dir, out_dir.parent / f"DBP_diagnostics_{stamp}.zip")
            say(f"请上传这个诊断 ZIP：\n{diag}")
        except Exception:
            say(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
