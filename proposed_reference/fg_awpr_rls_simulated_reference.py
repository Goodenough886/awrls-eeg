from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.signal.windows import tukey

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "本版本需要 PyWavelets。请先运行：pip install PyWavelets"
    ) from exc


MAD_GAUSSIAN_NORMALIZER = 0.6744897501960817

DEFAULT_WAVELET_CANDIDATES: Tuple[str, ...] = (
    "haar",
    "db2", "db3", "db4", "db5", "db6", "db8",
    "sym4", "sym5", "sym6", "sym8",
    "coif1", "coif2", "coif3", "coif5",
    "bior2.2", "bior3.3", "bior3.5", "bior4.4", "bior6.8",
    "rbio4.4", "dmey",
)

# FG-AWPR-RLS 关键变化：
# 1) Pmin = 0.85；
# 2) 22 个离散母小波由 PyWavelets 标准 filter bank 提供；
# 3) 检测阈值由每个 epoch 的 robust MAD + universal threshold 自动计算；
# 4) DWT 在完整 epoch 上计算，局部 blink 区间只负责评分与 RLS；
# 5) 不再硬性要求 Jstart=4。根据 fs 与 target_hz 自动得到搜索起始层 Jstart；
# 6) 每个 wavelet × level 组合都参与竞争，不再先选“最深层”；
# 7) Pmin=0.85 仍是硬约束；
# 8) 最终 Score = similarity × energy_fidelity × frequency_fidelity；
# 9) frequency_fidelity 让参考频带贴近目标 blink 频带，同时避免 level-1 偏置。


@dataclass
class Config:
    fs: float = 200.0

    # 步骤 1：低频表示
    lowpass_hz: float = 12.0
    lowpass_order: int = 4

    # 步骤 2：信号驱动阈值。没有固定 6*MAD / 4*MAD 倍数。
    energy_window_s: float = 0.08
    universal_mad_normalizer: float = MAD_GAUSSIAN_NORMALIZER
    min_peak_distance_s: float = 0.45
    peak_refine_window_s: float = 0.12

    # 步骤 3：零交叉点区间合法性
    min_artifact_duration_s: float = 0.18
    max_artifact_duration_s: float = 1.50

    # 步骤 4：全 epoch / 长上下文 DWT + 频率感知的自适应层级评分
    minimum_similarity: float = 0.85
    max_wavelet_level: int = 6
    wavelet_target_hz: float = 12.0
    wavelet_candidates: Tuple[str, ...] = DEFAULT_WAVELET_CANDIDATES
    wavelet_mode: str = "periodization"

    # 步骤 5：RLS 自适应噪声抵消
    rls_order: int = 12
    rls_forgetting_factor: float = 0.995
    rls_delta: float = 0.01
    rls_passes: int = 2

    # 区间边界平滑，避免拼接突变
    taper_alpha: float = 0.12


@dataclass
class IntervalResult:
    start: int
    end: int
    peak: int
    wavelet: str
    level: int
    j_start: int
    reference_upper_hz: float
    similarity: float
    energy_retention: float
    energy_fidelity: float
    frequency_fidelity: float
    selection_score: float
    reference: np.ndarray
    estimated_artifact: np.ndarray

    def serializable(self, row: int) -> dict:
        return {
            "row": int(row),
            "start": int(self.start),
            "end": int(self.end),
            "peak": int(self.peak),
            "wavelet": self.wavelet,
            "level": int(self.level),
            "j_start": int(self.j_start),
            "reference_upper_hz": float(self.reference_upper_hz),
            "similarity": float(self.similarity),
            "energy_retention": float(self.energy_retention),
            "energy_fidelity": float(self.energy_fidelity),
            "frequency_fidelity": float(self.frequency_fidelity),
            "selection_score": float(self.selection_score),
        }


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    median = np.median(x)
    return float(np.median(np.abs(x - median)) + 1e-12)


def centered_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def lowpass_filter(x: np.ndarray, cfg: Config) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("lowpass_filter 只接收一维信号。")
    if not (0 < cfg.lowpass_hz < cfg.fs / 2):
        raise ValueError("lowpass_hz 必须位于 0 与奈奎斯特频率之间。")

    sos = butter(
        cfg.lowpass_order,
        cfg.lowpass_hz,
        btype="low",
        fs=cfg.fs,
        output="sos",
    )
    return sosfiltfilt(sos, x)


def zero_crossings(x: np.ndarray) -> np.ndarray:
    """返回每次符号变化右侧样本的索引。"""
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


def robust_sigma(x: np.ndarray, normalizer: float = MAD_GAUSSIAN_NORMALIZER) -> float:
    """MAD-based robust sigma estimate under an approximately Gaussian background."""
    normalizer = max(float(normalizer), 1e-12)
    return float(robust_mad(x) / normalizer)


def universal_lambda(n_eff: int) -> float:
    """Universal-threshold scale sqrt(2 ln N_eff)."""
    return float(np.sqrt(2.0 * np.log(max(int(n_eff), 2))))


def detect_blink_intervals(
    x: np.ndarray,
    cfg: Config,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    List[Tuple[int, int, int]],
    dict,
]:
    """
    检测流程：
    1) 12 Hz 低通得到低频表示；
    2) 对低频能量使用 robust sigma + universal threshold；
    3) 峰 prominence 由当前能量噪声尺度决定；
    4) 主峰幅度门限同样由当前 epoch 的 robust sigma 与 N 自动得到；
    5) 起点 = 主峰左侧最近零交叉点；终点 = 主峰右侧第三个零交叉点。

    不再使用固定 6*MAD、4*MAD 等经验倍数。
    """
    lp = lowpass_filter(x, cfg)
    centered = lp - np.median(lp)

    energy_window = max(3, int(round(cfg.energy_window_s * cfg.fs)))
    energy = uniform_filter1d(
        centered ** 2,
        size=energy_window,
        mode="nearest",
    )

    energy_median = float(np.median(energy))
    energy_sigma = robust_sigma(energy, cfg.universal_mad_normalizer)

    # 滑动能量样本并非完全独立，因此使用窗口数近似有效样本数。
    n_eff = max(2, int(np.ceil(len(energy) / energy_window)))
    energy_lambda = universal_lambda(n_eff)
    threshold = energy_median + energy_sigma * energy_lambda

    # prominence 由当前 epoch 的能量噪声尺度直接决定。
    prominence = max(float(energy_sigma), np.finfo(float).eps)
    min_distance = max(1, int(round(cfg.min_peak_distance_s * cfg.fs)))
    raw_peaks, _ = find_peaks(
        energy,
        height=threshold,
        prominence=prominence,
        distance=min_distance,
    )

    crossings = zero_crossings(centered)
    signal_sigma = robust_sigma(centered, cfg.universal_mad_normalizer)
    amplitude_lambda = universal_lambda(len(centered))
    amplitude_threshold = signal_sigma * amplitude_lambda
    refine_radius = max(1, int(round(cfg.peak_refine_window_s * cfg.fs)))

    candidates: List[Tuple[int, int, int]] = []

    for p in raw_peaks:
        left = max(0, int(p) - refine_radius)
        right = min(len(centered), int(p) + refine_radius + 1)
        if right <= left:
            continue

        refined_peak = left + int(np.argmax(np.abs(centered[left:right])))
        left_crossings = crossings[crossings < refined_peak]
        right_crossings = crossings[crossings > refined_peak]

        if len(left_crossings) < 1 or len(right_crossings) < 3:
            continue

        start = int(left_crossings[-1])
        end = int(right_crossings[2])
        duration_s = (end - start) / cfg.fs

        if not (
            cfg.min_artifact_duration_s
            <= duration_s
            <= cfg.max_artifact_duration_s
        ):
            continue

        if abs(centered[refined_peak]) < amplitude_threshold:
            continue

        candidates.append((start, end, refined_peak))

    # 重叠区间合并，避免同一眨眼被重复处理。
    candidates.sort(key=lambda item: item[0])
    merged: List[Tuple[int, int, int]] = []

    for start, end, peak in candidates:
        if not merged or start > merged[-1][1]:
            merged.append((start, end, peak))
            continue

        old_start, old_end, old_peak = merged[-1]
        stronger_peak = (
            peak if abs(centered[peak]) > abs(centered[old_peak]) else old_peak
        )
        merged[-1] = (
            min(old_start, start),
            max(old_end, end),
            stronger_peak,
        )

    threshold_details = {
        "energy_median": energy_median,
        "energy_sigma_hat": float(energy_sigma),
        "energy_window_samples": int(energy_window),
        "n_eff": int(n_eff),
        "universal_lambda": float(energy_lambda),
        "energy_threshold": float(threshold),
        "prominence": float(prominence),
        "signal_sigma_hat": float(signal_sigma),
        "amplitude_lambda": float(amplitude_lambda),
        "amplitude_threshold": float(amplitude_threshold),
    }

    return lp, energy, threshold, crossings, merged, threshold_details


def validate_wavelet_candidates(cfg: Config) -> None:
    available = set(pywt.wavelist(kind="discrete"))
    missing = [name for name in cfg.wavelet_candidates if name not in available]
    if missing:
        raise ValueError(
            "当前 PyWavelets 不支持以下候选离散母小波：" + ", ".join(missing)
        )


def starting_level_from_frequency(fs: float, target_hz: float) -> int:
    """
    给频率感知的 level 搜索确定起始层 Jstart。

    对 DWT approximation A_j，用名义上限频率：
        f_upper(A_j) ~= fs / 2^(j+1)

    FG-AWPR-RLS 从“仍能够覆盖 target_hz 的最深层”开始搜索：
        Jstart = floor(log2(fs / (2*target_hz)))

    对本文半模拟数据 fs=200 Hz、target=12 Hz：Jstart=3，
    A3 的名义上限约为 12.5 Hz，与 12 Hz 目标眼电频带非常接近。
    后续 level 3,4,...Jmax 都参与评分，由 frequency_fidelity 自动决定，
    因而不是把 level 3 硬编码为最终答案。
    """
    if fs <= 0:
        raise ValueError("fs 必须 > 0。")
    if not (0 < target_hz < fs / 2):
        raise ValueError("wavelet_target_hz 必须位于 0 与奈奎斯特频率之间。")
    return max(1, int(np.floor(np.log2(fs / (2.0 * target_hz)))))

def approximation_upper_hz(fs: float, level: int) -> float:
    return float(fs / (2.0 ** (level + 1)))


def reconstruct_approximation(
    x: np.ndarray,
    wavelet: "pywt.Wavelet",
    level: int,
    mode: str,
) -> np.ndarray:
    """标准 PyWavelets DWT，只保留 A_level 并重构到原长度。"""
    x = np.asarray(x, dtype=float)
    coeffs = pywt.wavedec(x, wavelet=wavelet, mode=mode, level=level)
    approx_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    reconstructed = pywt.waverec(
        approx_only,
        wavelet=wavelet,
        mode=mode,
    )
    if len(reconstructed) < len(x):
        reconstructed = np.pad(
            reconstructed,
            (0, len(x) - len(reconstructed)),
            mode="edge",
        )
    return np.asarray(reconstructed[: len(x)], dtype=float)


def precompute_epoch_wavelets(
    epoch: np.ndarray,
    cfg: Config,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], dict]:
    """
    对完整 epoch 预计算候选小波的合法近似分量。

    关键变化：DWT 不再在 0.3~0.6 s 的短 blink 片段上单独执行。
    长滤波器（尤其 dmey / coif5 / bior6.8）因此不会因为短片段而天然只剩 level 1。
    """
    epoch = np.asarray(epoch, dtype=float)
    j_start = starting_level_from_frequency(cfg.fs, cfg.wavelet_target_hz)
    cache: Dict[str, Dict[int, np.ndarray]] = {}
    availability: Dict[str, dict] = {}

    for name in cfg.wavelet_candidates:
        wavelet = pywt.Wavelet(name)
        library_max = int(pywt.dwt_max_level(len(epoch), wavelet.dec_len))
        j_max = min(int(cfg.max_wavelet_level), library_max)

        availability[name] = {
            "dec_len": int(wavelet.dec_len),
            "library_max_level": int(library_max),
            "effective_max_level": int(j_max),
            "j_start": int(j_start),
            "eligible": bool(j_max >= j_start),
        }

        if j_max < j_start:
            continue

        level_map: Dict[int, np.ndarray] = {}
        for level in range(j_start, j_max + 1):
            level_map[level] = reconstruct_approximation(
                epoch,
                wavelet=wavelet,
                level=level,
                mode=cfg.wavelet_mode,
            )
        cache[name] = level_map

    return cache, {
        "j_start": int(j_start),
        "target_hz": float(cfg.wavelet_target_hz),
        "availability": availability,
    }


def energy_fidelity_score(energy_retention: float) -> float:
    """
    能量保持率越接近 1 越好。
    使用 |log(r)| 对 r>1 与 r<1 做近似对称惩罚，避免把能量过冲当作奖励。
    """
    r = max(float(energy_retention), 1e-12)
    return float(np.exp(-abs(np.log(r))))


def frequency_fidelity_score(
    reference_upper_hz: float,
    target_hz: float,
) -> float:
    """
    参考 approximation 的名义上限频率越接近目标 blink 频带上限越好。

        F_f = exp(-|ln(f_upper / f_target)|)

    该形式对高于/低于目标频率的偏差在对数频率轴上对称：
    例如 fs=200 Hz、target=12 Hz 时，A3=12.5 Hz 比 A4=6.25 Hz 更接近目标，
    但 A4 若在形态/能量方面明显更好，仍可通过综合 Score 获胜。
    """
    f_ref = max(float(reference_upper_hz), 1e-12)
    f_target = max(float(target_hz), 1e-12)
    return float(np.exp(-abs(np.log(f_ref / f_target))))


def combined_wavelet_score(
    similarity: float,
    energy_fidelity: float,
    frequency_fidelity: float,
) -> float:
    """无人工权重的乘积评分。Pmin 在进入评分前已作为硬约束。"""
    return float(similarity * energy_fidelity * frequency_fidelity)


def choose_pseudo_reference_from_epoch(
    epoch: np.ndarray,
    start: int,
    end: int,
    peak: int,
    wavelet_cache: Dict[str, Dict[int, np.ndarray]],
    wavelet_meta: dict,
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[dict], List[dict]]:
    """
    FG-AWPR-RLS：对所有 wavelet × level 组合做频率感知的联合自适应选择。

    1) 完整 epoch 上预计算 DWT approximation；
    2) level 从自动得到的 Jstart 到各母小波的 Jmax；
    3) P >= Pmin=0.85 是硬约束；
    4) 对通过 Pmin 的每个 wavelet × level 计算：
          F_E = exp(-|ln(E_retention)|)
          F_f = exp(-|ln(f_upper / f_target)|)
          Score = P × F_E × F_f
    5) 不再先给每个母小波挑“最深层”，而是在全部合法组合中直接取 Score 最大者。

    这样 level 3/4/5/6 都有机会被选择，同时 level 1/2 不会重新利用
    “与原始信号过度相似”的优势统治候选池。
    """
    epoch = np.asarray(epoch, dtype=float)
    segment = epoch[start:end]
    if len(segment) < 4:
        return None, None, []

    peak_relative = int(peak - start)
    half_window = max(4, len(segment) // 4)
    window_start = max(0, peak_relative - half_window)
    window_end = min(len(segment), peak_relative + half_window + 1)

    original_energy = (
        np.sum(segment[window_start:window_end] ** 2) + 1e-12
    )

    candidates: List[dict] = []

    for name in cfg.wavelet_candidates:
        level_map = wavelet_cache.get(name)
        if not level_map:
            continue

        for level in sorted(level_map):
            full_approximation = level_map[level]
            local_approximation = full_approximation[start:end]
            similarity = abs(centered_correlation(segment, local_approximation))

            local_energy = np.sum(
                local_approximation[window_start:window_end] ** 2
            )
            retention = float(local_energy / original_energy)
            e_fidelity = energy_fidelity_score(retention)

            upper_hz = approximation_upper_hz(cfg.fs, level)
            f_fidelity = frequency_fidelity_score(
                upper_hz,
                cfg.wavelet_target_hz,
            )

            # Pmin 是唯一硬形态阈值。未通过就不参与综合评分。
            if similarity < cfg.minimum_similarity:
                continue

            score = combined_wavelet_score(
                similarity,
                e_fidelity,
                f_fidelity,
            )

            candidates.append({
                "wavelet": name,
                "level": int(level),
                "similarity": float(similarity),
                "energy_retention": retention,
                "energy_fidelity": e_fidelity,
                "reference_upper_hz": upper_hz,
                "frequency_fidelity": f_fidelity,
                "selection_score": score,
                "reference": local_approximation,
            })

    candidate_info = [
        {
            "wavelet": str(item["wavelet"]),
            "level": int(item["level"]),
            "reference_upper_hz": float(item["reference_upper_hz"]),
            "similarity": float(item["similarity"]),
            "energy_retention": float(item["energy_retention"]),
            "energy_fidelity": float(item["energy_fidelity"]),
            "frequency_fidelity": float(item["frequency_fidelity"]),
            "selection_score": float(item["selection_score"]),
        }
        for item in candidates
    ]

    if not candidates:
        return None, None, candidate_info

    best = max(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["frequency_fidelity"],
            item["energy_fidelity"],
            item["similarity"],
        ),
    )

    reference = np.asarray(best["reference"], dtype=float).copy()

    # 局部 reference 去端点线性基线，降低 taper 后的边界跳变。
    endpoint_baseline = np.linspace(reference[0], reference[-1], len(reference))
    reference -= endpoint_baseline

    best_info = {
        "wavelet": str(best["wavelet"]),
        "level": int(best["level"]),
        "j_start": int(wavelet_meta["j_start"]),
        "reference_upper_hz": float(best["reference_upper_hz"]),
        "similarity": float(best["similarity"]),
        "energy_retention": float(best["energy_retention"]),
        "energy_fidelity": float(best["energy_fidelity"]),
        "frequency_fidelity": float(best["frequency_fidelity"]),
        "selection_score": float(best["selection_score"]),
    }

    return reference, best_info, candidate_info

def rls_artifact_estimate(
    desired: np.ndarray,
    reference: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    RLS 学习“伪参考 -> 实际污染片段中的眨眼伪影”映射。
    返回估计伪影 y(n)，最终清洁片段为 desired - y。
    """
    desired = np.asarray(desired, dtype=float)
    reference = np.asarray(reference, dtype=float)

    reference_scale = np.std(reference) + 1e-12
    normalized_reference = reference / reference_scale

    order = min(cfg.rls_order, max(1, len(desired) // 5))
    weights = np.zeros(order, dtype=float)
    inverse_covariance = np.eye(order, dtype=float) / cfg.rls_delta

    # 多遍学习是离线实现。实时场景可把 rls_passes 改为 1。
    for _ in range(cfg.rls_passes):
        history = np.zeros(order, dtype=float)

        for n in range(len(desired)):
            history[1:] = history[:-1]
            history[0] = normalized_reference[n]

            p_times_x = inverse_covariance @ history
            denominator = (
                cfg.rls_forgetting_factor + history @ p_times_x
            )
            if abs(denominator) < 1e-12:
                continue

            gain = p_times_x / denominator
            prediction_error = desired[n] - weights @ history
            weights += gain * prediction_error

            inverse_covariance = (
                inverse_covariance - np.outer(gain, p_times_x)
            ) / cfg.rls_forgetting_factor
            inverse_covariance = (
                inverse_covariance + inverse_covariance.T
            ) * 0.5

    estimated_artifact = np.zeros(len(desired), dtype=float)
    history = np.zeros(order, dtype=float)

    for n in range(len(desired)):
        history[1:] = history[:-1]
        history[0] = normalized_reference[n]
        estimated_artifact[n] = weights @ history

    return estimated_artifact


def process_epoch(
    x: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=float)
    output = x.copy()

    (
        lp,
        energy,
        threshold,
        crossings,
        detected_intervals,
        threshold_details,
    ) = detect_blink_intervals(x, cfg)

    interval_results: List[IntervalResult] = []
    candidate_tables: List[List[dict]] = []
    skipped_intervals: List[dict] = []

    wavelet_cache: Dict[str, Dict[int, np.ndarray]] = {}
    wavelet_meta = {
        "j_start": starting_level_from_frequency(cfg.fs, cfg.wavelet_target_hz),
        "target_hz": float(cfg.wavelet_target_hz),
        "availability": {},
    }

    if detected_intervals:
        wavelet_cache, wavelet_meta = precompute_epoch_wavelets(x, cfg)

    for start, end, peak in detected_intervals:
        segment = x[start:end].copy()
        if len(segment) < 20:
            skipped_intervals.append({
                "start": int(start),
                "end": int(end),
                "peak": int(peak),
                "reason": "segment_too_short",
            })
            continue

        reference, best_info, candidate_info = choose_pseudo_reference_from_epoch(
            x,
            start,
            end,
            peak,
            wavelet_cache,
            wavelet_meta,
            cfg,
        )

        if reference is None or best_info is None:
            skipped_intervals.append({
                "start": int(start),
                "end": int(end),
                "peak": int(peak),
                "reason": "no_wavelet_level_satisfies_Pmin_and_adaptive_search",
                "j_start": int(wavelet_meta["j_start"]),
                "minimum_similarity": float(cfg.minimum_similarity),
            })
            candidate_tables.append(candidate_info)
            continue

        estimated_artifact = rls_artifact_estimate(
            segment,
            reference,
            cfg,
        )

        taper = tukey(len(segment), alpha=cfg.taper_alpha)
        tapered_estimate = taper * estimated_artifact
        output[start:end] = segment - tapered_estimate

        interval_results.append(IntervalResult(
            start=start,
            end=end,
            peak=peak,
            wavelet=best_info["wavelet"],
            level=best_info["level"],
            j_start=best_info["j_start"],
            reference_upper_hz=best_info["reference_upper_hz"],
            similarity=best_info["similarity"],
            energy_retention=best_info["energy_retention"],
            energy_fidelity=best_info["energy_fidelity"],
            frequency_fidelity=best_info["frequency_fidelity"],
            selection_score=best_info["selection_score"],
            reference=reference,
            estimated_artifact=tapered_estimate,
        ))
        candidate_tables.append(candidate_info)

    diagnostics = {
        "lowpass": lp,
        "energy": energy,
        "threshold": float(threshold),
        "threshold_details": threshold_details,
        "crossings": crossings,
        "detected_count": int(len(detected_intervals)),
        # 保留检测阶段的原始事件区间，用于 paired-clean 半模拟数据上的事件级 TPR 评价。
        # 这里是检测结果，而不是通过 Pmin 后真正完成 RLS 校正的 interval_results。
        "detected_intervals_raw": [
            (int(start), int(end), int(peak))
            for start, end, peak in detected_intervals
        ],
        "processed_count": int(len(interval_results)),
        "skipped_count": int(len(skipped_intervals)),
        "skipped_intervals": skipped_intervals,
        "wavelet_meta": wavelet_meta,
        "intervals": interval_results,
        "candidate_tables": candidate_tables,
    }
    return output, diagnostics


def process_array(
    data: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, List[dict]]:
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        cleaned, diagnostics = process_epoch(data, cfg)
        return cleaned, [diagnostics]

    if data.ndim != 2:
        raise ValueError(
            "输入必须是一维信号，或二维 (n_epochs, n_samples) 数组。"
        )

    cleaned = np.empty_like(data, dtype=float)
    all_diagnostics: List[dict] = []

    for row in range(data.shape[0]):
        cleaned[row], diagnostics = process_epoch(data[row], cfg)
        all_diagnostics.append(diagnostics)

    return cleaned, all_diagnostics


def to_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum((estimate - reference) ** 2) + 1e-12
    return float(10.0 * np.log10(signal_power / noise_power))


def global_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    return centered_correlation(reference.ravel(), estimate.ravel())


def _contiguous_true_artifact_intervals(
    mixed_row: np.ndarray,
    clean_row: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """
    从 paired-clean 半模拟数据恢复植入眨眼的事件级真值。

    当前 mixedEEG 与 cleanEEG 是逐样本对应的：
        true_artifact = mixedEEG - cleanEEG
    在未植入伪影的样本处二者相等，因此差值为 0；连续非零区间即为植入伪影区间。

    返回 (start, end, peak)，其中 end 为 Python 切片式右开端点，peak 为该真实伪影
    区间内 |mixed-clean| 最大的位置。
    """
    mixed_row = np.asarray(mixed_row, dtype=float)
    clean_row = np.asarray(clean_row, dtype=float)
    artifact = mixed_row - clean_row

    scale = max(float(np.max(np.abs(artifact))), 1.0)
    tolerance = max(1e-12, 100.0 * np.finfo(float).eps * scale)
    mask = np.abs(artifact) > tolerance

    if not np.any(mask):
        return []

    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    intervals: List[Tuple[int, int, int]] = []
    for start, end in zip(edges[0::2], edges[1::2]):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        peak = start + int(np.argmax(np.abs(artifact[start:end])))
        intervals.append((start, end, peak))
    return intervals


def _match_detected_to_truth_by_peak(
    truth_intervals: Sequence[Tuple[int, int, int]],
    detected_intervals: Sequence[Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    """
    事件级一对一匹配。

    一个真实眨眼记为 TP，当且仅当存在一个尚未匹配的检测区间包含该真实眨眼的
    伪影峰值。若多个检测区间均包含该峰值，选择检测峰与真实峰距离最近者。

    这种定义不额外引入人为 IoU 阈值，且避免仅有极小边缘重叠就被判为命中。
    返回 TP, FN, FP。
    """
    used_detected = set()
    tp = 0

    for _ts, _te, true_peak in truth_intervals:
        candidates = []
        for j, (ds, de, detected_peak) in enumerate(detected_intervals):
            if j in used_detected:
                continue
            if int(ds) <= int(true_peak) < int(de):
                candidates.append((abs(int(detected_peak) - int(true_peak)), j))

        if candidates:
            _, best_j = min(candidates)
            used_detected.add(best_j)
            tp += 1

    fn = len(truth_intervals) - tp
    fp = len(detected_intervals) - len(used_detected)
    return int(tp), int(fn), int(fp)


def evaluate(
    mixed: np.ndarray,
    cleaned: np.ndarray,
    clean: np.ndarray,
    diagnostics: Sequence[dict],
) -> dict:
    """
    半模拟 paired-clean 评价指标：
      1) 眨眼伪影检测率 / TPR：事件级真值来自 mixed-clean 的真实植入区间；
      2) MAE；
      3) SNR；
      4) Pearson correlation。

    RMSE 不再作为本文评价指标。
    """
    if mixed.shape != clean.shape or cleaned.shape != clean.shape:
        raise ValueError("mixed、cleaned 和 clean 的形状必须完全一致。")

    mixed_2d = to_2d(mixed)
    cleaned_2d = to_2d(cleaned)
    clean_2d = to_2d(clean)
    n_epochs = mixed_2d.shape[0]

    error_before = mixed_2d - clean_2d
    error_after = cleaned_2d - clean_2d

    row_mae_before = np.mean(np.abs(error_before), axis=1)
    row_mae_after = np.mean(np.abs(error_after), axis=1)

    signal_power = np.sum(clean_2d ** 2, axis=1)
    noise_power_before = np.sum(error_before ** 2, axis=1) + 1e-12
    noise_power_after = np.sum(error_after ** 2, axis=1) + 1e-12
    row_snr_before = 10.0 * np.log10(signal_power / noise_power_before)
    row_snr_after = 10.0 * np.log10(signal_power / noise_power_after)

    row_correlation_before = np.asarray([
        centered_correlation(clean_2d[i], mixed_2d[i])
        for i in range(n_epochs)
    ], dtype=float)
    row_correlation_after = np.asarray([
        centered_correlation(clean_2d[i], cleaned_2d[i])
        for i in range(n_epochs)
    ], dtype=float)

    row_mae_reduction_percent = 100.0 * (
        1.0 - row_mae_after / (row_mae_before + 1e-12)
    )
    row_snr_improvement = row_snr_after - row_snr_before
    row_correlation_improvement = (
        row_correlation_after - row_correlation_before
    )

    # ---------- 事件级眨眼检测率 TPR ----------
    row_truth_events = np.zeros(n_epochs, dtype=int)
    row_tp = np.zeros(n_epochs, dtype=int)
    row_fn = np.zeros(n_epochs, dtype=int)
    row_fp = np.zeros(n_epochs, dtype=int)
    row_tpr_percent = np.full(n_epochs, np.nan, dtype=float)

    for i in range(n_epochs):
        truth = _contiguous_true_artifact_intervals(mixed_2d[i], clean_2d[i])
        detected = diagnostics[i].get("detected_intervals_raw", [])
        tp, fn, fp = _match_detected_to_truth_by_peak(truth, detected)

        row_truth_events[i] = len(truth)
        row_tp[i] = tp
        row_fn[i] = fn
        row_fp[i] = fp
        if len(truth) > 0:
            row_tpr_percent[i] = 100.0 * tp / len(truth)

    total_truth_events = int(np.sum(row_truth_events))
    total_tp = int(np.sum(row_tp))
    total_fn = int(np.sum(row_fn))
    total_fp = int(np.sum(row_fp))
    global_tpr_percent = (
        100.0 * total_tp / total_truth_events
        if total_truth_events > 0 else float("nan")
    )

    intervals_per_epoch = np.asarray([
        item.get("detected_count", len(item["intervals"])) for item in diagnostics
    ], dtype=float)
    processed_per_epoch = np.asarray([
        item.get("processed_count", len(item["intervals"])) for item in diagnostics
    ], dtype=float)
    skipped_per_epoch = np.asarray([
        item.get("skipped_count", 0) for item in diagnostics
    ], dtype=float)
    detected_intervals = int(np.sum(intervals_per_epoch))
    processed_intervals = int(np.sum(processed_per_epoch))
    skipped_intervals = int(np.sum(skipped_per_epoch))

    def mean_std(values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {"mean": float("nan"), "std": float("nan")}
        ddof = 1 if len(values) > 1 else 0
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=ddof)),
        }

    summary = {
        "BlinkDetectionRate": {
            "global_tpr_percent": float(global_tpr_percent),
            "per_epoch": mean_std(row_tpr_percent),
            "true_events": total_truth_events,
            "true_positive": total_tp,
            "false_negative": total_fn,
            "false_positive": total_fp,
            "unit": "%",
        },
        "MAE": {
            "before": mean_std(row_mae_before),
            "after": mean_std(row_mae_after),
            "change": mean_std(row_mae_reduction_percent),
            "change_description": "降低百分比",
            "unit": "",
        },
        "SNR": {
            "before": mean_std(row_snr_before),
            "after": mean_std(row_snr_after),
            "change": mean_std(row_snr_improvement),
            "change_description": "提升值",
            "unit": "dB",
        },
        "Correlation": {
            "before": mean_std(row_correlation_before),
            "after": mean_std(row_correlation_after),
            "change": mean_std(row_correlation_improvement),
            "change_description": "提高值",
            "unit": "",
        },
    }

    return {
        "shape": list(mixed.shape),
        "n_epochs": int(n_epochs),
        "detected_intervals": detected_intervals,
        "processed_intervals": processed_intervals,
        "skipped_intervals": skipped_intervals,
        "intervals_per_epoch": mean_std(intervals_per_epoch),
        "processed_per_epoch": mean_std(processed_per_epoch),
        "skipped_per_epoch": mean_std(skipped_per_epoch),
        "mae_improved_fraction": float(
            np.mean(row_mae_after < row_mae_before)
        ),
        "display_epoch": int(np.argmax(row_mae_before)),
        "summary": summary,
        "per_epoch": {
            "truth_blink_events": row_truth_events.tolist(),
            "true_positive": row_tp.tolist(),
            "false_negative": row_fn.tolist(),
            "false_positive": row_fp.tolist(),
            "blink_detection_tpr_percent": row_tpr_percent.tolist(),
            "mae_before": row_mae_before.tolist(),
            "mae_after": row_mae_after.tolist(),
            "snr_before_db": row_snr_before.tolist(),
            "snr_after_db": row_snr_after.tolist(),
            "correlation_before": row_correlation_before.tolist(),
            "correlation_after": row_correlation_after.tolist(),
            "mae_reduction_percent": row_mae_reduction_percent.tolist(),
            "snr_improvement_db": row_snr_improvement.tolist(),
            "correlation_improvement": row_correlation_improvement.tolist(),
        },
    }

def find_input_file(value: str) -> Path:
    """支持绝对路径，也自动搜索当前目录和常见 Windows 桌面目录。"""
    given = Path(value).expanduser()
    candidates = [given]

    if not given.is_absolute():
        home = Path.home()
        candidates.extend([
            Path.cwd() / given,
            home / "Desktop" / given,
            home / "桌面" / given,
            home / "OneDrive" / "Desktop" / given,
            home / "OneDrive" / "桌面" / given,
        ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"找不到文件：{value}\n已搜索：\n{searched}"
    )


def choose_plot_row(
    mixed: np.ndarray,
    clean: Optional[np.ndarray],
    requested_row: Optional[int],
) -> int:
    mixed_2d = to_2d(mixed)

    if requested_row is not None:
        if not (0 <= requested_row < mixed_2d.shape[0]):
            raise IndexError(
                f"plot_row={requested_row} 超出范围 0~{mixed_2d.shape[0]-1}。"
            )
        return requested_row

    if clean is not None:
        clean_2d = to_2d(clean)
        row_mae = np.mean(np.abs(mixed_2d - clean_2d), axis=1)
        return int(np.argmax(row_mae))

    return int(np.argmax(np.std(mixed_2d, axis=1)))


def save_stepwise_plot(
    mixed: np.ndarray,
    cleaned: np.ndarray,
    clean: Optional[np.ndarray],
    diagnostics: Sequence[dict],
    cfg: Config,
    row: int,
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mixed_2d = to_2d(mixed)
    cleaned_2d = to_2d(cleaned)
    clean_2d = None if clean is None else to_2d(clean)

    x = mixed_2d[row]
    y = cleaned_2d[row]
    c = None if clean_2d is None else clean_2d[row]
    diag = diagnostics[row]
    t = np.arange(len(x)) / cfg.fs

    intervals: List[IntervalResult] = diag["intervals"]

    # 选择峰值最强的检测区间，用于展示 DWT 伪参考和 RLS 估计。
    strongest: Optional[IntervalResult] = None
    if intervals:
        strongest = max(intervals, key=lambda item: abs(x[item.peak]))

    figure, axes = plt.subplots(6, 1, figsize=(15, 20), constrained_layout=True)

    # 1. 原始 clean / mixed 对比
    axes[0].plot(t, x, linewidth=0.8, label="Contaminated EEG")
    if c is not None:
        axes[0].plot(t, c, linewidth=0.8, label="Clean reference")
    axes[0].set_title(f"Step 0 | Input comparison, epoch {row}")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    # 2. 12 Hz 低频表示、零交叉点、检测区间
    lp = np.asarray(diag["lowpass"])
    axes[1].plot(t, lp, linewidth=1.0, label="12 Hz low-pass")
    crossings = np.asarray(diag["crossings"], dtype=int)
    if len(crossings):
        axes[1].scatter(
            crossings / cfg.fs,
            lp[crossings],
            s=9,
            label="Zero crossings",
        )
    for idx, item in enumerate(intervals):
        axes[1].axvspan(
            item.start / cfg.fs,
            item.end / cfg.fs,
            alpha=0.16,
            label="Detected interval" if idx == 0 else None,
        )
        axes[1].axvline(item.peak / cfg.fs, linestyle="--", linewidth=0.8)
    axes[1].set_title(
        "Step 1-3 | 12 Hz representation + zero-crossing intervals"
    )
    axes[1].set_ylabel("Low-frequency amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    # 3. 能量包络和自适应阈值
    axes[2].plot(t, diag["energy"], linewidth=1.0, label="Smoothed low-frequency energy")
    axes[2].axhline(
        diag["threshold"],
        linestyle="--",
        linewidth=1.0,
        label=f"Adaptive threshold = {diag['threshold']:.2f}",
    )
    for item in intervals:
        axes[2].axvline(item.peak / cfg.fs, linestyle=":", linewidth=0.8)
    axes[2].set_title(
        "Step 2 | Signal-adaptive universal threshold "
        f"(N_eff={diag.get('threshold_details', {}).get('n_eff', '?')})"
    )
    axes[2].set_ylabel("Energy")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.2)

    # 4. DWT 伪参考
    if strongest is not None:
        local_t = np.arange(strongest.end - strongest.start) / cfg.fs
        segment = x[strongest.start:strongest.end]
        axes[3].plot(local_t, segment, linewidth=0.9, label="Local contaminated segment")
        axes[3].plot(local_t, strongest.reference, linewidth=1.2, label="Selected pseudo-reference")
        axes[3].axvline(
            (strongest.peak - strongest.start) / cfg.fs,
            linestyle="--",
            linewidth=0.8,
            label="Main peak",
        )
        axes[3].set_title(
            "Step 4 | Full-epoch DWT -> local reference: "
            f"{strongest.wavelet}, level {strongest.level} "
            f"(Jstart={strongest.j_start}, <= {strongest.reference_upper_hz:.2f} Hz), "
            f"P={strongest.similarity:.3f}, "
            f"F_E={strongest.energy_fidelity:.3f}, "
            f"F_f={strongest.frequency_fidelity:.3f}, "
            f"Score={strongest.selection_score:.3f}"
        )
        axes[3].legend(loc="upper right")
    else:
        axes[3].text(0.5, 0.5, "No blink interval detected", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_title("Step 4 | DWT pseudo-reference")
    axes[3].set_ylabel("Amplitude")
    axes[3].grid(alpha=0.2)

    # 5. RLS 伪影估计；clean 只在这里作为“答案对照”，算法没有使用它
    if strongest is not None:
        local_t = np.arange(strongest.end - strongest.start) / cfg.fs
        axes[4].plot(local_t, strongest.estimated_artifact, linewidth=1.1, label="RLS estimated artifact")
        if c is not None:
            true_artifact = x[strongest.start:strongest.end] - c[strongest.start:strongest.end]
            axes[4].plot(local_t, true_artifact, linewidth=0.9, label="True artifact = mixed - clean (evaluation only)")
        axes[4].set_title("Step 5 | RLS artifact estimation")
        axes[4].legend(loc="upper right")
    else:
        axes[4].text(0.5, 0.5, "No RLS correction performed", ha="center", va="center", transform=axes[4].transAxes)
        axes[4].set_title("Step 5 | RLS artifact estimation")
    axes[4].set_ylabel("Amplitude")
    axes[4].grid(alpha=0.2)

    # 6. 最终结果
    axes[5].plot(t, x, linewidth=0.65, alpha=0.65, label="Contaminated EEG")
    axes[5].plot(t, y, linewidth=0.9, label="Cleaned EEG")
    if c is not None:
        axes[5].plot(t, c, linewidth=0.8, label="Clean reference")
    axes[5].set_title("Step 6 | Final output")
    axes[5].set_xlabel("Time (s)")
    axes[5].set_ylabel("Amplitude")
    axes[5].legend(loc="upper right")
    axes[5].grid(alpha=0.2)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def metrics_text(metrics: dict, cfg: Config) -> str:
    """生成论文式评价报告：TPR + MAE + SNR + Pearson。"""
    summary = metrics["summary"]

    def value(metric: str, stage: str, decimals: int = 6) -> str:
        item = summary[metric][stage]
        return f"{item['mean']:.{decimals}f} ± {item['std']:.{decimals}f}"

    detection = summary["BlinkDetectionRate"]
    mae_change = summary["MAE"]["change"]
    snr_change = summary["SNR"]["change"]
    corr_change = summary["Correlation"]["change"]
    intervals = metrics["intervals_per_epoch"]

    lines = [
        "EEG 眨眼伪影检测与去除评价结果",
        "=" * 72,
        f"样本数（epoch）: {metrics['n_epochs']}",
        f"输入形状: {tuple(metrics['shape'])}",
        f"采样率: {cfg.fs:g} Hz",
        f"每个 epoch 时长: {metrics['shape'][-1] / cfg.fs:.3f} s",
        f"低通截止频率: {cfg.lowpass_hz:g} Hz",
        f"DWT Pmin: {cfg.minimum_similarity:.3f}",
        f"DWT 自适应 level 搜索起点 Jstart: {starting_level_from_frequency(cfg.fs, cfg.wavelet_target_hz)}",
        f"DWT 目标频带上限: {cfg.wavelet_target_hz:g} Hz",
        "DWT 选择评分: Score = P × energy_fidelity × frequency_fidelity（P>=Pmin 为硬约束）",
        f"候选母小波: {', '.join(cfg.wavelet_candidates)}",
        f"小波边界模式: {cfg.wavelet_mode}",
        "检测阈值: robust universal threshold（每个 epoch 由 MAD 与 N_eff 自动计算）",
        "",
        "眨眼伪影检测性能",
        "-" * 72,
        f"真实植入眨眼事件总数: {detection['true_events']}",
        f"TP: {detection['true_positive']}",
        f"FN: {detection['false_negative']}",
        f"FP: {detection['false_positive']}",
        f"眨眼伪影检测率 TPR: {detection['global_tpr_percent']:.2f}%",
        (
            "每 epoch TPR: "
            f"{detection['per_epoch']['mean']:.2f}% ± "
            f"{detection['per_epoch']['std']:.2f}%"
        ),
        f"算法检测区间总数: {metrics['detected_intervals']}",
        f"实际完成小波/RLS处理的区间: {metrics.get('processed_intervals', metrics['detected_intervals'])}",
        f"因 Pmin/level 可用性等条件跳过的区间: {metrics.get('skipped_intervals', 0)}",
        f"每个 epoch 检测区间数: {intervals['mean']:.3f} ± {intervals['std']:.3f}",
        "",
        "伪影抑制与信号重构指标",
        "-" * 72,
        f"{'指标':<16}{'去除前（均值 ± 标准差）':<25}{'去除后（均值 ± 标准差）':<25}{'平均变化'}",
        "-" * 72,
        f"{'MAE':<16}{value('MAE', 'before'):<25}{value('MAE', 'after'):<25}降低 {mae_change['mean']:.2f}% ± {mae_change['std']:.2f}%",
        f"{'SNR (dB)':<16}{value('SNR', 'before', 3):<25}{value('SNR', 'after', 3):<25}提升 {snr_change['mean']:.3f} ± {snr_change['std']:.3f} dB",
        f"{'Pearson':<16}{value('Correlation', 'before'):<25}{value('Correlation', 'after'):<25}提高 {corr_change['mean']:.6f} ± {corr_change['std']:.6f}",
        "-" * 72,
        f"MAE 得到改善的 epoch 比例: {100 * metrics['mae_improved_fraction']:.2f}%",
        "",
        "论文式结果表述",
        "-" * 72,
        (
            f"在 {metrics['n_epochs']} 个 paired-clean EEG epoch 上，"
            f"从 mixedEEG-cleanEEG 自动恢复得到 {detection['true_events']} 个植入眨眼事件。"
            f"算法正确检出 {detection['true_positive']} 个，漏检 {detection['false_negative']} 个，"
            f"事件级眨眼伪影检测率 TPR 为 {detection['global_tpr_percent']:.2f}%。"
        ),
        (
            f"伪影抑制后，MAE 由 {value('MAE', 'before')} 降至 {value('MAE', 'after')}，"
            f"平均降低 {mae_change['mean']:.2f}% ± {mae_change['std']:.2f}%；"
            f"SNR 由 {value('SNR', 'before', 3)} dB 提高至 {value('SNR', 'after', 3)} dB，"
            f"平均提升 {snr_change['mean']:.3f} ± {snr_change['std']:.3f} dB；"
            f"Pearson 相关系数由 {value('Correlation', 'before')} 提高至 "
            f"{value('Correlation', 'after')}。"
        ),
        "",
        "配置参数",
        "-" * 72,
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        "",
        (
            "注意：clean 仅用于评价和可视化。事件级真值区间由 mixed-clean 的非零污染区间恢复；"
            "clean 不参与眨眼检测、小波选择或 RLS 去除过程。"
        ),
    ]
    return "\n".join(lines)

def save_metrics_summary_csv(metrics: dict, output_path: Path) -> None:
    """保存论文汇总指标：TPR、MAE、SNR、Pearson。"""
    summary = metrics["summary"]
    detection = summary["BlinkDetectionRate"]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "指标",
            "去除前均值",
            "去除前标准差",
            "去除后均值",
            "去除后标准差",
            "变化/检测率",
            "变化标准差",
            "单位",
            "补充信息",
        ])
        writer.writerow([
            "Blink detection rate (TPR)",
            "", "", "", "",
            detection["global_tpr_percent"],
            detection["per_epoch"]["std"],
            "%",
            (
                f"truth={detection['true_events']}; TP={detection['true_positive']}; "
                f"FN={detection['false_negative']}; FP={detection['false_positive']}; "
                f"per_epoch_mean={detection['per_epoch']['mean']:.6f}%"
            ),
        ])

        for metric, unit in [("MAE", ""), ("SNR", "dB"), ("Correlation", "")]:
            writer.writerow([
                "Pearson" if metric == "Correlation" else metric,
                summary[metric]["before"]["mean"],
                summary[metric]["before"]["std"],
                summary[metric]["after"]["mean"],
                summary[metric]["after"]["std"],
                summary[metric]["change"]["mean"],
                summary[metric]["change"]["std"],
                unit,
                "",
            ])


def save_per_epoch_metrics_csv(metrics: dict, output_path: Path) -> None:
    """保存逐 epoch 的 TPR、MAE、SNR 和 Pearson。"""
    values = metrics["per_epoch"]
    n_epochs = metrics["n_epochs"]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "truth_blink_events",
            "true_positive",
            "false_negative",
            "false_positive",
            "blink_detection_tpr_percent",
            "mae_before",
            "mae_after",
            "mae_reduction_percent",
            "snr_before_db",
            "snr_after_db",
            "snr_improvement_db",
            "pearson_before",
            "pearson_after",
            "pearson_improvement",
        ])
        for i in range(n_epochs):
            writer.writerow([
                i,
                values["truth_blink_events"][i],
                values["true_positive"][i],
                values["false_negative"][i],
                values["false_positive"][i],
                values["blink_detection_tpr_percent"][i],
                values["mae_before"][i],
                values["mae_after"][i],
                values["mae_reduction_percent"][i],
                values["snr_before_db"][i],
                values["snr_after_db"][i],
                values["snr_improvement_db"][i],
                values["correlation_before"][i],
                values["correlation_after"][i],
                values["correlation_improvement"][i],
            ])

def save_wavelet_selection_csv(
    diagnostics: Sequence[dict],
    output_path: Path,
) -> None:
    """统计真正被选择的母小波，方便检查 dmey/level-1 偏置是否消失。"""
    rows: Dict[str, List[IntervalResult]] = {}
    total = 0
    for diag in diagnostics:
        for item in diag["intervals"]:
            rows.setdefault(item.wavelet, []).append(item)
            total += 1

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "wavelet",
            "count",
            "percentage",
            "mean_level",
            "min_level",
            "max_level",
            "mean_similarity",
            "mean_energy_retention",
            "mean_energy_fidelity",
            "mean_frequency_fidelity",
            "mean_selection_score",
        ])

        ordered = sorted(rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for wavelet, items in ordered:
            levels = np.asarray([item.level for item in items], dtype=float)
            similarities = np.asarray([item.similarity for item in items], dtype=float)
            retentions = np.asarray([item.energy_retention for item in items], dtype=float)
            fidelities = np.asarray([item.energy_fidelity for item in items], dtype=float)
            frequency_fidelities = np.asarray([item.frequency_fidelity for item in items], dtype=float)
            selection_scores = np.asarray([item.selection_score for item in items], dtype=float)
            writer.writerow([
                wavelet,
                len(items),
                100.0 * len(items) / max(total, 1),
                float(np.mean(levels)),
                int(np.min(levels)),
                int(np.max(levels)),
                float(np.mean(similarities)),
                float(np.mean(retentions)),
                float(np.mean(fidelities)),
                float(np.mean(frequency_fidelities)),
                float(np.mean(selection_scores)),
            ])


def save_level_selection_csv(
    diagnostics: Sequence[dict],
    output_path: Path,
) -> None:
    """统计最终被选择的 DWT level，直接检查 FG-AWPR-RLS 是否从 level 4 过度集中中恢复。"""
    rows: Dict[int, List[IntervalResult]] = {}
    total = 0
    for diag in diagnostics:
        for item in diag["intervals"]:
            rows.setdefault(int(item.level), []).append(item)
            total += 1

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "level",
            "reference_upper_hz",
            "count",
            "percentage",
            "mean_similarity",
            "mean_energy_fidelity",
            "mean_frequency_fidelity",
            "mean_selection_score",
        ])
        for level in sorted(rows):
            items = rows[level]
            writer.writerow([
                int(level),
                float(items[0].reference_upper_hz),
                len(items),
                100.0 * len(items) / max(total, 1),
                float(np.mean([item.similarity for item in items])),
                float(np.mean([item.energy_fidelity for item in items])),
                float(np.mean([item.frequency_fidelity for item in items])),
                float(np.mean([item.selection_score for item in items])),
            ])


def save_skipped_intervals_json(
    diagnostics: Sequence[dict],
    output_path: Path,
) -> None:
    records = []
    for row, diag in enumerate(diagnostics):
        for item in diag.get("skipped_intervals", []):
            records.append({"row": int(row), **item})
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单通道 EEG：自适应眨眼检测 + 全 epoch 多母小波/多层级频率感知评分 + RLS 去除。"
    )
    parser.add_argument(
        "--mixed",
        type=str,
        default="mixed.npy",
        help="污染 EEG 的 npy 文件名或完整路径。",
    )
    parser.add_argument(
        "--clean",
        type=str,
        default="clean.npy",
        help="无眨眼 EEG 的 npy 文件名或完整路径，仅用于评估。",
    )
    parser.add_argument("--fs", type=float, default=200.0, help="采样率，默认 200 Hz。")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认写入 mixed 文件所在目录下的 blink_removal_output。",
    )
    parser.add_argument(
        "--plot-row",
        type=int,
        default=None,
        help="详细画图的 epoch 索引；默认选择污染最严重的 epoch。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    cfg = Config(fs=args.fs)
    validate_wavelet_candidates(cfg)

    mixed_path = find_input_file(args.mixed)
    clean_path = find_input_file(args.clean) if args.clean else None

    mixed = np.load(mixed_path, allow_pickle=False)
    clean = None if clean_path is None else np.load(clean_path, allow_pickle=False)

    if clean is not None and clean.shape != mixed.shape:
        raise ValueError(
            f"mixed 形状 {mixed.shape} 与 clean 形状 {clean.shape} 不一致。"
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else mixed_path.parent / "blink_removal_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned, diagnostics = process_array(mixed, cfg)

    cleaned_path = output_dir / "eeg_blink_cleaned.npy"
    intervals_path = output_dir / "eeg_blink_intervals.json"
    config_path = output_dir / "eeg_blink_config.json"
    step_plot_path = output_dir / "eeg_blink_stepwise.png"
    wavelet_selection_path = output_dir / "eeg_blink_wavelet_selection.csv"
    level_selection_path = output_dir / "eeg_blink_level_selection.csv"
    skipped_path = output_dir / "eeg_blink_skipped_intervals.json"

    np.save(cleaned_path, cleaned)

    serializable_intervals = []
    for row, diag in enumerate(diagnostics):
        for item in diag["intervals"]:
            serializable_intervals.append(item.serializable(row))

    intervals_path.write_text(
        json.dumps(serializable_intervals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_wavelet_selection_csv(diagnostics, wavelet_selection_path)
    save_level_selection_csv(diagnostics, level_selection_path)
    save_skipped_intervals_json(diagnostics, skipped_path)

    row = choose_plot_row(mixed, clean, args.plot_row)
    save_stepwise_plot(
        mixed,
        cleaned,
        clean,
        diagnostics,
        cfg,
        row,
        step_plot_path,
    )

    print(f"输入 mixed: {mixed_path}")
    if clean_path is not None:
        print(f"评估 clean: {clean_path}")
    print(f"输入形状: {mixed.shape}")
    n_samples = int(mixed.shape[-1])
    print(f"采样率: {cfg.fs:g} Hz")
    print(f"每个 epoch 采样点数: {n_samples}")
    print(f"每个 epoch 时长: {n_samples / cfg.fs:.3f} s")
    print(f"详细图 epoch: {row}")
    print(f"保存清洁 EEG: {cleaned_path}")
    print(f"保存步骤图: {step_plot_path}")
    print(f"保存区间记录: {intervals_path}")
    print(f"保存母小波选择统计: {wavelet_selection_path}")
    print(f"保存层级选择统计: {level_selection_path}")
    print(f"保存跳过区间记录: {skipped_path}")

    if clean is not None:
        metrics = evaluate(mixed, cleaned, clean, diagnostics)
        metrics_json_path = output_dir / "eeg_blink_metrics.json"
        metrics_txt_path = output_dir / "eeg_blink_metrics.txt"
        metrics_table_path = output_dir / "eeg_blink_metrics_summary.csv"
        per_epoch_path = output_dir / "eeg_blink_metrics_by_epoch.csv"

        metrics_json_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics_txt_path.write_text(
            metrics_text(metrics, cfg),
            encoding="utf-8",
        )
        save_metrics_summary_csv(metrics, metrics_table_path)
        save_per_epoch_metrics_csv(metrics, per_epoch_path)

        summary = metrics["summary"]
        detection = summary["BlinkDetectionRate"]
        print(
            "眨眼伪影检测率 TPR: "
            f"{detection['global_tpr_percent']:.2f}% "
            f"(TP={detection['true_positive']}, FN={detection['false_negative']}, "
            f"FP={detection['false_positive']}, truth={detection['true_events']})"
        )
        print(
            "MAE（均值 ± 标准差）: "
            f"{summary['MAE']['before']['mean']:.6f} ± "
            f"{summary['MAE']['before']['std']:.6f} -> "
            f"{summary['MAE']['after']['mean']:.6f} ± "
            f"{summary['MAE']['after']['std']:.6f}"
        )
        print(
            "SNR（均值 ± 标准差）: "
            f"{summary['SNR']['before']['mean']:.3f} ± "
            f"{summary['SNR']['before']['std']:.3f} dB -> "
            f"{summary['SNR']['after']['mean']:.3f} ± "
            f"{summary['SNR']['after']['std']:.3f} dB"
        )
        print(
            "Pearson（均值 ± 标准差）: "
            f"{summary['Correlation']['before']['mean']:.6f} ± "
            f"{summary['Correlation']['before']['std']:.6f} -> "
            f"{summary['Correlation']['after']['mean']:.6f} ± "
            f"{summary['Correlation']['after']['std']:.6f}"
        )
        print(f"保存论文式评估报告: {metrics_txt_path}")
        print(f"保存平均指标表格: {metrics_table_path}")
        print(f"保存逐 epoch 指标: {per_epoch_path}")


if __name__ == "__main__":
    main()
