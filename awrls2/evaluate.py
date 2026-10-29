"""Evaluation: paired (semi-simulated) metrics, reference-free real-EEG
metrics, the five-band log-power deviation D_BP of the manuscript, the
EEGdenoiseNet metrics (RRMSE_t, RRMSE_s, CC), paired statistics and an
SNR-level generator.

The paired event metrics replicate the manuscript protocol exactly: a common
method-independent evaluation detector (12 Hz LP, 0.08 s energy window,
0.25 s minimum distance, 0.45 s half window, 0.10 s minimum duration) is run on
the known implanted artifact a = x - s and on the removed component
a_hat = x - s_hat; events are matched (peak inside the other interval or
|dt| <= 0.15 s); BARR is computed inside the truth windows padded by 0.05 s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, welch
from scipy.signal.windows import hann

from .detect import lowpass, robust_sigma

EPS = 1e-12
PAPER_BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0), "gamma": (30.0, 40.0)}


# ----------------------------------------------------------------------------
# Common evaluation detector (method independent)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Event:
    start: int
    end: int
    peak: int
    score: float


def eval_detector(x: np.ndarray, fs: float, lowpass_hz=12.0, energy_window_s=0.08, min_distance_s=0.25,
                  half_width_s=0.45, min_duration_s=0.10) -> List[Event]:
    x = np.asarray(x, dtype=float).ravel()
    lp = lowpass(x, fs, lowpass_hz, 4)
    centered = lp - np.median(lp)
    win = max(3, int(round(energy_window_s * fs)))
    energy = uniform_filter1d(centered * centered, size=win, mode="nearest")
    n_eff = max(2, int(np.ceil(len(x) / win)))
    threshold = np.median(energy) + robust_sigma(energy) * np.sqrt(2.0 * np.log(n_eff))
    distance = max(1, int(round(min_distance_s * fs)))
    cands, props = find_peaks(energy, height=threshold, distance=distance, prominence=max(robust_sigma(energy), 1e-12))
    half = max(2, int(round(half_width_s * fs)))
    min_duration = max(2, int(round(min_duration_s * fs)))
    sigma = robust_sigma(centered)
    events: List[Event] = []
    for p, h in zip(cands, props.get("peak_heights", np.zeros(len(cands)))):
        ls, le = max(0, int(p) - half), min(len(x), int(p) + half + 1)
        base = max(sigma, 0.08 * abs(centered[int(p)]), 1e-12)
        left = int(p)
        while left > ls and abs(centered[left]) > base:
            left -= 1
        right = int(p)
        while right < le - 1 and abs(centered[right]) > base:
            right += 1
        if right - left < min_duration:
            left = max(0, int(p) - min_duration // 2)
            right = min(len(x), left + min_duration)
        events.append(Event(left, right, int(p), float(h)))
    if not events:
        return []
    ordered = sorted(events, key=lambda e: (e.start, e.end))
    merged = [ordered[0]]
    for e in ordered[1:]:
        prev = merged[-1]
        if e.start <= prev.end:
            best = e if e.score > prev.score else prev
            merged[-1] = Event(prev.start, max(prev.end, e.end), best.peak, max(prev.score, e.score))
        else:
            merged.append(e)
    return merged


def match_events(reference: Sequence[Event], candidate: Sequence[Event], fs: float, tol_s: float = 0.15):
    tol = int(round(tol_s * fs))
    used: set = set()
    tp = 0
    for t in reference:
        opts = []
        for i, c in enumerate(candidate):
            if i in used:
                continue
            if c.start <= t.peak <= c.end or t.start <= c.peak <= t.end or abs(c.peak - t.peak) <= tol:
                opts.append((abs(c.peak - t.peak), i))
        if opts:
            _, i = min(opts)
            used.add(i)
            tp += 1
    return tp, len(reference) - tp, len(candidate) - len(used)


def event_mask(n: int, events: Sequence[Event], pad: int = 0) -> np.ndarray:
    m = np.zeros(n, bool)
    for e in events:
        m[max(0, e.start - pad): min(n, e.end + pad)] = True
    return m


# ----------------------------------------------------------------------------
# Spectral helpers
# ----------------------------------------------------------------------------
def band_log_power(z: np.ndarray, fs: float, win_s: float = 4.0, overlap: float = 0.5) -> Dict[str, float]:
    """Manuscript D_BP protocol: Hann windows, 50 % overlap, trapezoid band power, log, window mean."""
    z = np.asarray(z, dtype=float)
    n_win = int(round(win_s * fs))
    step = max(1, int(round(n_win * (1 - overlap))))
    starts = list(range(0, len(z) - n_win + 1, step))
    if not starts:
        starts = [0]
        n_win = len(z)
    w = hann(n_win, sym=False)
    f = np.fft.rfftfreq(n_win, 1.0 / fs)
    out = {b: [] for b in PAPER_BANDS}
    for s in starts:
        seg = z[s: s + n_win]
        seg = seg - np.mean(seg)
        spec = np.fft.rfft(seg * w)
        psd = (np.abs(spec) ** 2) / (fs * np.sum(w ** 2))
        psd[1:-1] *= 2.0
        for b, (lo, hi) in PAPER_BANDS.items():
            if lo >= fs / 2:
                continue
            hi = min(hi, fs / 2)
            grid = np.unique(np.r_[lo, f[(f > lo) & (f < hi)], hi])
            p = np.interp(grid, f, psd)
            out[b].append(10.0 * np.log10(trapezoid(p, grid) + EPS))
    return {b: float(np.mean(v)) if v else float("nan") for b, v in out.items()}


def dbp(clean: np.ndarray, cleaned: np.ndarray, fs: float) -> Dict[str, float]:
    a, b = band_log_power(clean, fs), band_log_power(cleaned, fs)
    out = {f"dbp_{k}": abs(b[k] - a[k]) for k in PAPER_BANDS}
    out["dbp_total"] = float(np.nansum([out[f"dbp_{k}"] for k in PAPER_BANDS]))
    return out


def rrmse_spectral(clean: np.ndarray, cleaned: np.ndarray, fs: float) -> float:
    nper = min(len(clean), max(64, int(round(fs))))
    _, p_ref = welch(clean, fs=fs, nperseg=nper)
    _, p_est = welch(cleaned, fs=fs, nperseg=nper)
    return float(np.sqrt(np.mean((p_est - p_ref) ** 2)) / (np.sqrt(np.mean(p_ref ** 2)) + EPS))


# ----------------------------------------------------------------------------
# Paired metrics (semi-simulated data with clean reference)
# ----------------------------------------------------------------------------
def support_events(artifact: np.ndarray, fs: float, merge_gap_s: float = 0.10, min_peak_fraction: float = 0.0) -> List[Event]:
    """Exact truth events from the support of the implanted artifact (a = x - s is
    identically zero outside the inserted blink windows).  Runs closer than
    ``merge_gap_s`` are merged; runs whose peak is below ``min_peak_fraction`` of the
    largest run peak are ignored (0 keeps every run)."""
    a = np.asarray(artifact, dtype=float)
    scale = max(float(np.max(np.abs(a))), 1.0)
    nz = np.abs(a) > 100.0 * np.finfo(float).eps * scale
    edges = np.flatnonzero(np.diff(np.r_[False, nz, False].astype(np.int8)))
    runs = [(int(s), int(e)) for s, e in zip(edges[0::2], edges[1::2]) if e > s]
    gap = int(round(merge_gap_s * fs))
    merged: List[Tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    events = []
    for s, e in merged:
        p = s + int(np.argmax(np.abs(a[s:e])))
        events.append(Event(s, e, p, float(abs(a[p]))))
    if events and min_peak_fraction > 0:
        top = max(ev.score for ev in events)
        events = [ev for ev in events if ev.score >= min_peak_fraction * top]
    return events


def _event_metrics(truth: List[Event], removed: List[Event], fs: float, prefix: str) -> Dict[str, float]:
    tp, fn, fp = match_events(truth, removed, fs)
    return {f"{prefix}true_events": len(truth), f"{prefix}tp": tp, f"{prefix}fn": fn, f"{prefix}fp": fp,
            f"{prefix}recall_percent": 100.0 * tp / max(len(truth), 1),
            f"{prefix}precision_percent": 100.0 * tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            f"{prefix}f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def paired_metrics(mixed: np.ndarray, clean: np.ndarray, cleaned: np.ndarray, fs: float,
                   truth_mode: str = "support", removed_amplitude_floor: float = 0.10) -> Dict[str, float]:
    """Paired metrics.

    truth_mode="support"  : truth events = exact support of the implanted artifact
                            (recommended; the manuscript protocol is also reported
                            with the ``paper_`` prefix for comparability).
    truth_mode="detector" : manuscript protocol only (common detector on a = x - s).
    removed_amplitude_floor: removed-component events whose peak is below this
                            fraction of the largest removed peak are not counted
                            (suppresses counting of tiny secondary lobes as FPs).
    """
    mixed, clean, cleaned = (np.asarray(v, dtype=float) for v in (mixed, clean, cleaned))
    a = mixed - clean
    a_hat = mixed - cleaned
    e = cleaned - clean
    truth_paper = eval_detector(a, fs)
    removed_paper = eval_detector(a_hat, fs)
    out: Dict[str, float] = {}
    out.update(_event_metrics(truth_paper, removed_paper, fs, "paper_"))
    if truth_mode == "support":
        truth = support_events(a, fs)
        removed = removed_paper
        if removed and removed_amplitude_floor > 0:
            top = max(abs(a_hat[ev.peak]) for ev in removed)
            removed = [ev for ev in removed if abs(a_hat[ev.peak]) >= removed_amplitude_floor * top]
        out.update(_event_metrics(truth, removed, fs, ""))
    else:
        truth = truth_paper
        out.update(_event_metrics(truth_paper, removed_paper, fs, ""))
    mask = event_mask(len(mixed), truth, int(round(0.05 * fs)))
    if not mask.any():
        mask = np.ones(len(mixed), bool)
    e_before, e_after = float(np.sum(a[mask] ** 2)), float(np.sum(e[mask] ** 2))
    snr_before = 10 * np.log10((np.sum(clean ** 2) + EPS) / (np.sum(a ** 2) + EPS))
    snr_after = 10 * np.log10((np.sum(clean ** 2) + EPS) / (np.sum(e ** 2) + EPS))
    rms_clean = np.sqrt(np.mean(clean ** 2)) + EPS
    out.update({
        "barr_percent": 100.0 * (1 - e_after / (e_before + EPS)),
        "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae_before": float(np.mean(np.abs(a))), "rmse_before": float(np.sqrt(np.mean(a ** 2))),
        "snr_db": float(snr_after), "snr_before_db": float(snr_before), "delta_snr_db": float(snr_after - snr_before),
        "cc": float(np.corrcoef(clean, cleaned)[0, 1]) if np.std(cleaned) > 0 else np.nan,
        "rrmse_t": float(np.sqrt(np.mean(e ** 2)) / rms_clean),
        "rrmse_s": rrmse_spectral(clean, cleaned, fs),
        "rmse_in_window": float(np.sqrt(np.mean(e[mask] ** 2))),
        "rmse_out_window": float(np.sqrt(np.mean(e[~mask] ** 2))) if (~mask).any() else np.nan,
        "changed_out_window_percent": 100.0 * float(np.mean(np.abs(e[~mask] - a[~mask]) > 1e-9)) if (~mask).any() else np.nan,
    })
    out.update(dbp(clean, cleaned, fs))
    return out


def clean_control_metrics(clean: np.ndarray, processed_clean: np.ndarray) -> Dict[str, float]:
    d = np.asarray(processed_clean, float) - np.asarray(clean, float)
    return {"clean_altered": float(np.any(np.abs(d) > 1e-9)),
            "clean_altered_sample_percent": 100.0 * float(np.mean(np.abs(d) > 1e-9)),
            "clean_rmse_change": float(np.sqrt(np.mean(d ** 2)))}


# ----------------------------------------------------------------------------
# Reference-free metrics for real EEG (manuscript Table 4)
# ----------------------------------------------------------------------------
def real_metrics(original: np.ndarray, cleaned: np.ndarray, fs: float) -> Dict[str, float]:
    original, cleaned = np.asarray(original, float), np.asarray(cleaned, float)
    events = eval_detector(original, fs)
    blink = event_mask(len(original), events, int(round(0.05 * fs)))
    nonblink = ~event_mask(len(original), events, int(round(0.20 * fs)))
    lp_in, lp_out = lowpass(original, fs, 12.0), lowpass(cleaned, fs, 12.0)
    out = {"candidate_count": len(events)}
    if blink.any():
        out["lf_energy_reduction_percent"] = 100.0 * (1 - np.sum(lp_out[blink] ** 2) / (np.sum(lp_in[blink] ** 2) + EPS))
    else:
        out["lf_energy_reduction_percent"] = np.nan
    if nonblink.sum() > 8:
        out["outside_window_cc"] = float(np.corrcoef(original[nonblink], cleaned[nonblink])[0, 1])
        out["outside_window_nrmse_percent"] = 100.0 * float(np.sqrt(np.mean((cleaned[nonblink] - original[nonblink]) ** 2)) / (np.sqrt(np.mean(original[nonblink] ** 2)) + EPS))
    else:
        out["outside_window_cc"] = np.nan
        out["outside_window_nrmse_percent"] = np.nan
    hp_in, hp_out = original - lowpass(original, fs, 13.0), cleaned - lowpass(cleaned, fs, 13.0)
    out["hf_13_40_cc"] = float(np.corrcoef(hp_in, hp_out)[0, 1]) if np.std(hp_out) > 0 else np.nan
    return out


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    gt = sum(np.sum(x > b) for x in a)
    lt = sum(np.sum(x < b) for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def paired_comparison(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """x: proposed, y: comparator, same epochs. Wilcoxon + bootstrap CI + effect sizes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = x - y
    out = {"n": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan}
    if len(d) >= 5 and np.any(d != 0):
        try:
            w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_p"] = float(w.pvalue)
        except ValueError:
            out["wilcoxon_p"] = np.nan
        rng = np.random.default_rng(seed)
        boots = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)])
        out["ci95_low"], out["ci95_high"] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        pos, neg = np.sum(d > 0), np.sum(d < 0)
        out["sign_balance_original"] = float((pos - neg) / max(pos + neg, 1))
        nonzero = d[d != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        out["rank_biserial"] = float(np.sum(ranks * np.sign(nonzero)) / np.sum(ranks))
        out["cliffs_delta"] = cliffs_delta(x, y)
        out["win_rate_percent"] = 100.0 * float(np.mean(d > 0))
    elif len(d) >= 5 and np.all(d == 0):
        out.update(wilcoxon_p=1.0, ci95_low=0.0, ci95_high=0.0, rank_biserial=0.0,
                   sign_balance_original=0.0, cliffs_delta=0.0, win_rate_percent=0.0)
    else:
        out.update(wilcoxon_p=np.nan, ci95_low=np.nan, ci95_high=np.nan, rank_biserial=np.nan,
                   cliffs_delta=np.nan, win_rate_percent=np.nan)
    return out


def holm_correction(pvalues: Sequence[float]) -> List[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = np.sum(np.isfinite(p))
    adjusted = np.full_like(p, np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


# ----------------------------------------------------------------------------
# SNR-level generator (EEGdenoiseNet style) from paired mixed/clean epochs
# ----------------------------------------------------------------------------
def rescale_to_snr(clean: np.ndarray, mixed: np.ndarray, snr_db: float) -> np.ndarray:
    """Return clean + lambda*(mixed-clean) with the artifact scaled to the requested SNR."""
    clean, mixed = np.asarray(clean, float), np.asarray(mixed, float)
    a = mixed - clean
    p_s, p_a = np.sum(clean ** 2, axis=-1, keepdims=True), np.sum(a ** 2, axis=-1, keepdims=True) + EPS
    lam = np.sqrt(p_s / (p_a * 10 ** (snr_db / 10.0)))
    return clean + lam * a


def input_snr_db(clean: np.ndarray, mixed: np.ndarray) -> np.ndarray:
    clean, mixed = np.asarray(clean, float), np.asarray(mixed, float)
    a = mixed - clean
    return 10 * np.log10((np.sum(clean ** 2, axis=-1) + EPS) / (np.sum(a ** 2, axis=-1) + EPS))
