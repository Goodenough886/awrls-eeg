"""Target-only OVME-HHO-REG reimplementation of Silpa & Hota (2024).

VME and HHO are ports of the authors' MATLAB implementations. See
official_ovme/{VME,HHO}_LICENSE.txt and docs/OVME_REG_REPRODUCTION.md.
Equation (10) is ambiguous: the primary objective uses the stated MSE;
``printed_mean`` preserves the unsquared expression for sensitivity checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gamma, pi, sin
import time

import numpy as np

from .base import MethodOutput


@dataclass(frozen=True)
class OVMEConfig:
    alpha_min: float = 1000.0
    alpha_max: float = 10000.0
    omega_min_hz: float = 0.5
    omega_max_hz: float = 7.5
    population: int = 30
    iterations: int = 20
    vme_max_iterations: int = 300
    vme_tolerance: float = 1e-7
    vme_tau: float = 0.0
    objective: str = "stated_mse"
    polarity: str = "positive"
    before_peak_s: float = 0.125
    after_peak_s: float = 0.375
    block_s: float = 3.0
    overlap_s: float = 1.0
    seed: int = 20260926

    def validate(self, fs):
        numeric = [fs, self.alpha_min, self.alpha_max, self.omega_min_hz,
                   self.omega_max_hz, self.vme_tolerance, self.vme_tau,
                   self.before_peak_s, self.after_peak_s, self.block_s, self.overlap_s]
        if not np.all(np.isfinite(numeric)) or fs <= 0:
            raise ValueError("All numeric settings and fs must be finite; fs > 0")
        if not 0 < self.alpha_min < self.alpha_max:
            raise ValueError("Invalid alpha bounds")
        if not 0 <= self.omega_min_hz < self.omega_max_hz < fs / 2:
            raise ValueError("Frequency bounds must be below Nyquist")
        for value, minimum in [(self.population, 2), (self.iterations, 1),
                               (self.vme_max_iterations, 2), (self.seed, 0)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError("Invalid integer setting")
        if self.vme_tolerance <= 0 or self.vme_tau < 0:
            raise ValueError("Invalid VME numerical settings")
        if not 0 <= self.overlap_s < self.block_s or self.block_s * fs < 4:
            raise ValueError("Invalid block/overlap duration")
        if min(self.before_peak_s, self.after_peak_s) < 0:
            raise ValueError("Negative blink window duration")
        if self.objective not in ("stated_mse", "printed_mean"):
            raise ValueError("Unsupported objective; clean EEG is never an input")
        if self.polarity not in ("positive", "absolute"):
            raise ValueError("Unsupported peak polarity")


def _vector(x):
    values = np.asarray(x)
    if values.ndim != 1 or values.size < 2 or np.iscomplexobj(values):
        raise ValueError("Expected one real target-channel vector with >= 2 samples")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("EEG must be finite")
    return values


class VMEWorkspace:
    """Cache a single target block's FFT; keep only the current ADMM state.

    Copyright (c) 2020, Mojtaba Nazari. BSD license in official_ovme.
    The v2 MATLAB quartic kernel and complex-square stopping rule are retained.
    Odd-length input is edge-padded by one sample, then restored on return.
    """
    def __init__(self, x):
        x = _vector(x)
        self.length = x.size
        if x.size % 2:
            x = np.pad(x, (0, 1), mode="edge")
        self.even_length = len(x)
        half = x.size // 2
        extended = np.concatenate((x[:half][::-1], x, x[half:][::-1]))
        self.nfft = extended.size
        # Negative-frequency iterates remain zero, so store only the positive half.
        self.frequency = (np.arange(1, self.nfft + 1) / self.nfft - 0.5
                          - 1.0 / self.nfft)[self.nfft // 2:]
        self.spectrum = np.fft.fftshift(np.fft.fft(extended))[self.nfft // 2:]

    def extract(self, alpha, omega_hz, fs, *, tau=0.0, tolerance=1e-7, max_iterations=300):
        if not np.all(np.isfinite([alpha, omega_hz, fs, tau, tolerance])):
            raise ValueError("Nonfinite VME parameters")
        if alpha <= 0 or fs <= 0 or not 0 <= omega_hz < fs / 2 or tau < 0 or tolerance <= 0:
            raise ValueError("Invalid VME parameters")
        if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 2:
            raise ValueError("VME max_iterations must be an integer >= 2")
        mode = np.zeros_like(self.spectrum)
        dual = np.zeros_like(mode)
        omega = omega_hz / fs
        history = [float(omega)]
        change = float("inf")
        for step in range(1, max_iterations):
            difference2 = (self.frequency - omega) ** 2
            kernel = alpha ** 2 * difference2 ** 2
            current = (self.spectrum + mode * kernel + dual / 2) / ((1 + kernel) * (1 + 2 * kernel))
            power = np.abs(current) ** 2
            total = float(power.sum())
            if total > 0:
                omega = float(np.dot(self.frequency, power) / total)
            if tau:
                kernel_new = alpha ** 2 * (self.frequency - omega) ** 4
                residual = kernel_new * (self.spectrum - current) / (1 + 2 * kernel_new)
                dual += tau * (self.spectrum - current - residual)
            delta = current - mode
            # MATLAB delta*conj(delta)' uses complex squares, not |delta|**2.
            change = float(abs(np.finfo(float).eps + np.dot(delta, delta) / self.nfft))
            mode = current
            history.append(float(omega))
            if change <= tolerance:
                break
        spectrum = np.zeros(self.nfft, dtype=complex)
        half = self.nfft // 2
        spectrum[half:] = mode
        spectrum[np.arange(half, 0, -1)] = mode.conj()
        spectrum[0] = spectrum[-1].conj()
        restored = np.fft.ifft(np.fft.ifftshift(spectrum)).real
        restored = restored[self.nfft // 4:3 * self.nfft // 4][:self.length].copy()
        if not np.all(np.isfinite(restored)):
            raise FloatingPointError("VME produced nonfinite samples")
        return restored, {"iterations": step, "converged": change <= tolerance,
                          "criterion": change, "omega_final_hz": omega * fs,
                          "omega_history_normalized": history}


def hho_minimize(objective, lower, upper, population, iterations, seed):
    """Bounded HHO port; Copyright (c) 2019 Ali Asghar Heidari (MIT).

    Bounds are enforced before ALL evaluations, including rapid dives (the
    published MATLAB code only clamps at the next population evaluation).
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.ndim != 1 or lower.shape != upper.shape or not np.all(np.isfinite([lower, upper])):
        raise ValueError("Invalid HHO bounds")
    if np.any(upper <= lower) or population < 2 or iterations < 1:
        raise ValueError("Invalid HHO search size")
    rng = np.random.default_rng(seed)
    position = rng.uniform(lower, upper, size=(population, len(lower)))
    best = position[0].copy()
    energy = float("inf")
    evaluations = 0
    trace = []
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2)
             / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    def evaluate(point):
        nonlocal evaluations
        bounded = np.clip(point, lower, upper)
        score = float(objective(bounded))
        evaluations += 1
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite HHO objective")
        return bounded, score

    for iteration in range(iterations):
        for i in range(population):
            position[i], score = evaluate(position[i])
            if score < energy:
                energy, best = score, position[i].copy()
        energy_scale = 2 * (1 - iteration / iterations)
        for i in range(population):
            escape = energy_scale * (2 * rng.random() - 1)
            if abs(escape) >= 1:
                q = rng.random()
                other = position[int(rng.integers(population))].copy()
                if q < 0.5:
                    position[i] = other - rng.random() * abs(other - 2 * rng.random() * position[i])
                else:
                    position[i] = best - position.mean(axis=0) - rng.random() * (lower + rng.random() * (upper - lower))
            else:
                r = rng.random()
                if r >= 0.5:
                    if abs(escape) < 0.5:
                        position[i] = best - escape * abs(best - position[i])
                    else:
                        jump = 2 * (1 - rng.random())
                        position[i] = best - position[i] - escape * abs(jump * best - position[i])
                else:
                    jump = 2 * (1 - rng.random())
                    base = position[i] if abs(escape) >= 0.5 else position.mean(axis=0)
                    candidate = best - escape * abs(jump * best - base)
                    candidate, trial = evaluate(candidate)
                    _, old = evaluate(position[i])
                    if trial < old:
                        position[i] = candidate
                    else:
                        # Uniform multiplier and Mantegna Levy flight match HHO.m.
                        multiplier = rng.random(len(lower))
                        flight = (rng.normal(size=len(lower)) * sigma
                                  / np.maximum(abs(rng.normal(size=len(lower))), np.finfo(float).tiny) ** (1 / beta))
                        candidate, trial = evaluate(best - escape * abs(jump * best - base) + multiplier * flight)
                        if trial < old:
                            position[i] = candidate
            position[i] = np.clip(position[i], lower, upper)
        trace.append(float(energy))
    # As in HHO.m, return the best evaluated population (not an unevaluated move).
    return best, {"fitness": float(energy), "convergence": trace, "evaluations": evaluations}


def objective_value(mode, interpretation):
    if interpretation == "stated_mse":
        return float(np.mean(np.square(mode)))
    if interpretation == "printed_mean":
        return float(np.mean(mode))
    raise ValueError("Unknown Eq. (10) interpretation")


def blink_intervals(mode, fs, config):
    mode = _vector(mode)
    threshold = float(np.median(np.abs(mode)) / 0.6745 * np.sqrt(2 * np.log(len(mode))))
    detection = mode if config.polarity == "positive" else np.abs(mode)
    peaks = np.flatnonzero((detection[1:-1] > detection[:-2])
                           & (detection[1:-1] > detection[2:])
                           & (detection[1:-1] > threshold)) + 1
    pre = int(np.floor(config.before_peak_s * fs + 0.5))
    post = int(np.floor(config.after_peak_s * fs + 0.5))
    windows = []
    for peak in peaks:
        start, end = max(0, int(peak) - pre), min(len(mode), int(peak) + post + 1)
        # Merge overlapping windows so a sample is not subtracted twice.
        if windows and start < windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    return peaks.tolist(), windows, threshold


def regress_intervals(x, mode, intervals):
    x, mode = _vector(x), _vector(mode)
    if x.shape != mode.shape:
        raise ValueError("Mismatched EEG and VME lengths")
    correction = np.zeros_like(x)
    coefficients = []
    for start, end in intervals:
        if not 0 <= start < end <= len(x):
            raise ValueError("Invalid regression interval")
        eeg, reference = x[start:end], mode[start:end]
        centered = reference - reference.mean()
        denominator = float(np.dot(centered, centered))
        coefficient = (float(np.dot(eeg - eeg.mean(), centered)) / denominator
                       if denominator > 0 else 0.0)
        correction[start:end] = coefficient * reference
        coefficients.append(coefficient)
    return correction, coefficients


def _clean_block(x, fs, cfg, seed):
    if np.ptp(x) == 0:
        return np.zeros_like(x), {"skipped": "constant", "peak_count": 0, "intervals": []}
    workspace = VMEWorkspace(x)
    cache = {}
    total_iterations = 0
    nonconverged = 0

    def extract(parameters):
        return workspace.extract(float(parameters[0]), float(parameters[1]), fs,
                                 tau=cfg.vme_tau, tolerance=cfg.vme_tolerance,
                                 max_iterations=cfg.vme_max_iterations)

    def objective(parameters):
        nonlocal total_iterations, nonconverged
        key = tuple(float(value) for value in parameters)
        if key not in cache:
            mode, audit = extract(parameters)
            total_iterations += audit["iterations"]
            nonconverged += int(not audit["converged"])
            cache[key] = objective_value(mode, cfg.objective)
        return cache[key]

    started = time.perf_counter()
    parameters, optimization = hho_minimize(
        objective, [cfg.alpha_min, cfg.omega_min_hz], [cfg.alpha_max, cfg.omega_max_hz],
        cfg.population, cfg.iterations, seed)
    mode, vme = extract(parameters)
    peaks, intervals, threshold = blink_intervals(mode, fs, cfg)
    correction, coefficients = regress_intervals(x, mode, intervals)
    return correction, {"alpha": float(parameters[0]), "omega_initial_hz": float(parameters[1]),
                        "optimization": optimization, "vme": vme,
                        "unique_vme_evaluations": len(cache), "search_vme_iterations": total_iterations,
                        "search_nonconverged": nonconverged, "peaks": peaks,
                        "peak_count": len(peaks), "intervals": intervals,
                        "threshold": threshold, "regression_coefficients": coefficients,
                        "wall_seconds": time.perf_counter() - started, "seed": int(seed)}


def ovme_reg_clean(x, fs, config=None):
    """Only x, its sampling rate and fixed settings enter the entire pipeline."""
    values = _vector(x)
    cfg = config if isinstance(config, OVMEConfig) else OVMEConfig(**(config or {}))
    cfg.validate(fs)
    size = max(4, int(np.floor(cfg.block_s * fs + 0.5)))
    overlap = int(np.floor(cfg.overlap_s * fs + 0.5))
    if overlap >= size:
        raise ValueError("Rounded overlap must be shorter than a block")
    if len(values) <= size:
        starts = [0]
    else:
        starts = list(range(0, len(values) - size + 1, size - overlap))
        if starts[-1] != len(values) - size:
            starts.append(len(values) - size)
    summed = np.zeros_like(values)
    weight = np.zeros_like(values)
    audits = []
    for number, start in enumerate(starts):
        end = min(len(values), start + size)
        seed = int(np.random.SeedSequence([cfg.seed, number]).generate_state(1)[0])
        correction, audit = _clean_block(values[start:end], fs, cfg, seed)
        summed[start:end] += correction
        weight[start:end] += 1
        audits.append({"start": start, "end": end, **audit})
    correction = summed / weight
    cleaned = values - correction
    if not np.all(np.isfinite(cleaned)):
        raise FloatingPointError("Nonfinite OVME-REG output")
    return MethodOutput(cleaned, correction, {
        "method": "OVME-HHO-REG", "implementation": "paper_reimplementation_with_declared_ambiguities",
        "numeric_input_channels": 1, "auxiliary_channel_count": 0,
        "uses_clean_reference": False, "uses_external_artifact_reference": False,
        "config": asdict(cfg), "blocks": audits,
        "peak_count_including_overlap_duplicates": sum(b["peak_count"] for b in audits),
        "block_count": len(audits), "changed_samples": int(np.count_nonzero(correction)),
    })
