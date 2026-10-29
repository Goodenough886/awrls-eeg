"""Regression tests for AWRLS-2.  Run:  python -m pytest tests -q   (or python -m unittest)"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from awrls2 import Config, V1_REPRODUCTION, ablation_variants, clean, evaluate, process_epoch, wavelet_bank  # noqa: E402


def synthetic(fs=200.0, seconds=12.0, seed=0, n_blinks=4, amp=120.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    # coloured background: 1/f-like EEG with alpha
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / fs)
    spec *= 1.0 / np.sqrt(np.maximum(f, 0.5))
    eeg = np.fft.irfft(spec, n)
    eeg = 6.0 * eeg / np.std(eeg) + 3.0 * np.sin(2 * np.pi * 10 * t)
    artifact = np.zeros(n)
    peaks = np.linspace(1.5, seconds - 1.5, n_blinks)
    for p in peaks:
        artifact += amp * np.exp(-0.5 * ((t - p) / 0.09) ** 2) * (1 - 0.3 * np.tanh((t - p) / 0.1))
    artifact[np.abs(artifact) < 1e-3] = 0.0
    return eeg + artifact, eeg


class WaveletBankTest(unittest.TestCase):
    def test_self_test(self):
        rep = wavelet_bank.self_test()
        for k, v in rep.items():
            if k.endswith("pr_error"):
                self.assertLess(v, 1e-9)
            if k.endswith("dc_gain"):
                self.assertAlmostEqual(v, 1.0, places=6)

    def test_swt_is_shift_invariant(self):
        x = np.random.default_rng(1).standard_normal(1024)
        a = wavelet_bank.swt_approximation(x, "db4", 3)
        b = wavelet_bank.swt_approximation(np.roll(x, 5), "db4", 3)
        margin = 160  # beyond the kernel support, the response must be exactly shifted
        self.assertLess(np.max(np.abs(np.roll(a, 5)[margin:-margin] - b[margin:-margin])), 1e-8)


class PipelineTest(unittest.TestCase):
    def test_v1_and_proposed_remove_synthetic_blinks(self):
        x, s = synthetic()
        for name in ("V1_reproduction", "PROPOSED"):
            cfg = Config(fs=200.0).with_updates(**ablation_variants()[name])
            y, diag = process_epoch(x, cfg)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(y)))
            m = evaluate.paired_metrics(x, s, y, 200.0)
            self.assertGreaterEqual(diag["processed_count"], 3, name)
            self.assertGreater(m["delta_snr_db"], 8.0, name)
            self.assertGreater(m["barr_percent"], 80.0, name)

    def test_proposed_not_worse_than_v1_on_synthetic(self):
        x, s = synthetic(seed=3)
        v1 = process_epoch(x, Config(fs=200.0).with_updates(**V1_REPRODUCTION))[0]
        pr = process_epoch(x, Config(fs=200.0))[0]
        self.assertLessEqual(np.sqrt(np.mean((pr - s) ** 2)), 1.15 * np.sqrt(np.mean((v1 - s) ** 2)))

    def test_outside_windows_unchanged(self):
        x, s = synthetic(seed=2)
        y, diag = process_epoch(x, Config(fs=200.0))
        mask = np.zeros(len(x), bool)
        for e in diag["events"]:
            mask[e["start"]:e["end"]] = True
        self.assertTrue(np.array_equal(y[~mask], x[~mask]))

    def test_every_variant_runs(self):
        x, _ = synthetic(seed=4, seconds=8.0, n_blinks=3)
        for name, ov in ablation_variants().items():
            y, _ = process_epoch(x, Config(fs=200.0).with_updates(**ov))
            self.assertTrue(np.all(np.isfinite(y)), name)

    def test_block_processing_matches_shape(self):
        x, s = synthetic(seed=5, seconds=95.0, n_blinks=30)
        y, diag = clean(x, 200.0, Config(fs=200.0), block_s=30, overlap_s=4)
        self.assertEqual(y.shape, x.shape)
        self.assertGreater(diag["processed_count"], 20)
        self.assertGreater(evaluate.paired_metrics(x, s, y, 200.0)["delta_snr_db"], 8.0)

    def test_clean_input_mostly_untouched(self):
        _, s = synthetic(seed=6)
        y, _ = process_epoch(s, Config(fs=200.0))
        ctrl = evaluate.clean_control_metrics(s, y)
        self.assertLess(ctrl["clean_altered_sample_percent"], 5.0)

    def test_constant_and_short_inputs(self):
        y, d = process_epoch(np.zeros(400), Config(fs=200.0))
        self.assertEqual(d["processed_count"], 0)
        y, d = process_epoch(np.random.default_rng(0).standard_normal(30), Config(fs=200.0))
        self.assertEqual(y.shape, (30,))


class EvaluationTest(unittest.TestCase):
    def test_perfect_output(self):
        x, s = synthetic(seed=7)
        m = evaluate.paired_metrics(x, s, s.copy(), 200.0)
        self.assertAlmostEqual(m["mae"], 0.0)
        self.assertAlmostEqual(m["barr_percent"], 100.0)
        self.assertEqual(m["recall_percent"], 100.0)
        self.assertEqual(m["true_events"], 4)

    def test_support_events_exact(self):
        x, s = synthetic(seed=8, n_blinks=5)
        ev = evaluate.support_events(x - s, 200.0)
        self.assertEqual(len(ev), 5)

    def test_snr_rescaling(self):
        x, s = synthetic(seed=9)
        y = evaluate.rescale_to_snr(s[None], x[None], -3.0)[0]
        self.assertAlmostEqual(float(evaluate.input_snr_db(s[None], y[None])[0]), -3.0, places=6)

    def test_statistics(self):
        a = np.random.default_rng(0).standard_normal(50) + 0.5
        b = np.random.default_rng(1).standard_normal(50)
        st = evaluate.paired_comparison(a, b)
        self.assertLess(st["wilcoxon_p"], 0.05)
        self.assertEqual(len(evaluate.holm_correction([0.01, 0.04, 0.03])), 3)


if __name__ == "__main__":
    unittest.main()
