"""Numerical, channel-contract and boundary tests for the OVME port."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from blink_repro.methods.ovme_reg import (
    OVMEConfig, VMEWorkspace, blink_intervals, hho_minimize,
    objective_value, ovme_reg_clean, regress_intervals,
)


def matlab_vme_reference(x, alpha, omega_hz, fs, tau=0, tol=1e-7, maximum=300):
    """Direct full-spectrum translation of archived VME v2, independent of cache."""
    half = len(x) // 2
    f = np.concatenate([x[:half][::-1], x, x[half:][::-1]])
    size = len(f)
    frequency = np.arange(1, size + 1) / size - .5 - 1 / size
    spectrum = np.fft.fftshift(np.fft.fft(f))
    spectrum[:size // 2] = 0
    modes = np.zeros((maximum, size), complex)
    dual = np.zeros_like(modes)
    omega = np.zeros(maximum)
    omega[0] = omega_hz / fs
    change, n = tol + np.finfo(float).eps, 0
    while change > tol and n < maximum - 1:
        k = alpha ** 2 * (frequency - omega[n]) ** 4
        modes[n + 1] = (spectrum + modes[n] * k + dual[n] / 2) / ((1 + k) * (1 + 2 * k))
        power = abs(modes[n + 1, size // 2:]) ** 2
        omega[n + 1] = np.dot(frequency[size // 2:], power) / power.sum()
        k = alpha ** 2 * (frequency - omega[n + 1]) ** 4
        dual[n + 1] = dual[n] + tau * (spectrum - modes[n + 1] - k * (spectrum - modes[n + 1]) / (1 + 2 * k))
        n += 1
        delta = modes[n] - modes[n - 1]
        change = abs(np.finfo(float).eps + np.dot(delta, delta) / size)
    full = np.zeros(size, complex)
    full[size // 2:] = modes[n, size // 2:]
    full[np.arange(size // 2, 0, -1)] = modes[n, size // 2:].conj()
    full[0] = full[-1].conj()
    mode = np.fft.ifft(np.fft.ifftshift(full)).real[size // 4:3 * size // 4]
    return mode, omega[:n + 1]


class TestVME(unittest.TestCase):
    def test_reference_agreement(self):
        rng = np.random.default_rng(321)
        x = rng.normal(size=600) + 8 * np.exp(-((np.arange(600) - 290) / 25) ** 2)
        for tau in [0, .2]:
            for alpha, omega in [(1000, .5), (9492, 2.3), (10000, 7.5)]:
                with self.subTest(tau=tau, alpha=alpha):
                    expected, frequencies = matlab_vme_reference(x, alpha, omega, 200, tau=tau)
                    actual, audit = VMEWorkspace(x).extract(alpha, omega, 200, tau=tau)
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(audit['omega_history_normalized'], frequencies, rtol=1e-10, atol=1e-12)

    def test_zero_mode_and_odd_length(self):
        y, audit = VMEWorkspace(np.zeros(601)).extract(1000, 1, 200)
        self.assertEqual(y.shape, (601,))
        self.assertTrue(np.all(y == 0))
        self.assertTrue(audit['converged'])

    def test_published_vme_example(self):
        t = np.arange(1, 1001) / 1000
        desired = np.cos(30*np.pi*t) * (1+np.cos(2*np.pi*t)) / 2
        x = 2*np.cos(4*np.pi*t) + desired + np.cos(80*np.pi*t) * (1+np.sin(2*np.pi*t))/2
        mode, _ = VMEWorkspace(x).extract(20000, 10, 1000)
        self.assertGreater(np.corrcoef(mode, desired)[0, 1], .9)

    def test_bad_vme_parameters(self):
        workspace = VMEWorkspace(np.ones(10))
        for alpha, omega, fs in [(0, 1, 200), (1000, 101, 200), (np.nan, 1, 200), (1000, 1, 0)]:
            with self.assertRaises(ValueError):
                workspace.extract(alpha, omega, fs)


class TestDetectionRegression(unittest.TestCase):
    def test_equation_12_and_windows(self):
        mode = np.zeros(600)
        mode[200] = 10
        peaks, windows, threshold = blink_intervals(mode, 200, OVMEConfig())
        self.assertEqual(peaks, [200])
        self.assertEqual(windows, [[175, 276]])
        self.assertEqual(threshold, 0)

    def test_overlap_merged_and_edges_clipped(self):
        mode = np.zeros(600)
        mode[[1, 50, 598]] = 10
        _, windows, _ = blink_intervals(mode, 200, OVMEConfig())
        self.assertEqual(windows, [[0, 126], [573, 600]])

    def test_positive_only_is_paper_default(self):
        mode = np.zeros(600)
        mode[300] = -10
        self.assertEqual(blink_intervals(mode, 200, OVMEConfig())[0], [])
        self.assertEqual(blink_intervals(mode, 200, replace(OVMEConfig(), polarity='absolute'))[0], [300])

    def test_regression_formula_and_untouched_region(self):
        mode = np.linspace(-1, 2, 100)
        eeg = 3 * mode + 7
        correction, coefficients = regress_intervals(eeg, mode, [[20, 60]])
        self.assertAlmostEqual(coefficients[0], 3)
        np.testing.assert_allclose(correction[20:60], 3 * mode[20:60])
        np.testing.assert_array_equal(correction[:20], 0)
        np.testing.assert_array_equal(correction[60:], 0)

    def test_constant_reference_safe(self):
        correction, coefficients = regress_intervals(np.arange(10.), np.ones(10), [[0, 10]])
        np.testing.assert_array_equal(correction, 0)
        self.assertEqual(coefficients, [0])

    def test_objective_ambiguity_explicit(self):
        x = np.array([-2., 0, 2])
        self.assertAlmostEqual(objective_value(x, 'stated_mse'), 8/3)
        self.assertEqual(objective_value(x, 'printed_mean'), 0)


class TestContracts(unittest.TestCase):
    def test_hho_bounds_reproducibility(self):
        points = []
        def sphere(x):
            self.assertTrue(np.all(x >= 1) and np.all(x <= 5))
            points.append(x.copy())
            return np.sum((x-2)**2)
        a, audit = hho_minimize(sphere, [1, 1], [5, 5], 8, 8, 123)
        b, again = hho_minimize(sphere, [1, 1], [5, 5], 8, 8, 123)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(audit, again)
        self.assertTrue(np.all(np.diff(audit['convergence']) <= 0))
        self.assertAlmostEqual(audit['fitness'], sphere(a))

    def test_invalid_input_rejected(self):
        for x in [[], [1], [[1, 2]], np.ones((2, 100)), [np.nan, 1], [np.inf, 0], [1+1j, 2]]:
            with self.assertRaises(ValueError):
                ovme_reg_clean(x, 200)

    def test_invalid_config_rejected(self):
        for cfg in [dict(objective='clean_mse'), dict(population=1), dict(iterations=1.5),
                    dict(overlap_s=3), dict(omega_max_hz=100), dict(seed=-1), dict(vme_tau=-1)]:
            with self.assertRaises(ValueError):
                ovme_reg_clean(np.arange(100.), 200, cfg)

    def test_input_not_mutated_and_no_hidden_reference(self):
        x = np.random.default_rng(89).normal(size=601)
        before = x.copy()
        cfg = OVMEConfig(population=3, iterations=2)
        a, b = ovme_reg_clean(x, 200, cfg), ovme_reg_clean(x, 200, cfg)
        np.testing.assert_array_equal(x, before)
        np.testing.assert_array_equal(a.cleaned, b.cleaned)
        self.assertEqual(a.cleaned.shape, x.shape)
        np.testing.assert_allclose(a.cleaned+a.estimated_artifact, x, atol=1e-14)
        self.assertFalse(a.diagnostics['uses_clean_reference'])
        self.assertEqual(a.diagnostics['auxiliary_channel_count'], 0)

    def test_constant_signal_unchanged(self):
        for length in [2, 600, 601, 2000]:
            x = np.full(length, 5.)
            out = ovme_reg_clean(x, 200)
            np.testing.assert_array_equal(out.cleaned, x)

    def test_block_coverage(self):
        def fake(x, fs, cfg, seed):
            return np.ones_like(x), dict(peak_count=1, intervals=[[0, len(x)]])
        with patch('blink_repro.methods.ovme_reg._clean_block', side_effect=fake):
            x = np.arange(2001.)
            result = ovme_reg_clean(x, 200)
            np.testing.assert_array_equal(result.cleaned, x-1)
            self.assertEqual(result.diagnostics['blocks'][0]['start'], 0)
            self.assertEqual(result.diagnostics['blocks'][-1]['end'], len(x))


if __name__ == '__main__':
    unittest.main()
