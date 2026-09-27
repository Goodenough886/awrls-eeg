"""Synthetic tests for the public target-only entry point."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from run_paper import clean_one, METHODS, simulated_metrics, real_metrics
from blink_repro.strict_single_channel import awrls_settings, read_target, preprocess_target


class PublicContract(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / 'configs/paper.json').read_text(encoding='utf-8'))
        self.fs = 200.0
        t = np.arange(2000) / self.fs
        self.truth = 8 * np.sin(2 * np.pi * 10 * t)
        self.x = self.truth + 90 * np.exp(-.5 * ((t - 4.4) / .12) ** 2)

    def test_all_entry_points_reject_matrices_and_nonfinite_values(self):
        for method in METHODS:
            for x in [self.x[:, None], self.x[None, :], np.array([np.nan, 1]), np.array([])]:
                with self.subTest(method=method), self.assertRaises(ValueError):
                    clean_one(method, x, self.fs, self.cfg)

    def test_awrls_matches_frozen_core(self):
        core, settings = awrls_settings(self.fs, self.cfg['awrls'])
        expected, _ = core.process_epoch(self.x, settings)
        original = self.x.copy()
        actual = clean_one('AWRLS-strict', self.x, self.fs, self.cfg)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.x, original)
        self.assertEqual(len(settings.wavelet_candidates), 22)

    def test_kaya_other_channels_cannot_change_preprocessing_or_output(self):
        settings = self.cfg['datasets']['Kaya2018']
        outputs = []
        with tempfile.TemporaryDirectory() as temp:
            for index, (data, names) in enumerate([
                (np.column_stack([self.x, np.zeros_like(self.x)]), ['Fp1', 'Fp2']),
                (np.column_stack([np.full_like(self.x, np.nan), self.x]), ['Fz', 'Fp1']),
                (self.x[:, None], ['Fp1']),
            ]):
                path = Path(temp) / f'case{index}.mat'
                savemat(path, {'o': dict(data=data, chnames=np.array(names, dtype=object), sampFreq=self.fs)})
                raw, fs, _ = read_target(path, 'Kaya2018', 'Fp1')
                x, fs, audit = preprocess_target(raw, fs, 'Fp1', 'Kaya2018',
                                                 settings['preprocessing'], settings['target_fs'])
                self.assertEqual(audit['numeric_input_channels'], 1)
                self.assertEqual(audit['reference_channel_count'], 0)
                outputs.append(clean_one('AWRLS-strict', x, fs, self.cfg, real=True))
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])

    def test_cross_channel_configuration_rejected(self):
        for extra in [{'reference_mode': 'common_average'}, {'reference_channels': ['Fp2']}]:
            cfg = {**self.cfg['datasets']['Kaya2018']['preprocessing'], **extra}
            with self.assertRaises(ValueError):
                preprocess_target(self.x, self.fs, 'Fp1', 'Kaya2018', cfg)

    def test_deprecated_dataset_is_not_distributed(self):
        with self.assertRaises(ValueError):
            read_target(Path('does_not_exist.mat'), 'Cho2017', 'Fp1')

    def test_paper_metrics_include_spectral_distortion(self):
        scores = simulated_metrics(self.x, self.truth, self.truth, self.fs, self.cfg)
        self.assertEqual(scores['mae_after'], 0)
        self.assertAlmostEqual(scores['dbp_total'], 0)
        self.assertAlmostEqual(scores['pearson_after'], 1)

    def test_real_identity_energy_proxy_is_zero(self):
        scores = real_metrics(self.x, self.x, self.fs, self.cfg)
        self.assertAlmostEqual(scores['candidate_window_low_frequency_energy_reduction_percent'], 0)
        self.assertAlmostEqual(scores['outside_window_pearson_r'], 1)


if __name__ == '__main__':
    unittest.main()
