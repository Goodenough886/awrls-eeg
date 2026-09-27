from pathlib import Path
import sys
import unittest

import numpy as np
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blink_repro.methods.mca import mca_clean
from blink_repro.methods.mca_legacy import mca_clean as legacy_clean


class MCATests(unittest.TestCase):
    def test_dictionary_analysis_atoms_have_equal_norms(self):
        impulse = np.zeros(1024)
        impulse[512] = 1.0
        for ca, cd in pywt.swt(impulse, "db4", level=4, norm=False):
            self.assertAlmostEqual(float(ca @ ca), 1.0, places=12)
            self.assertAlmostEqual(float(cd @ cd), 1.0, places=12)

    def test_blinks_removed_without_erasing_oscillatory_eeg(self):
        for fs in (200, 512):
            for polarity in (-1, 1):
                with self.subTest(fs=fs, polarity=polarity):
                    t = np.arange(10 * fs + 3) / fs
                    eeg = 8*np.sin(2*np.pi*10*t) + 2*np.sin(2*np.pi*20*t)
                    blink = polarity * 90*np.exp(-0.5*((t-4.4)/0.12)**2)
                    x = eeg + blink
                    result = mca_clean(x, fs)
                    gain = 10*np.log10(np.sum(blink**2) / np.sum((result.cleaned-eeg)**2))
                    self.assertGreater(gain, 15)
                    self.assertGreater(np.corrcoef(eeg, result.cleaned)[0, 1], 0.95)
                    np.testing.assert_allclose(result.cleaned + result.estimated_artifact, x, atol=1e-12)

    def test_regression_for_normalized_swt_failure(self):
        t = np.arange(2000) / 200
        eeg = 8*np.sin(2*np.pi*10*t) + 2*np.sin(2*np.pi*20*t)
        x = eeg + 90*np.exp(-0.5*((t-4.4)/0.12)**2)
        old_error = np.sum((legacy_clean(x, 200).cleaned-eeg)**2)
        new_error = np.sum((mca_clean(x, 200).cleaned-eeg)**2)
        self.assertLess(new_error, old_error / 20)

    def test_uncontaminated_rhythms_are_preserved(self):
        t = np.arange(2000) / 200
        eeg = 8*np.sin(2*np.pi*10*t) + 2*np.sin(2*np.pi*20*t)
        cleaned = mca_clean(eeg, 200).cleaned
        self.assertLess(np.linalg.norm(cleaned-eeg)/np.linalg.norm(eeg), 0.05)

    def test_units_and_polarity_do_not_change_result(self):
        t = np.arange(2003) / 200
        x = 8*np.sin(2*np.pi*10*t) + 90*np.exp(-0.5*((t-4.4)/0.12)**2)
        saved = x.copy()
        expected = mca_clean(x, 200).cleaned
        for factor in (1e-6, 1e6, -1):
            np.testing.assert_allclose(mca_clean(x*factor, 200).cleaned/factor, expected, atol=1e-9, rtol=1e-9)
        np.testing.assert_array_equal(x, saved)

    def test_empty_short_and_constant_signals(self):
        for x in (np.array([]), np.array([2.]), np.zeros(20), np.ones(20)*50, np.array([1., -1.])):
            result = mca_clean(x, 200)
            self.assertEqual(result.cleaned.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(result.cleaned)))
            np.testing.assert_allclose(result.cleaned + result.estimated_artifact, x)
            if len(x) < 2 or np.ptp(x) == 0:
                np.testing.assert_array_equal(result.cleaned, x)

    def test_invalid_inputs_fail_explicitly(self):
        for x, fs, cfg in (([np.nan, 1], 200, {}), ([1, 2], 0, {}), ([1, 2], 200, {"lambda": 0}), ([1, 2], 200, {"inner_iterations": 0}), ([1, 2], 200, {"level": 13}), ([1, 2], 200, {"threshold_schedule": "unknown"})):
            with self.assertRaises(ValueError):
                mca_clean(x, fs, cfg)


if __name__ == "__main__":
    unittest.main()
