"""Signal tests: catch mixed bands, changed legacy preprocessing and split leakage."""
import importlib
import importlib.util
import unittest
import numpy as np
from scipy import signal, stats


class FilterBankTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("bci.filter_bank"),
                             "filter-bank frontend has not been implemented")
        return importlib.import_module("bci.filter_bank")

    def test_sine_waves_land_in_their_own_bands(self):
        fb = self.module()
        t = np.arange(1000) / 100
        x = np.array([np.sin(2 * np.pi * f * t) for f in (6, 10, 20, 35)])[:, None, :]
        y = fb.filter_bank_transform(x, 100, standardize=False)
        self.assertEqual(y.shape, (4, 1, 1000, 4))
        self.assertEqual(y.dtype, np.float32)
        power = np.mean(y[:, 0, 300:, :] ** 2, axis=1)
        np.testing.assert_array_equal(np.argmax(power, axis=1), [0, 1, 2, 3])
        self.assertTrue(np.all(np.diag(power) > 0.35))
        for band in range(4):
            self.assertGreater(power[band, band], 3 * np.max(np.delete(power[band], band)))

    def test_chunking_keeps_values_and_constant_inputs_are_finite(self):
        fb = self.module()
        x = np.random.default_rng(4).normal(size=(5, 3, 300))
        np.testing.assert_array_equal(
            fb.filter_bank_transform(x, 100, chunk_size=1),
            fb.filter_bank_transform(x, 100, chunk_size=128))
        y = fb.filter_bank_transform(np.zeros((2, 3, 300)), 100)
        self.assertTrue(np.isfinite(y).all())
        np.testing.assert_array_equal(y, 0)

    def test_standardization_is_independent_of_eeg_physical_units(self):
        fb = self.module()
        time = np.arange(300) / 100
        signal_uv = np.sin(2 * np.pi * 35 * time)[None, None, :]
        reference = fb.filter_bank_transform(signal_uv, 100)
        in_volts = fb.filter_bank_transform(signal_uv * 1e-7, 100)
        np.testing.assert_allclose(in_volts, reference, atol=1e-5)

    def test_off_matches_previous_preprocessing_numerically(self):
        fb = self.module()
        x = np.random.default_rng(8).normal(size=(3, 4, 300))
        b, a = signal.butter(4, [4, 40], btype="bandpass", fs=100)
        padded = np.pad(x, ((0, 0), (0, 0), (100, 100)))
        reference = stats.zscore(signal.lfilter(b, a, padded, axis=-1)[:, :, 100:-100],
                                 axis=2, nan_policy="omit")
        actual = fb.preprocess_windows(x, 100, mode="off")
        self.assertEqual(actual.shape, (3, 4, 300, 1))
        np.testing.assert_array_equal(actual[..., 0], reference)

    def test_invalid_bands_and_modes_fail_early(self):
        fb = self.module()
        x = np.zeros((1, 2, 300))
        for bands in ([(4, 8), (8, 13), (13, 30), (30, 50)], [(8, 4)], []):
            with self.assertRaises(ValueError):
                fb.filter_bank_transform(x, 100, bands=bands)
        with self.assertRaises(ValueError):
            fb.preprocess_windows(x, 100, mode="typo")
        with self.assertRaises(ValueError):
            fb.filter_bank_transform(x, 100, chunk_size=0)
        with self.assertRaises(ValueError):
            fb.filter_bank_transform(np.full((1, 2, 300), np.nan), 100)

    def test_trial_split_is_repeatable_disjoint_and_seed_sensitive(self):
        fb = self.module()
        train, val = fb.split_trial_indices(30, 42)
        train2, val2 = fb.split_trial_indices(30, 42)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(val, val2)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(sorted(np.r_[train, val]), list(range(30)))
        self.assertFalse(np.array_equal(train, fb.split_trial_indices(30, 43)[0]))
        with self.assertRaises(ValueError):
            fb.split_trial_indices(1, 42)


if __name__ == "__main__":
    unittest.main()
