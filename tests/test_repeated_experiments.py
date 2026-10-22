import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import numpy as np


class WindowCacheTests(unittest.TestCase):
    def test_cached_training_and_finetuning_preserve_split_and_seed(self):
        from experiment_window_cache import build_window_cache
        from Functions import train_models
        from filter_bank import split_trial_indices
        data = np.random.default_rng(8).normal(size=(8, 4, 500)) * 1e-6
        labels = np.tile([1, 2], 4)
        params = dict(srate=100, downsrate=100, windowlen=3, block_size=128,
                      maxtriallen=5, nclass=2, bandpass_filt=[4, 40], seed=43,
                      depth=1, model_arch="EEGConformer", filter_bank="weighted",
                      broadband_residual="gated", epochs=1, batch_size=8, verbose=0)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            build_window_cache(data, labels, params, directory / "cache")
            params["preprocessed_cache"] = str(directory / "cache")
            for phase in ["pretrain", "finetune"]:
                if phase == "finetune":
                    params.update(modelpath=str(directory / "pretrain.h5"),
                                  finetune_policy="conformer_baseline")
                with contextlib.redirect_stdout(io.StringIO()):
                    train_models(data, labels, str(directory / (phase + ".h5")), params)
                config = json.loads((directory / (phase + ".config.json")).read_text())
                self.assertEqual(config["seed"], 43)
                self.assertEqual(config["status"], "complete")
                split = np.load(directory / (phase + ".split.npz"))
                train, val = split_trial_indices(8, 43)
                np.testing.assert_array_equal(split["train_trial_indices"], train)
                np.testing.assert_array_equal(split["val_trial_indices"], val)

    def test_cache_matches_existing_preprocessing_and_window_order(self):
        self.assertIsNotNone(importlib.util.find_spec("experiment_window_cache"),
                             "A reusable, equivalent preprocessing cache is required")
        from experiment_window_cache import build_window_cache, load_cached_split
        from Functions import segment_data
        from filter_bank import split_trial_indices, preprocess_windows
        from scipy.signal import resample
        data = np.random.default_rng(7).normal(size=(8, 4, 500)) * 1e-6
        data[1, :, 350:] = np.nan
        data[3, :, 290:] = np.nan
        labels = np.tile([1, 2], 4)
        params = dict(srate=100, downsrate=100, windowlen=3, block_size=128,
                      maxtriallen=5, nclass=2, bandpass_filt=[4, 40])
        with tempfile.TemporaryDirectory() as directory:
            build_window_cache(data, labels, params, Path(directory))
            for seed in [42, 43]:
                for indices in split_trial_indices(len(data), seed):
                    x, y, i = segment_data(data[indices], labels[indices], 300, 128)
                    x = resample(x, 300, axis=2)
                    for mode, residual in [("off", "off"), ("weighted", "off"), ("weighted", "gated")]:
                        expected = preprocess_windows(x, 100, mode, broadband_residual=residual)
                        actual, ay, ai = load_cached_split(Path(directory), data, labels, params,
                                                           indices, mode, residual)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(ay, y)
                        np.testing.assert_array_equal(ai, i)
            bad = dict(params, windowlen=2)
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_cached_split(Path(directory), data, labels, bad, np.arange(4), "off", "off")
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_cached_split(Path(directory), data * 2, labels, params, np.arange(4), "off", "off")

    def test_repeat_summary_uses_one_checkpoint_per_seed(self):
        self.assertIsNotNone(importlib.util.find_spec("repeated_experiments"),
                             "Repeated experiment summarization is required")
        from repeated_experiments import summarize_histories
        rows = summarize_histories([
            dict(seed=42, variant="baseline", phase="finetune",
                 history=dict(loss=[1., .6, .5], accuracy=[.5, .6, .7],
                              val_loss=[1.2, .8, .7], val_accuracy=[.5, .8, .8])),
            dict(seed=43, variant="baseline", phase="finetune",
                 history=dict(loss=[.9, .6], accuracy=[.5, .7],
                              val_loss=[.9, .6], val_accuracy=[.7, .6]))])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["best_epoch"] for r in rows], [2, 1])
        self.assertEqual([r["val_loss_at_best_accuracy"] for r in rows], [.8, .9])


if __name__ == "__main__":
    unittest.main()
