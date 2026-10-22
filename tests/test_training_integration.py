"""Exercise the real CLI, saved experiment metadata and training/finetuning."""
import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf


class TrainingIntegrationTests(unittest.TestCase):
    def cli(self):
        self.assertIsNotNone(importlib.util.find_spec("training_cli"),
                             "shared filter-bank CLI has not been implemented")
        return importlib.import_module("training_cli")

    def test_legacy_cli_and_experiment_names(self):
        cli = self.cli()
        old = cli.parse_training_args(["1", "2", "2", "ME", "Orig"])
        self.assertEqual(old.filter_bank, "off")
        self.assertIsNone(old.seed)
        self.assertEqual(cli.experiment_name(old), "S01_Sess02_ME_2class_EEGNet_Orig.h5")
        paths = set()
        for mode in ("off", "mean", "weighted", "concat"):
            args = cli.parse_training_args(["1", "2", "2", "ME", "Orig", "EEGConformer",
                                            "--filter-bank", mode, "--seed", "42"])
            paths.add(cli.experiment_name(args))
        self.assertEqual(len(paths), 4)
        a = cli.parse_training_args(["1,2", "2", "2", "ME", "Orig"], multi_subject=True)
        b = cli.parse_training_args(["3,4", "2", "2", "ME", "Orig"], multi_subject=True)
        self.assertNotEqual(cli.experiment_name(a), cli.experiment_name(b))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_training_args(["1", "2", "2", "ME", "Orig", "--filter-bank", "typo"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_training_args(["1", "2", "2", "ME", "Orig", "--init-model", "old.h5"])

    def test_filterbank_rejects_missing_or_incompatible_checkpoint_metadata(self):
        self.cli()
        artifacts = importlib.import_module("training_artifacts")
        expected = {"model_arch": "EEGNet", "nclass": 2, "filter_bank": "weighted",
                    "bands": [[4., 8.], [8., 13.], [13., 30.], [30., 40.]],
                    "chans": 4, "samples": 300, "downsrate": 100,
                    "windowlen": 3, "bandpass_filt": [4, 40], "depth": 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.h5"
            path.touch()
            with self.assertRaises(ValueError):
                artifacts.validate_checkpoint(path, expected)
            other = dict(expected, filter_bank="mean")
            path.with_suffix(".config.json").write_text(json.dumps(other))
            with self.assertRaises(ValueError):
                artifacts.validate_checkpoint(path, expected)
            path.with_suffix(".config.json").write_text(json.dumps(expected))
            artifacts.validate_checkpoint(path, expected)
            wrong_rate = dict(expected, downsrate=200)
            with self.assertRaises(ValueError):
                artifacts.validate_checkpoint(path, wrong_rate)

    def test_one_epoch_training_and_finetuning_save_reusable_artifacts(self):
        self.cli()
        from Functions import train_models
        from FilterBankModels_tf import get_band_weights
        rng = np.random.default_rng(31)
        data = rng.normal(size=(12, 4, 500))
        labels = np.tile([1, 2], 6)
        params = dict(maxtriallen=5, windowlen=3, block_size=128, downsrate=100,
                      srate=100, bandpass_filt=[4, 40], nclass=2, model_arch="EEGNet",
                      filter_bank="weighted", seed=42, epochs=1, batch_size=8,
                      verbose=0)
        with tempfile.TemporaryDirectory() as directory:
            for arch in ("EEGNet", "EEGConformer"):
                with self.subTest(arch=arch):
                    tf.keras.backend.clear_session()
                    orig = Path(directory) / (arch + "_Orig.h5")
                    with contextlib.redirect_stdout(io.StringIO()):
                        train_models(data, labels, str(orig), dict(params, model_arch=arch))
                    config = json.loads(orig.with_suffix(".config.json").read_text())
                    self.assertEqual(config["filter_bank"], "weighted")
                    self.assertEqual(config["input_shape"], [4, 300, 4])
                    self.assertEqual(config["epochs_completed"], 1)
                    self.assertEqual(config["status"], "complete")
                    split = np.load(orig.with_suffix(".split.npz"))
                    self.assertFalse(set(split["train_window_trial_indices"]) &
                                     set(split["val_window_trial_indices"]))
                    model = tf.keras.models.load_model(orig, compile=False)
                    exported = json.loads(orig.with_suffix(".band_weights.json").read_text())
                    np.testing.assert_allclose(exported["weights"], get_band_weights(model))
                    self.assertTrue(orig.with_suffix(".history.json").is_file())

                    fine = Path(directory) / (arch + "_Finetune.h5")
                    with contextlib.redirect_stdout(io.StringIO()):
                        train_models(data, labels, str(fine),
                                     dict(params, model_arch=arch, modelpath=str(orig)))
                    tuned = tf.keras.models.load_model(fine, compile=False)
                    self.assertTrue(tuned.get_layer("band_fusion").trainable)
                    self.assertFalse(tuned.get_layer("band_0_temporal").trainable)
                    self.assertGreater(np.max(np.abs(
                        np.array(get_band_weights(model)) - np.array(get_band_weights(tuned)))), 1e-9)



    def test_real_single_and_multi_subject_commands(self):
        self.cli()
        import os
        import subprocess
        import sys
        from scipy.io import savemat
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "Data"
            save_root = Path(directory) / "Save"
            for subject in (1, 2):
                folder = data_root / f"S{subject:02}" / "OfflineMovement"
                folder.mkdir(parents=True)
                eeg = np.random.default_rng(subject).normal(size=(4, 3000))
                events = np.empty((1, 12), dtype=[("type", object), ("sample", object), ("value", object)])
                for trial in range(6):
                    events[0, 2 * trial] = ("Target", trial * 500 + 1, 1 if trial % 2 == 0 else 4)
                    events[0, 2 * trial + 1] = ("TrialEnd", (trial + 1) * 500 + 1, 0)
                savemat(folder / "run.mat", {"eeg": {"data": eeg, "fsample": 100.}, "event": events})
            outputs = []
            for script, subjects, mode, residual in (
                ("main_model_training.py", "1", "off", "off"),
                ("main_multi_subject_training.py", "1,2", "concat", "off"),
                ("main_model_training.py", "1", "weighted", "gated"),
                ("main_multi_subject_training.py", "1,2", "mean", "fixed"),
            ):
                command = [sys.executable, str(root / script), subjects, "1", "2", "ME", "Orig",
                           "EEGNet", "--filter-bank", mode, "--seed", "42", "--epochs", "1",
                           "--batch-size", "4", "--data-dir", str(data_root), "--save-dir", str(save_root),
                           "--broadband-residual", residual]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=120, env=os.environ)
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
                args = self.cli().parse_training_args(command[2:], multi_subject="multi" in script)
                checkpoint = save_root / self.cli().experiment_name(args)
                self.assertTrue(checkpoint.is_file())
                config = json.loads(checkpoint.with_suffix(".config.json").read_text())
                self.assertEqual(len(config["data_files"]), len(args.subj_ids))
                self.assertEqual(config["filter_bank"], mode)
                self.assertEqual(config["broadband_residual"], residual)
                self.assertEqual(config["input_shape"][-1], 1 if mode == "off" else 4 + int(residual != "off"))
                outputs.append(checkpoint)
            self.assertEqual(len(set(outputs)), 4)

    def test_best_checkpoint_keeps_best_epoch_not_last(self):
        from training_callbacks import BestAccuracyCheckpoint
        model = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1, use_bias=False)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.h5"
            callback = BestAccuracyCheckpoint(path)
            callback.set_model(model)
            model.set_weights([np.array([[2.]], dtype="float32")])
            callback.on_epoch_end(0, {"val_accuracy": .8})
            model.set_weights([np.array([[3.]], dtype="float32")])
            callback.on_epoch_end(1, {"val_accuracy": .7})
            saved = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_array_equal(saved(np.ones((1, 1))).numpy(), [[2.]])

if __name__ == "__main__":
    unittest.main()
