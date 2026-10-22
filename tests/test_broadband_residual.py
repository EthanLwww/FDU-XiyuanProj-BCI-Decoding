"""Broadband residual regression: data routing, fusion, compatibility and training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from bci import filter_bank as signals
from bci import FilterBankModels_tf as models
from bci import training_cli as cli
from bci.training_artifacts import validate_checkpoint


class BroadbandResidualTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(23)

    def test_appended_broadband_matches_original_and_preserves_subbands(self):
        raw = np.random.default_rng(2).normal(size=(3, 4, 300))
        four = signals.preprocess_windows(raw, mode="weighted")
        five = signals.preprocess_windows(raw, mode="weighted", broadband_residual="gated")
        reference = signals.preprocess_windows(raw, mode="off")
        self.assertEqual(five.shape, (3, 4, 300, 5))
        self.assertEqual(five.dtype, np.float32)
        np.testing.assert_array_equal(five[..., :4], four)
        np.testing.assert_allclose(five[..., 4:], reference, atol=2e-7)
        np.testing.assert_array_equal(
            five, signals.preprocess_windows(raw, mode="weighted", broadband_residual="fixed", chunk_size=1))
        zeros = signals.preprocess_windows(np.zeros((2, 4, 300)), mode="mean", broadband_residual="gated")
        self.assertTrue(np.isfinite(zeros).all())

    def test_invalid_residual_combinations_are_rejected(self):
        raw = np.zeros((1, 4, 300))
        for fb, residual in (("off", "gated"), ("off", "fixed"), ("mean", "typo")):
            with self.subTest(fb=fb, residual=residual):
                with self.assertRaises(ValueError):
                    signals.preprocess_windows(raw, mode=fb, broadband_residual=residual)
                with self.assertRaises(ValueError):
                    models.build_model("EEGNet", 2, 4, 300, mode=fb, broadband_residual=residual)

    def test_residual_equation_and_initial_gate(self):
        self.assertTrue(hasattr(models, "BroadbandResidual"), "Residual layer is not implemented")
        x, y = tf.constant([[1., 2.]]), tf.constant([[3., 5.]])
        gated = models.BroadbandResidual(mode="gated")
        np.testing.assert_allclose(gated([x, y]).numpy(), [[1.3, 2.5]], atol=1e-6)
        self.assertAlmostEqual(float(gated.coefficient()), .1, places=6)
        fixed = models.BroadbandResidual(mode="fixed")
        np.testing.assert_array_equal(fixed([x, y]).numpy(), [[4., 7.]])
        self.assertEqual(len(fixed.trainable_weights), 0)
        gated.gate_logit.assign(-80.)
        np.testing.assert_array_equal(gated([x, y]).numpy(), x.numpy())

    def test_all_fusion_modes_and_disabled_input_shape(self):
        for arch in ("EEGNet", "EEGConformer"):
            for fb in ("mean", "weighted", "concat"):
                for residual in ("off", "fixed", "gated"):
                    with self.subTest(arch=arch, fb=fb, residual=residual):
                        model = models.build_model(arch, 3, 4, 300, mode=fb, broadband_residual=residual)
                        n_inputs = 4 if residual == "off" else 5
                        self.assertEqual(model.input_shape, (None, 4, 300, n_inputs))
                        raw = np.random.default_rng(5).normal(size=(2, 4, 300, n_inputs)).astype("float32")
                        output = model(raw, training=False).numpy()
                        self.assertTrue(np.isfinite(output).all())
                        np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6)
                        info = models.get_residual_info(model)
                        if residual == "off":
                            self.assertIsNone(info)
                        else:
                            self.assertEqual(info["mode"], residual)
                            self.assertAlmostEqual(info["alpha"], 1. if residual == "fixed" else .1, places=6)

    def test_broadband_is_independent_and_receives_gradients(self):
        for arch in ("EEGNet", "EEGConformer"):
            with self.subTest(arch=arch):
                model = models.build_model(arch, 2, 4, 300, mode="weighted", broadband_residual="gated")
                raw = np.random.default_rng(7).normal(size=(3, 4, 300, 5)).astype("float32")
                altered = raw.copy()
                altered[..., 4] = np.random.default_rng(9).normal(size=(3, 4, 300))
                before_fusion = tf.keras.Model(model.input, [model.get_layer("band_fusion").output,
                                                             model.get_layer("broadband_features").output])
                first = before_fusion(raw, training=False)
                second = before_fusion(altered, training=False)
                np.testing.assert_array_equal(first[0].numpy(), second[0].numpy())
                self.assertGreater(np.max(np.abs(first[1].numpy() - second[1].numpy())), 1e-5)
                gate = model.get_layer("broadband_residual").gate_logit
                kernel = model.get_layer("broadband_temporal").kernel
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                        tf.constant([0, 1, 0]), model(raw, training=True)))
                gradients = tape.gradient(loss, [gate, kernel])
                for gradient in gradients:
                    self.assertIsNotNone(gradient)
                    self.assertTrue(np.isfinite(gradient.numpy()).all())
                    self.assertGreater(np.max(np.abs(gradient.numpy())), 1e-8)

    def test_residual_h5_roundtrip(self):
        for arch in ("EEGNet", "EEGConformer"):
            for residual in ("fixed", "gated"):
                with self.subTest(arch=arch, residual=residual), tempfile.TemporaryDirectory() as directory:
                    model = models.build_model(arch, 2, 4, 300, mode="concat", broadband_residual=residual)
                    if residual == "gated":
                        model.get_layer("broadband_residual").gate_logit.assign(-.7)
                    raw = np.random.default_rng(4).normal(size=(2, 4, 300, 5)).astype("float32")
                    path = Path(directory) / "model.h5"
                    model.save(path)
                    restored = tf.keras.models.load_model(path, compile=False)
                    np.testing.assert_allclose(model(raw, training=False), restored(raw, training=False), atol=1e-6)
                    self.assertAlmostEqual(models.get_residual_info(model)["alpha"],
                                           models.get_residual_info(restored)["alpha"], places=6)

    def test_cli_names_and_old_checkpoint_compatibility(self):
        args = ["1", "2", "2", "ME", "Orig", "EEGNet", "--filter-bank", "weighted", "--seed", "42"]
        old = cli.parse_training_args(args)
        self.assertEqual(cli.experiment_name(old), "S01_Sess02_ME_2class_EEGNet_FB-weighted_Seed42_Orig.h5")
        names = {cli.experiment_name(cli.parse_training_args(args + ["--broadband-residual", mode]))
                 for mode in ("off", "fixed", "gated")}
        self.assertEqual(len(names), 3)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_training_args(["1", "2", "2", "ME", "Orig", "--broadband-residual", "gated"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.h5"
            path.touch()
            old_config = {"filter_bank": "weighted"}
            path.with_suffix(".config.json").write_text(json.dumps(old_config))
            validate_checkpoint(path, dict(old_config, broadband_residual="off"))
            with self.assertRaises(ValueError):
                validate_checkpoint(path, dict(old_config, broadband_residual="gated"))
            path.with_suffix(".config.json").write_text(json.dumps(dict(old_config, broadband_residual="fixed")))
            with self.assertRaises(ValueError):
                validate_checkpoint(path, dict(old_config, broadband_residual="gated"))

    def test_training_and_finetuning_export_best_gate(self):
        from bci.Functions import train_models
        data = np.random.default_rng(25).normal(size=(12, 4, 500))
        labels = np.tile([1, 2], 6)
        params = dict(maxtriallen=5, windowlen=3, block_size=128, downsrate=100, srate=100,
                      bandpass_filt=[4, 40], nclass=2, filter_bank="weighted", broadband_residual="gated",
                      seed=42, epochs=1, batch_size=8, verbose=0)
        for arch in ("EEGNet", "EEGConformer"):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                tf.keras.backend.clear_session()
                original = Path(directory) / "Orig.h5"
                tuned_path = Path(directory) / "Finetune.h5"
                with contextlib.redirect_stdout(io.StringIO()):
                    train_models(data, labels, str(original), dict(params, model_arch=arch))
                config = json.loads(original.with_suffix(".config.json").read_text())
                self.assertEqual(config["broadband_residual"], "gated")
                self.assertEqual(config["input_shape"], [4, 300, 5])
                model = tf.keras.models.load_model(original, compile=False)
                record = json.loads(original.with_suffix(".residual.json").read_text())
                self.assertAlmostEqual(record["alpha"], models.get_residual_info(model)["alpha"], places=6)
                with contextlib.redirect_stdout(io.StringIO()):
                    train_models(data, labels, str(tuned_path),
                                 dict(params, model_arch=arch, modelpath=str(original)))
                tuned = tf.keras.models.load_model(tuned_path, compile=False)
                self.assertTrue(tuned.get_layer("broadband_residual").trainable)
                self.assertFalse(tuned.get_layer("broadband_temporal").trainable)
                self.assertGreater(abs(models.get_residual_info(tuned)["alpha"] -
                                       models.get_residual_info(model)["alpha"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
