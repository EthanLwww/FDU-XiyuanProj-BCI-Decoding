"""Model contracts: independence, trainability, disabled compatibility and persistence."""
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
from bci.EEGModels_tf import EEGNet, EEGConformer


class FilterBankModelTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(17)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("bci.FilterBankModels_tf"),
                             "filter-bank models have not been implemented")
        return importlib.import_module("bci.FilterBankModels_tf")

    def test_disabled_models_preserve_existing_predictions(self):
        fb = self.module()
        for arch in ("EEGNet", "EEGConformer"):
            with self.subTest(arch=arch):
                reference = (EEGNet(3, Chans=4, Samples=300, kernLength=32)
                             if arch == "EEGNet"
                             else EEGConformer(3, Chans=4, Samples=300, depth=3))
                actual = fb.build_model(arch, 3, 4, 300, mode="off")
                self.assertEqual(reference.count_params(), actual.count_params())
                actual.set_weights(reference.get_weights())
                x = np.random.default_rng(0).normal(size=(2, 4, 300, 1)).astype("float32")
                np.testing.assert_array_equal(reference(x, training=False).numpy(),
                                              actual(x, training=False).numpy())

    def test_band_branches_are_independent_before_fusion(self):
        fb = self.module()
        for arch in ("EEGNet", "EEGConformer"):
            with self.subTest(arch=arch):
                model = fb.build_model(arch, 2, 4, 300, mode="weighted")
                branch = tf.keras.Model(model.input, [
                    model.get_layer("band_0_features").output,
                    model.get_layer("band_1_features").output])
                x = np.random.default_rng(5).normal(size=(2, 4, 300, 4)).astype("float32")
                altered = x.copy()
                altered[..., 1] = np.random.default_rng(6).normal(size=(2, 4, 300))
                before, after = branch(x, training=False), branch(altered, training=False)
                np.testing.assert_array_equal(before[0].numpy(), after[0].numpy())
                self.assertGreater(np.max(np.abs(before[1].numpy() - after[1].numpy())), 1e-4)

    def test_all_modes_output_valid_probabilities(self):
        fb = self.module()
        for arch in ("EEGNet", "EEGConformer"):
            for mode in ("mean", "weighted", "concat"):
                with self.subTest(arch=arch, mode=mode):
                    model = fb.build_model(arch, 3, 4, 300, mode=mode)
                    x = np.random.default_rng(7).normal(size=(2, 4, 300, 4)).astype("float32")
                    y = model(x, training=False).numpy()
                    self.assertEqual(y.shape, (2, 3))
                    self.assertTrue(np.isfinite(y).all())
                    np.testing.assert_allclose(y.sum(axis=-1), 1, atol=1e-6)
                    weights = fb.get_band_weights(model)
                    if mode == "concat":
                        self.assertIsNone(weights)
                    else:
                        np.testing.assert_allclose(weights, [0.25] * 4)

    def test_weighted_logits_receive_gradients_and_survive_h5_reload(self):
        fb = self.module()
        for arch in ("EEGNet", "EEGConformer"):
            with self.subTest(arch=arch):
                model = fb.build_model(arch, 2, 4, 300, mode="weighted")
                x = np.random.default_rng(8).normal(size=(3, 4, 300, 4)).astype("float32")
                logits = model.get_layer("band_fusion").logits
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                        tf.constant([0, 1, 0]), model(x, training=True)))
                gradient = tape.gradient(loss, logits)
                self.assertIsNotNone(gradient)
                self.assertTrue(np.isfinite(gradient.numpy()).all())
                self.assertGreater(np.max(np.abs(gradient.numpy())), 1e-7)
                tf.keras.optimizers.Adam(0.01).apply_gradients([(gradient, logits)])
                self.assertGreater(np.max(np.abs(np.asarray(fb.get_band_weights(model)) - .25)), 1e-7)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.h5"
                    model.save(path)
                    restored = tf.keras.models.load_model(path, compile=False)
                    np.testing.assert_allclose(model(x, training=False).numpy(),
                                               restored(x, training=False).numpy(), atol=1e-6)
                    np.testing.assert_allclose(fb.get_band_weights(model),
                                               fb.get_band_weights(restored), atol=1e-7)

    def test_invalid_mode_and_too_short_window_are_rejected(self):
        fb = self.module()
        with self.assertRaises(ValueError):
            fb.build_model("EEGNet", 2, 4, 300, mode="typo")
        with self.assertRaises(ValueError):
            fb.build_model("EEGConformer", 2, 4, 50, mode="weighted")


if __name__ == "__main__":
    unittest.main()
