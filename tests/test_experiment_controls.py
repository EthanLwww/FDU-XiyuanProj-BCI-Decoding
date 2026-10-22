import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import tensorflow as tf
import training_cli
import training_callbacks


class ExperimentControlTests(unittest.TestCase):
    def test_matched_finetuning_keeps_same_dense_weights_plus_plugin_gates(self):
        from Functions import configure_finetuning
        from FilterBankModels_tf import build_model
        expected = [((160, 40), (40,)), ((560, 256), (256,)),
                    ((256, 32), (32,)), ((32, 2), (2,))]
        for mode, residual, extra in [("off", "off", 0), ("weighted", "off", 1),
                                      ("weighted", "gated", 2)]:
            tf.keras.backend.clear_session()
            model = build_model("EEGConformer", 2, 64, 300, mode=mode, depth=1,
                                broadband_residual=residual)
            configure_finetuning(model, "conformer_baseline")
            dense = [layer for layer in model.layers if layer.trainable
                     and isinstance(layer, tf.keras.layers.Dense)]
            self.assertEqual([tuple(tuple(w.shape) for w in layer.weights) for layer in dense], expected)
            self.assertEqual(len(model.trainable_weights), 8 + extra)

    def test_cli_depth_and_window_are_recorded_without_changing_defaults(self):
        args = training_cli.parse_training_args(["1", "2", "2", "ME", "Orig", "EEGConformer",
                                                "--depth", "1", "--windowlen", "3"])
        self.assertEqual(args.depth, 1)
        self.assertEqual(args.windowlen, 3)
        self.assertIn("Depth1", training_cli.experiment_name(args))
        default = training_cli.parse_training_args(["1","2","2","ME","Orig"])
        self.assertEqual(training_cli.experiment_name(default), "S01_Sess02_ME_2class_EEGNet_Orig.h5")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            training_cli.parse_training_args(["1","2","2","ME","Orig","--windowlen","6"])

    def test_history_is_saved_each_epoch(self):
        self.assertTrue(hasattr(training_callbacks, "EpochHistory"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.h5"
            callback = training_callbacks.EpochHistory(checkpoint)
            callback.on_train_begin()
            callback.on_epoch_end(0, {"loss": 1.2, "accuracy": .4, "val_loss": 1.1, "val_accuracy": .5})
            callback.on_epoch_end(1, {"loss": .9, "accuracy": .7, "val_loss": .8, "val_accuracy": .6})
            history = json.loads(checkpoint.with_suffix(".history.json").read_text())
            self.assertEqual(history["val_accuracy"], [.5, .6])
            progress = json.loads(checkpoint.with_suffix(".progress.json").read_text())
            self.assertEqual(progress["epoch"], 2)
            self.assertGreaterEqual(progress["elapsed_seconds"], 0)

    def test_depth_one_reaches_saved_model_and_config(self):
        from Functions import train_models
        data = np.random.default_rng(1).normal(size=(8, 4, 500))
        labels = np.tile([1,2], 4)
        params = dict(maxtriallen=5,windowlen=3,block_size=128,downsrate=100,srate=100,
                      bandpass_filt=[4,40],nclass=2,model_arch="EEGConformer",depth=1,
                      filter_bank="weighted",broadband_residual="gated",seed=42,
                      epochs=1,batch_size=8,verbose=0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "depth1.h5"
            with contextlib.redirect_stdout(io.StringIO()):
                train_models(data,labels,str(checkpoint),params)
            config = json.loads(checkpoint.with_suffix(".config.json").read_text())
            self.assertEqual(config["depth"],1)
            restored = tf.keras.models.load_model(checkpoint,compile=False)
            self.assertEqual(sum(isinstance(layer,tf.keras.layers.MultiHeadAttention)
                                 for layer in restored.layers),1)


if __name__ == "__main__":
    unittest.main()
