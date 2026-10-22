"""Checkpoint callback compatible with both Keras 2 and Keras 3 HDF5 saving."""
import math
import json
import time
from pathlib import Path
import tensorflow as tf


class EpochHistory(tf.keras.callbacks.Callback):
    """Persist metrics atomically after each completed epoch."""
    def __init__(self, filepath):
        super().__init__()
        self.filepath = Path(filepath)

    def on_train_begin(self, logs=None):
        self.started = time.monotonic()
        self.history = {}

    def _write(self, suffix, value):
        path = self.filepath.with_suffix(suffix)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)

    def on_epoch_end(self, epoch, logs=None):
        metrics = {key: float(value) for key, value in (logs or {}).items()}
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(value)
        self._write(".history.json", self.history)
        self._write(".progress.json", dict(epoch=epoch + 1,
                    elapsed_seconds=time.monotonic() - self.started, metrics=metrics))


class BestAccuracyCheckpoint(tf.keras.callbacks.Callback):
    """Keep the first checkpoint attaining the maximum validation accuracy.

    Some Keras 3 releases restrict ModelCheckpoint to .keras, although model.save
    still supports .h5. Explicit saving retains compatibility with this project's
    existing load_weights and checkpoint naming conventions.
    """
    def __init__(self, filepath, verbose=0):
        super().__init__()
        self.filepath = str(Path(filepath))
        self.verbose = verbose
        self.best = -math.inf

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_accuracy")
        if value is None or not math.isfinite(float(value)):
            return
        if float(value) > self.best:
            self.model.save(self.filepath)
            self.best = float(value)
            if self.verbose:
                print(f"Epoch {epoch + 1}: saved best val_accuracy={self.best:.6f} to {self.filepath}")
