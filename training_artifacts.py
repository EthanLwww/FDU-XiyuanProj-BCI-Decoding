"""Experiment metadata and compatibility checks, kept independent of TensorFlow."""
import hashlib
import json
from pathlib import Path
import numpy as np


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError("Unsupported JSON value: " + type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n",
                          encoding="utf-8")


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path, config):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(path))
    metadata = path.with_suffix(".config.json")
    if not metadata.exists():
        if config["filter_bank"] != "off":
            raise ValueError("Filter-bank finetuning requires the checkpoint's .config.json.")
        return  # Legacy single-band models are additionally checked by load_weights.
    previous = json.loads(metadata.read_text(encoding="utf-8"))
    keys = ("model_arch", "nclass", "filter_bank", "bands", "chans", "samples",
            "downsrate", "windowlen", "bandpass_filt", "depth", "frontend_version")
    mismatched = [key for key in keys if previous.get(key) != config.get(key)]
    # Checkpoints created before the residual plugin have an implicit off mode.
    if previous.get("broadband_residual", "off") != config.get("broadband_residual", "off"):
        mismatched.append("broadband_residual")
    if mismatched:
        raise ValueError("Incompatible checkpoint configuration: " + ", ".join(mismatched))
    if previous.get("checkpoint_sha256") and previous["checkpoint_sha256"] != checkpoint_hash(path):
        raise ValueError("Checkpoint SHA256 does not match its configuration sidecar.")
