"""Cache deterministic per-window transforms without changing trial split order."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample
from bci.filter_bank import preprocess_windows


def cache_signature(data, labels, params):
    signature = {key: params[key] for key in
                 ("srate", "downsrate", "windowlen", "block_size", "bandpass_filt", "nclass")}
    signature.update(shape=list(data.shape),
                     labels_sha256=hashlib.sha256(np.asarray(labels).tobytes()).hexdigest())
    filename = getattr(data, "filename", None)
    if filename:
        path = Path(filename)
        signature["raw_cache"] = dict(path=str(path.resolve()), size=path.stat().st_size,
                                      mtime_ns=path.stat().st_mtime_ns)
    else:
        signature["data_dtype"] = str(data.dtype)
        signature["data_sha256"] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    return signature


def build_window_cache(data, labels, params, directory):
    from bci.training_artifacts import write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(data, labels, params)
    ready = directory / "ready.json"
    if ready.exists():
        if json.loads(ready.read_text())["signature"] != signature:
            raise ValueError("Cached preprocessing configuration differs.")
        return
    length = int(params["windowlen"] * params["srate"])
    samples = int(params["windowlen"] * params["downsrate"])
    starts, trials = [], []
    # Mirror segment_data's start-major, then trial-major ordering.
    for start in range(0, data.shape[2] - length + 1, params["block_size"]):
        for first in range(0, len(data), 64):
            chunk = data[first:first + 64, :, start:start + length]
            valid = ~np.isnan(chunk).any(axis=(1, 2))
            indices = np.arange(first, first + len(chunk))[valid]
            trials.extend(indices.tolist())
            starts.extend([start] * len(indices))
    trial_ids, start_ids = np.array(trials), np.array(starts)
    if not len(trial_ids):
        raise ValueError("No complete windows for preprocessing cache.")
    shape = (len(trial_ids), data.shape[1], samples)
    baseline = np.lib.format.open_memmap(directory / "baseline.npy", mode="w+",
                                        dtype=np.float64, shape=(*shape, 1))
    banks = np.lib.format.open_memmap(directory / "filterbank.npy", mode="w+",
                                     dtype=np.float32, shape=(*shape, 4))
    residual = np.lib.format.open_memmap(directory / "filterbank_residual.npy", mode="w+",
                                        dtype=np.float32, shape=(*shape, 5))
    for first in range(0, len(trial_ids), 128):
        stop = min(first + 128, len(trial_ids))
        raw = np.stack([data[trial_ids[i], :, start_ids[i]:start_ids[i] + length]
                        for i in range(first, stop)])
        x = resample(raw, samples, axis=2)
        baseline[first:stop] = preprocess_windows(x, params["downsrate"], "off",
                                                 bandpass=params["bandpass_filt"])
        five = preprocess_windows(x, params["downsrate"], "weighted",
                                  bandpass=params["bandpass_filt"], broadband_residual="gated")
        banks[first:stop] = five[..., :4]
        residual[first:stop] = five
    for array in (baseline, banks, residual):
        array.flush()
    np.savez(directory / "index.npz", trial_ids=trial_ids, starts=start_ids)
    write_json(ready, dict(signature=signature, windows=len(trial_ids)))


def load_cached_split(directory, data, labels, params, indices, mode, residual_mode):
    directory = Path(directory)
    metadata = json.loads((directory / "ready.json").read_text())
    if metadata["signature"] != cache_signature(data, labels, params):
        raise ValueError("Cached preprocessing configuration differs.")
    mapping = np.load(directory / "index.npz")
    trial_ids, starts = mapping["trial_ids"], mapping["starts"]
    ranks = np.full(len(data), -1, dtype=np.int64)
    ranks[indices] = np.arange(len(indices))
    selected = np.flatnonzero(ranks[trial_ids] >= 0)
    selected = selected[np.lexsort((ranks[trial_ids[selected]], starts[selected]))]
    variant = ("baseline" if mode == "off" else
               "filterbank_residual" if residual_mode != "off" else "filterbank")
    source = np.load(directory / (variant + ".npy"), mmap_mode="r")
    return source[selected], labels[trial_ids[selected]], ranks[trial_ids[selected]]
