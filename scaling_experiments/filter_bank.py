"""Signal preprocessing for the optional four-band frontend (no TensorFlow dependency)."""
import numpy as np
from scipy import signal, stats

DEFAULT_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0))
FILTER_BANK_MODES = ("off", "mean", "weighted", "concat")
BROADBAND_RESIDUAL_MODES = ("off", "fixed", "gated")


def validate_residual_mode(filter_bank_mode, broadband_residual):
    if broadband_residual not in BROADBAND_RESIDUAL_MODES:
        raise ValueError("Unknown broadband residual mode: " + str(broadband_residual))
    if broadband_residual != "off" and filter_bank_mode == "off":
        raise ValueError("Broadband residual requires an enabled filter bank.")


def validate_bands(bands, sampling_rate):
    bands = tuple(tuple(map(float, band)) for band in bands)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0 or not bands:
        raise ValueError("Provide nonempty bands and a positive sampling rate.")
    for band in bands:
        if len(band) != 2 or not (0 < band[0] < band[1] < sampling_rate / 2):
            raise ValueError("Every band must satisfy 0 < low < high < Nyquist.")
    return bands


def filter_bank_transform(data, sampling_rate=100, bands=DEFAULT_BANDS,
                          standardize=True, chunk_size=128, include_broadband=False,
                          broadband_band=(4, 40)):
    """Return float32 (N, C, T, B), or B+1 planes when broadband is requested.

    Z-score is computed per window/channel/band. It intentionally removes absolute
    band power. Windows are filtered independently and causally. Chunking bounds
    intermediate arrays, not the final output size.
    """
    data = np.asarray(data)
    if data.ndim != 3 or min(data.shape) <= 0:
        raise ValueError("Expected nonempty EEG windows with shape (N, C, T).")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    bands = validate_bands(bands, sampling_rate)
    filters = [signal.butter(4, band, btype="bandpass", fs=sampling_rate, output="sos")
               for band in bands]
    if include_broadband:
        validate_bands([broadband_band], sampling_rate)
        broadband_filter = signal.butter(4, broadband_band, btype="bandpass", fs=sampling_rate)
    result = np.empty((*data.shape, len(bands) + int(include_broadband)), dtype=np.float32)
    for start in range(0, len(data), chunk_size):
        chunk = data[start:start + chunk_size]
        if not np.isfinite(chunk).all():
            raise ValueError("EEG windows must be finite before filter-bank preprocessing.")
        padded = np.pad(chunk, ((0, 0), (0, 0), (100, 100)))
        for band_index, sos in enumerate(filters):
            filtered = signal.sosfilt(sos, padded, axis=-1)[..., 100:-100]
            if standardize:
                filtered -= filtered.mean(axis=-1, keepdims=True)
                std = filtered.std(axis=-1, keepdims=True)
                filtered = np.divide(filtered, std, out=np.zeros_like(filtered),
                                     where=std > 0)
            result[start:start + len(chunk), ..., band_index] = filtered
        if include_broadband:
            # Preserve the original broadband filter and normalize only once across
            # its full spectrum, independently of the four normalized subbands.
            filtered = signal.lfilter(*broadband_filter, padded, axis=-1)[..., 100:-100]
            if standardize:
                filtered -= filtered.mean(axis=-1, keepdims=True)
                std = filtered.std(axis=-1, keepdims=True)
                filtered = np.divide(filtered, std, out=np.zeros_like(filtered), where=std > 0)
            result[start:start + len(chunk), ..., -1] = filtered
    return result


def preprocess_windows(data, sampling_rate=100, mode="off", bands=DEFAULT_BANDS,
                       bandpass=(4, 40), chunk_size=128, broadband_residual="off"):
    if mode not in FILTER_BANK_MODES:
        raise ValueError("Unknown filter-bank mode: " + str(mode))
    validate_residual_mode(mode, broadband_residual)
    if mode != "off":
        return filter_bank_transform(data, sampling_rate, bands, chunk_size=chunk_size,
                                     include_broadband=broadband_residual != "off",
                                     broadband_band=bandpass)
    # Preserve the existing single-band path numerically for ablation controls.
    padded = np.pad(data, ((0, 0), (0, 0), (100, 100)), "constant", constant_values=0)
    b, a = signal.butter(4, bandpass, btype="bandpass", fs=sampling_rate)
    filtered = signal.lfilter(b, a, padded, axis=-1)[..., 100:-100]
    return stats.zscore(filtered, axis=2, nan_policy="omit")[..., np.newaxis]


def split_trial_indices(n_trials, seed=None):
    """Split trials before overlapping windows are created."""
    if n_trials < 2:
        raise ValueError("At least two trials are needed for training and validation.")
    shuffled = (np.random.permutation(n_trials) if seed is None
                else np.random.default_rng(seed).permutation(n_trials))
    count = max(1, min(int(0.8 * n_trials), n_trials - 1))
    train = shuffled[:count]
    return train, np.setdiff1d(shuffled, train)
