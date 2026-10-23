"""Paper-compatible raw-trial windowing; no TensorFlow or model fitting here."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter, resample
from scipy.stats import zscore

PIPELINE_VERSION = 'paper-raw1024-window1024-stride128-resample100-lfilter-v2'


def save_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(payload, encoding='utf-8')
    pending.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def iter_windows(data, fs=1024, band=(4, 40), chunk_size=16):
    """Match released Functions.py: segment, drop NaN, resample, lfilter, zscore."""
    if fs != 1024:
        raise ValueError(f'Expected 1024 Hz, got {fs}')
    if data.ndim != 3 or data.shape[-1] != 5120:
        raise ValueError('Expected trials x channels x 5120 (5 s including NaN padding)')
    b, a = butter(4, band, btype='bandpass', fs=100)
    for start in range(0, 5120 - 1024 + 1, 128):
        for first in range(0, len(data), chunk_size):
            raw = data[first:first+chunk_size, :, start:start+1024]
            valid = np.isfinite(raw).all(axis=(1, 2))
            trial_ids = np.arange(first, first+len(raw))[valid]
            if not len(trial_ids):
                continue
            down = resample(raw[valid], 100, axis=2)
            padded = np.pad(down, ((0, 0), (0, 0), (100, 100)))
            filtered = lfilter(b, a, padded, axis=-1)[:, :, 100:-100]
            with np.errstate(invalid='ignore', divide='ignore'):
                output = zscore(filtered, axis=2, nan_policy='omit').astype(np.float32)
            if not np.isfinite(output).all():
                raise ValueError(f'Nonfinite transformed window at raw start={start}, trials={trial_ids}')
            yield output[..., None], trial_ids, np.full(len(trial_ids), start, dtype=np.int64)


def preprocess_trials(data, fs=1024, band=(4, 40)):
    """Small-array convenience for parity tests; training uses the streaming cache."""
    chunks = list(iter_windows(data, fs=fs, band=band))
    if not chunks:
        return (np.empty((0, data.shape[1], 100, 1), np.float32),
                np.empty(0, np.int64), np.empty(0, np.int64))
    return tuple(np.concatenate([chunk[i] for chunk in chunks]) for i in range(3))


def decode_file(path):
    """Load one run, pair events in order, and retain source IDs including short trials."""
    from scaling_experiments.streaming_data import pair_trial_events
    path = Path(path)
    before = path.stat()
    payload = path.read_bytes()
    mat = loadmat(io.BytesIO(payload))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed while loading {path}')
    eeg, event = mat['eeg'], mat['event']
    signals = eeg['data'][0, 0]
    fs = int(np.asarray(eeg['fsample'][0, 0]).item())
    if fs != 1024 or signals.shape[0] != 128:
        raise ValueError(f'Unexpected rate/channels in {path}: {fs}, {signals.shape}')
    events = []
    for i in range(event.shape[1]):
        evt = event[0, i]
        kind = str(evt['type'][0])
        if kind in ('Target', 'TrialEnd'):
            events.append((kind, int(evt['sample'][0, 0])-1, int(evt['value'][0, 0]), i))
    pairs, ignored = pair_trial_events(events)
    data, trials = [], []
    for start, end, label, ordinal in pairs:
        if not 0 <= start < end <= signals.shape[1] or label not in (1, 2, 3, 4):
            raise ValueError(f'Invalid trial in {path}: {start}, {end}, label={label}')
        raw = signals[:, start:min(end, start+5120)].astype(np.float64)
        valid_length = raw.shape[-1]
        raw = raw - raw.mean(axis=0, keepdims=True)
        data.append(np.pad(raw, ((0, 0), (0, 5120-valid_length)), constant_values=np.nan))
        trials.append(dict(trial_id=f'{path.name}:{ordinal}', label=label,
                           valid_raw_samples=valid_length, source_start=start, source_end=end))
    metadata = dict(path=str(path.resolve()), size=before.st_size, mtime_ns=before.st_mtime_ns,
                    sha256=hashlib.sha256(payload).hexdigest(), ignored_orphan_ends=ignored)
    return np.asarray(data), trials, metadata


def prepare_cache(cache_root, data_root, subject, task, band=(4, 40)):
    """Write one subject/band shard a run at a time. Reuse only matching manifests."""
    directory = Path(cache_root) / f'{task}_S{subject:02d}_{band[0]:g}-{band[1]:g}'
    directory.mkdir(parents=True, exist_ok=True)
    folder = 'OfflineMovement' if task == 'ME' else 'OfflineImagery'
    files = sorted((Path(data_root)/f'S{subject:02d}'/folder).glob('*.mat'))
    if not files:
        raise FileNotFoundError(f'No {folder} files for S{subject:02d}')
    ready = directory/'ready.json'
    if ready.exists():
        meta = json.loads(ready.read_text(encoding='utf-8'))
        if meta['pipeline_version'] != PIPELINE_VERSION or meta['band'] != list(band):
            raise ValueError(f'Cache configuration mismatch: {directory}')
        if [str(f.resolve()) for f in files] != [f['path'] for f in meta['files']]:
            raise ValueError('Source file list changed')
        for info in meta['files']:
            st = Path(info['path']).stat()
            if (st.st_size, st.st_mtime_ns) != (info['size'], info['mtime_ns']):
                raise ValueError(f'Source changed: {info["path"]}')
        if (directory/'windows.dat').stat().st_size != int(np.prod(meta['shape']))*4:
            raise ValueError('Cache truncated')
        return directory
    groups, positions, trial_meta, manifests = [], [], [], []
    count = 0
    pending = directory/'windows.dat.partial'
    with pending.open('wb') as stream:
        for i, path in enumerate(files):
            data, trials, manifest = decode_file(path)
            offset = len(trial_meta)
            for windows, ids, starts in iter_windows(data, band=band):
                windows.tofile(stream)
                groups.extend((ids+offset).tolist())
                positions.extend(starts.tolist())
                count += len(windows)
            trial_meta.extend(trials)
            manifests.append(manifest)
            save_json(directory/'progress.json', dict(files_done=i+1, files_total=len(files), windows=count))
    if not count:
        raise ValueError('No valid windows')
    labels = np.array([t['label'] for t in trial_meta], dtype=np.int64)
    ids = np.asarray(groups, dtype=np.int64)
    counts = np.bincount(ids, minlength=len(labels))
    for trial, n in zip(trial_meta, counts):
        trial['n_windows'] = int(n)
    np.savez(directory/'index.npz', trial_ids=ids, starts=np.asarray(positions, dtype=np.int64),
             trial_labels=labels, labels=labels[ids])
    pending.replace(directory/'windows.dat')
    save_json(ready, dict(pipeline_version=PIPELINE_VERSION, band=list(band), subject=subject, task=task,
                         shape=[count, 128, 100, 1], files=manifests, trials=trial_meta,
                         no_window_trial_indices=np.flatnonzero(counts == 0).tolist(),
                         short_trial_count=int(sum(t['valid_raw_samples'] < 5120 for t in trial_meta))))
    return directory


def make_folds(trial_labels, groups, n_splits=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    eligible = np.unique(groups)
    labels = trial_labels[eligible]
    if min(np.bincount(labels)[np.unique(labels)]) < n_splits:
        raise ValueError('Insufficient eligible trials per class for stratified CV')
    result = []
    for i, (train, val) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                                   random_state=seed).split(eligible, labels), 1):
        tr, va = eligible[train], eligible[val]
        trw, vaw = np.flatnonzero(np.isin(groups, tr)), np.flatnonzero(np.isin(groups, va))
        if np.intersect1d(groups[trw], groups[vaw]).size:
            raise AssertionError('Trial leakage across folds')
        result.append(dict(fold=i, train_trials=tr, val_trials=va, train_windows=trw, val_windows=vaw))
    return result


def vote_trials(pred, groups, trial_encoded_labels, labels):
    pred, groups = np.asarray(pred), np.asarray(groups)
    if pred.ndim != 1 or pred.shape != groups.shape or not len(pred):
        raise ValueError('Predictions and window group IDs must have equal nonempty shape')
    rows = []
    for g in np.unique(groups):
        votes = np.bincount(pred[groups == g], minlength=len(labels))
        winner = int(np.argmax(votes))
        truth = int(trial_encoded_labels[g])
        rows.append(dict(trial_index=int(g), true_label=int(labels[truth]), pred_label=int(labels[winner]),
                         correct=bool(winner == truth), n_windows=int(votes.sum()),
                         tie=bool(np.count_nonzero(votes == votes.max()) > 1), votes=votes.tolist()))
    return rows
