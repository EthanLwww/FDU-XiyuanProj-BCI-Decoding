"""File-at-a-time EEG preprocessing and bounded, random-access window batches."""
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
from bci.filter_bank import preprocess_windows, split_trial_indices

DATA_ROOT = Path(os.environ.get('BCI_DATA_ROOT', '/Dataset2/dataset_bci_decode_kxy/data'))
TRANSFORM_VERSION = 'original-car-resample-butter-zscore-f32-v1'


def transform_version(mode='off', residual_mode='off'):
    """Return a cache version that includes the frontend configuration."""
    if mode == 'off' and residual_mode == 'off':
        return TRANSFORM_VERSION
    return f'{TRANSFORM_VERSION}|filter_bank={mode}|broadband_residual={residual_mode}'


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def files_for_subject(sid, data_root=DATA_ROOT):
    p = Path(data_root) / f'S{sid:02d}'
    dirs = ([p / f'OnlineMovement_Sess02_{c}class_Base' for c in (2, 3)]
            if sid == 21 else
            [p / 'OfflineMovement'] + sorted(p.glob('OnlineMovement_Sess01*'))
            + sorted(p.glob('OnlineMovement_Sess02*')))
    if any(not d.is_dir() for d in dirs) or (sid != 21 and len(dirs) != 9):
        raise ValueError(f'Missing ME session directory for S{sid:02d}')
    if any(not list(d.glob('*.mat')) for d in dirs):
        raise ValueError(f'Empty ME session directory for S{sid:02d}')
    return [f for d in dirs for f in sorted(d.glob('*.mat'))]


def pair_trial_events(events):
    """Pair each Target with its following TrialEnd, recording orphan ends."""
    pending, trials, ignored, ordinal = None, [], [], 0
    for kind, sample, value, event_index in events:
        if kind == 'Target':
            if pending is not None:
                raise ValueError('Target encountered before the previous TrialEnd')
            pending = (sample, value, ordinal)
            ordinal += 1
        elif kind == 'TrialEnd':
            if pending is None:
                ignored.append(dict(event_index=event_index, sample_zero_based=sample,
                                    reason='TrialEnd without an open Target'))
                continue
            start, target, trial_index = pending
            trials.append((start, sample, target, trial_index))
            pending = None
    if pending is not None or not trials:
        raise ValueError('Missing TrialEnd or no complete trials')
    return trials, ignored


def decode_mat(path):
    """Match original extraction, retaining a source digest and original trial IDs."""
    path = Path(path)
    before = path.stat()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    mat = loadmat(io.BytesIO(payload))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed while reading: {path}')
    eeg, event = mat['eeg'], mat['event']
    signals = eeg['data'][0][0]
    srate = int(eeg['fsample'][0][0][0][0])
    if srate != 1024 or signals.shape[0] != 128:
        raise ValueError(f'Unexpected sampling rate/channels in {path}: {srate}, {signals.shape}')
    events = []
    for i in range(event.shape[1]):
        evt = event[0, i]
        kind = evt['type'][0]
        sample = int(evt['sample'][0][0]) - 1
        if kind in ('Target', 'TrialEnd'):
            events.append((kind, sample, int(evt['value'][0][0]), i))
    try:
        trials, ignored = pair_trial_events(events)
    except ValueError as error:
        raise ValueError(f'{path}: {error}') from error
    selected, labels, original_ids = [], [], []
    for start, end, target, i in trials:
        if not 0 <= start < end <= signals.shape[1]:
            raise ValueError(f'Invalid trial bounds {path}, trial {i}: {start}, {end}')
        if target not in (1, 4):
            continue
        trial = signals[:, start:end][:, :5120]
        trial = np.pad(trial, ((0, 0), (0, 5120-trial.shape[1])), constant_values=np.nan)
        selected.append(trial)
        labels.append(1 if target == 1 else 2)
        original_ids.append(i)
    if not selected:
        raise ValueError(f'No thumb/pinky trials in {path}')
    x = np.array(selected)
    x = x - x.mean(axis=1, keepdims=True)
    metadata = dict(path=str(path.resolve()), size=before.st_size,
                    mtime_ns=before.st_mtime_ns, sha256=digest,
                    trials_before_label_filter=len(trials), selected_trials=len(labels),
                    original_trial_indices=original_ids, ignored_orphan_trial_ends=ignored)
    return x, np.array(labels, dtype=np.int64), metadata


def transform_trials(data, labels, chunk_size=32, mode='off', residual_mode='off'):
    """Yield transformed windows and their trial/start indices, never all windows."""
    for start in range(0, data.shape[2] - 3072 + 1, 128):
        for first in range(0, len(data), chunk_size):
            raw = data[first:first+chunk_size, :, start:start+3072]
            valid = ~np.isnan(raw).any(axis=(1, 2))
            ids = np.arange(first, first+len(raw))[valid]
            if not len(ids):
                continue
            down = resample(raw[valid], 300, axis=2)
            x = preprocess_windows(down, 100, mode, bandpass=[4, 40],
                                    broadband_residual=residual_mode).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite preprocessed windows')
            yield x, ids, np.full(len(ids), start, dtype=np.int64)


def prepare_subject(cache_root, sid, data_root=DATA_ROOT, mode='off', residual_mode='off'):
    directory = Path(cache_root) / f'S{sid:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    files = files_for_subject(sid, data_root)
    ready = directory / 'ready.json'
    if ready.exists():
        info = json.loads(ready.read_text())
        assert info['transform_version'] == transform_version(mode, residual_mode)
        assert [str(f.resolve()) for f in files] == [f['path'] for f in info['files']]
        for f in info['files']:
            st = Path(f['path']).stat()
            assert (st.st_size, st.st_mtime_ns) == (f['size'], f['mtime_ns'])
        assert (directory/'windows.dat').stat().st_size == int(np.prod(info['shape']))*4
        return info
    start_time = time.monotonic()
    trial_ids, starts, all_labels, trial_files, original_trials, manifests = [], [], [], [], [], []
    offset, count = 0, 0
    pending = directory / 'windows.dat.partial'
    with pending.open('wb') as stream:
        for file_index, path in enumerate(files):
            data, labels, manifest = decode_mat(path)
            for x, ids, positions in transform_trials(data, labels, mode=mode,
                                                      residual_mode=residual_mode):
                x.tofile(stream)
                trial_ids.extend((ids + offset).tolist())
                starts.extend(positions.tolist())
                count += len(x)
            all_labels.extend(labels.tolist())
            trial_files.extend([file_index]*len(labels))
            original_trials.extend(manifest['original_trial_indices'])
            offset += len(labels)
            manifests.append(manifest)
            save_json(directory/'progress.json', dict(subject=sid, files_done=file_index+1,
                files_total=len(files), trials=offset, windows=count,
                elapsed_seconds=time.monotonic()-start_time))
            print(f'S{sid:02d} {file_index+1}/{len(files)}: {offset} trials, {count} windows', flush=True)
            del data, labels
    if not count:
        raise ValueError(f'No finite complete windows in S{sid:02d}')
    ids = np.asarray(trial_ids, dtype=np.int64)
    labels = np.asarray(all_labels, dtype=np.int64)
    np.savez_compressed(directory/'index.npz', trial_ids=ids, starts=np.array(starts),
                        labels=labels[ids], trial_file_indices=trial_files,
                        original_trial_indices=original_trials)
    pending.replace(directory/'windows.dat')
    planes = 1 if mode == 'off' else 4 + int(residual_mode != 'off')
    info = dict(subject=sid, transform_version=transform_version(mode, residual_mode),
                filter_bank=mode, broadband_residual=residual_mode,
                shape=[count, 128, 300, planes],
                n_trials=offset, trial_labels=all_labels, files=manifests,
                class_counts={str(c):int((labels==c).sum()) for c in (1, 2)},
                elapsed_seconds=time.monotonic()-start_time)
    save_json(ready, info)
    return info


class WindowStore:
    """Memory-map subject shards. read() allocates only the requested batch."""
    def __init__(self, directories):
        self.directories = [Path(p) for p in directories]
        self.shards, self.metadata = [], []
        ids, starts, labels, trial_labels = [], [], [], []
        bounds, offset = [0], 0
        for p in self.directories:
            info = json.loads((p/'ready.json').read_text())
            self.metadata.append(info)
            shape = tuple(info['shape'])
            if (p/'windows.dat').stat().st_size != int(np.prod(shape))*4:
                raise ValueError(f'Truncated cache: {p}')
            self.shards.append(np.memmap(p/'windows.dat', dtype=np.float32, mode='r', shape=shape))
            index = np.load(p/'index.npz')
            ids.append(index['trial_ids']+offset)
            starts.append(index['starts'])
            labels.append(index['labels'])
            trial_labels.extend(info['trial_labels'])
            offset += info['n_trials']
            bounds.append(bounds[-1]+shape[0])
        self.bounds = np.asarray(bounds)
        self.trial_ids = np.concatenate(ids)
        self.starts = np.concatenate(starts)
        self.labels = np.concatenate(labels)
        self.trial_labels = np.asarray(trial_labels)
        self.n_trials = offset
        self.shape = (bounds[-1], *self.shards[0].shape[1:])

    def read(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.shape[0]):
            raise IndexError('Window index out of range')
        out = np.empty((len(ids), *self.shape[1:]), dtype=np.float32)
        owners = np.searchsorted(self.bounds[1:], ids, side='right')
        for owner in np.unique(owners):
            mask = owners == owner
            out[mask] = self.shards[owner][ids[mask]-self.bounds[owner]]
        return out

    def select_trials(self, indices):
        ranks = np.full(self.n_trials, -1, dtype=np.int64)
        ranks[indices] = np.arange(len(indices))
        selected = np.flatnonzero(ranks[self.trial_ids] >= 0)
        return selected[np.lexsort((ranks[self.trial_ids[selected]], self.starts[selected]))]

    def split(self, seed=42):
        train, val = split_trial_indices(self.n_trials, seed)
        return self.select_trials(train), self.select_trials(val)


def epoch_batches(indices, batch_size, seed=None):
    order = np.asarray(indices).copy()
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    for first in range(0, len(order), batch_size):
        yield order[first:first+batch_size]


def tf_dataset(store, indices, batch_size=32, shuffle=False, seed=42):
    """Bound prefetch to one batch. An iterator completes exactly one epoch."""
    import tensorflow as tf
    epoch = [0]
    def generate():
        current = epoch[0]
        epoch[0] += 1
        for ids in epoch_batches(indices, batch_size, seed+current if shuffle else None):
            yield store.read(ids), np.eye(2, dtype=np.float32)[store.labels[ids]-1]
    ds = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((None, *store.shape[1:]), tf.float32), tf.TensorSpec((None, 2), tf.float32)))
    ds = ds.apply(tf.data.experimental.assert_cardinality((len(indices)+batch_size-1)//batch_size))
    opts = tf.data.Options()
    opts.threading.private_threadpool_size = 1
    opts.threading.max_intra_op_parallelism = 1
    opts.deterministic = True
    return ds.with_options(opts).prefetch(1)
