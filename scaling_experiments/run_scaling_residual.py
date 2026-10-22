"""Single-seed weighted filter-bank plus gated residual scaling study with S21 transfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

from streaming_data import DATA_ROOT, WindowStore, files_for_subject, prepare_subject, save_json

ROOT = Path('/home/kxy/LYC/experiments/conformer_scaling_S21_filterbank_residual_20261021')
BASELINE_ROOT = Path('/home/kxy/LYC/experiments/conformer_scaling_S21_20261021')
GPUS = (0, 3, 4)
FILTER_BANK_MODE = 'weighted'
BROADBAND_RESIDUAL_MODE = 'gated'
GROUPS = {'s3_reference': [1, 2, 5], 's5': list(range(1, 6)),
          's10': list(range(1, 11)), 's20': list(range(1, 21))}


class ExistingWorker:
    """Observe a still-running worker after its supervisor was interrupted."""
    def __init__(self, pid, root, group):
        self.pid, self.root, self.group = pid, root, group
        self.returncode = None

    def poll(self):
        path = Path(f'/proc/{self.pid}/cmdline')
        try:
            cmd = path.read_bytes().split(b'\0')
            alive = b'worker' in cmd and str(self.root).encode() in cmd and self.group.encode() in cmd
        except FileNotFoundError:
            alive = False
        if alive:
            return None
        phases = ['pretrain', 'finetune']
        self.returncode = 0 if all((self.root/self.group/f'{phase}.verification.json').exists() for phase in phases) else 1
        return self.returncode


def env_for(gpu=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TF_CPP_MIN_LOG_LEVEL='2',
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='4',
                TF_NUM_INTEROP_THREADS='2', PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
                LD_LIBRARY_PATH='/home/kxy/LYC/experiments/conformer_depth1_win3_ME2_20261020/runtime/cudnn:'
                + os.environ.get('LD_LIBRARY_PATH', ''))


def train(root, group, phase):
    import numpy as np
    import tensorflow as tf
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=16384)])
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('GPU required for this training run')
    from Functions import configure_finetuning
    from FilterBankModels_tf import build_model, get_band_weights, get_residual_info
    from streaming_data import tf_dataset
    from training_artifacts import checkpoint_hash, validate_checkpoint
    from training_callbacks import BestAccuracyCheckpoint, EpochHistory
    checkpoint = root/group/f'{phase}.h5'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    verification = checkpoint.with_suffix('.verification.json')
    if verification.exists():
        result = json.loads(verification.read_text())
        assert checkpoint_hash(checkpoint) == result['checkpoint_sha256']
        print('Already verified', checkpoint, flush=True)
        return
    fine = phase == 'finetune'
    subjects = [21] if fine else GROUPS[group]
    store = WindowStore([root/'cache'/f'S{s:02d}' for s in subjects])
    baseline_split_path = BASELINE_ROOT/group/f'{phase}.split.npz'
    with np.load(baseline_split_path) as baseline_split:
        np.testing.assert_array_equal(store.trial_labels, baseline_split['trial_labels'])
        train_trials = baseline_split['train_trial_indices'].copy()
        val_trials = baseline_split['val_trial_indices'].copy()
    train_ids, val_ids = store.select_trials(train_trials), store.select_trials(val_trials)
    with np.load(baseline_split_path) as baseline_split:
        for part, ids in [('train', train_ids), ('val', val_ids)]:
            np.testing.assert_array_equal(store.trial_ids[ids], baseline_split[f'{part}_window_trial_indices'])
            if f'{part}_window_indices' in baseline_split:
                np.testing.assert_array_equal(ids, baseline_split[f'{part}_window_indices'])
    assert len(train_ids) and len(val_ids)
    assert not np.intersect1d(store.trial_ids[train_ids], store.trial_ids[val_ids]).size
    for label in (1, 2):
        assert (store.labels[train_ids] == label).any() and (store.labels[val_ids] == label).any()
    tf.keras.utils.set_random_seed(42)
    tf.keras.backend.set_image_data_format('channels_last')
    config = dict(model_arch='EEGConformer', nclass=2, chans=128, samples=300, depth=1,
        filter_bank=FILTER_BANK_MODE, broadband_residual=BROADBAND_RESIDUAL_MODE,
        bands=[[4, 8], [8, 13], [13, 30], [30, 40]], frontend_version=1,
        input_shape=[128, 300, 5],
        baseline_split_source=str(baseline_split_path),
        baseline_split_sha256=hashlib.sha256(baseline_split_path.read_bytes()).hexdigest(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        windowlen=3, downsrate=100, srate=1024, maxtriallen=5, block_size=128,
        bandpass_filt=[4, 40], task='ME', session_num=2, subjects=subjects,
        pretraining_subjects=GROUPS[group], target_subject=21, seed=42, batch_size=32,
        dropout=.65 if fine else .5, epochs_requested=100 if fine else 300,
        learning_rate=.00005 if fine else .0005, gpu_memory_limit_mib=16384,
        input_storage='file shards, numpy.memmap, batch32, prefetch1',
        shuffle='numpy.default_rng(seed+epoch), one complete permutation per epoch',
        dtype='float32 (same model-input precision as original TensorFlow cast)',
        n_trials=store.n_trials, n_train_windows=len(train_ids), n_validation_windows=len(val_ids),
        n_train_trials=len(train_trials), n_validation_trials=len(val_trials),
        class_counts={str(c):int((store.trial_labels==c).sum()) for c in (1, 2)},
        tensorflow_version=tf.__version__, status='running', data_root=str(DATA_ROOT),
        cache_manifests=[str(p/'ready.json') for p in store.directories])
    if fine:
        config['modelpath'] = str(root/group/'pretrain.h5')
        config['finetune_policy'] = 'conformer_baseline'
        validate_checkpoint(config['modelpath'], config)
    model = build_model('EEGConformer', 2, 128, 300, mode=FILTER_BANK_MODE,
                        dropout=config['dropout'], depth=1,
                        broadband_residual=BROADBAND_RESIDUAL_MODE)
    if fine:
        model.load_weights(config['modelpath'])
        configure_finetuning(model, 'conformer_baseline')
    assert model.input_shape == (None, 128, 300, 5)
    assert model.count_params() == 412751
    config.update(parameter_count=model.count_params(),
                  trainable_parameter_count=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
                  trainable_weight_layers=[layer.name for layer in model.layers if layer.trainable_weights])
    model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.1), metrics=['accuracy'])
    ds_train = tf_dataset(store, train_ids, shuffle=True)
    ds_val = tf_dataset(store, val_ids)
    config_path = checkpoint.with_suffix('.config.json')
    np.savez_compressed(checkpoint.with_suffix('.split.npz'), train_trial_indices=train_trials,
        val_trial_indices=val_trials, train_window_indices=train_ids, val_window_indices=val_ids,
        train_window_trial_indices=store.trial_ids[train_ids], val_window_trial_indices=store.trial_ids[val_ids],
        trial_labels=store.trial_labels)
    started = time.monotonic()
    if config_path.exists() and json.loads(config_path.read_text()).get('status') == 'complete':
        config = json.loads(config_path.read_text())
        hist = json.loads(checkpoint.with_suffix('.history.json').read_text())
    else:
        save_json(config_path, config)
        print(json.dumps(config, indent=2), flush=True)
        history = model.fit(ds_train, validation_data=ds_val, epochs=config['epochs_requested'],
            verbose=2, callbacks=[BestAccuracyCheckpoint(checkpoint, verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=30),
                tf.keras.callbacks.TerminateOnNaN(), EpochHistory(checkpoint)])
        hist = {k:[float(v) for v in vals] for k, vals in history.history.items()}
        if not all(np.isfinite(v).all() for v in hist.values()):
            raise RuntimeError('Nonfinite training history')
        config.update(status='complete', epochs_completed=len(hist['loss']),
            elapsed_seconds=time.monotonic()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            best_val_accuracy=max(hist['val_accuracy']), checkpoint_sha256=checkpoint_hash(checkpoint))
        save_json(checkpoint.with_suffix('.history.json'), hist)
        save_json(config_path, config)
    # Reload the saved model itself, then evaluate every validation window.
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.1), metrics=['accuracy'])
    metrics = {k:float(v) for k,v in model.evaluate(ds_val, verbose=0, return_dict=True).items()}
    best = int(np.argmax(hist['val_accuracy']))
    assert abs(metrics['accuracy']-hist['val_accuracy'][best]) < 1e-6, metrics
    assert abs(metrics['loss']-hist['val_loss'][best]) < 1e-4, metrics
    save_json(verification, dict(status='verified', group=group, phase=phase,
        checkpoint_sha256=checkpoint_hash(checkpoint), best_epoch=best+1,
        recomputed_validation=metrics, n_validation_windows=len(val_ids),
        band_weights=get_band_weights(model), broadband_residual=get_residual_info(model),
        baseline_split_sha256=config['baseline_split_sha256']))
    print('VERIFIED', group, phase, metrics, flush=True)


def audit(root):
    root.mkdir(parents=True, exist_ok=True)
    rows=[]
    for sid in range(1, 22):
        files=files_for_subject(sid)
        counts={}
        for f in files:
            counts[f.parent.name]=counts.get(f.parent.name, 0)+1
        rows.append(dict(subject=sid, files=len(files), directory_counts=counts,
                         bytes=sum(f.stat().st_size for f in files)))
    save_json(root/'data_inventory.json', rows)
    print(json.dumps(rows, indent=2))


def prepare_verified(root, sid):
    """Check relocated source contents and window indexing against the baseline."""
    import numpy as np
    info = prepare_subject(root/'cache', sid, mode=FILTER_BANK_MODE,
                           residual_mode=BROADBAND_RESIDUAL_MODE)
    old_dir = BASELINE_ROOT/'cache'/f'S{sid:02d}'
    old = json.loads((old_dir/'ready.json').read_text())
    for key in ['n_trials', 'trial_labels', 'class_counts']:
        assert info[key] == old[key], (sid, key)
    assert info['shape'][:3] == old['shape'][:3] and info['shape'][3] == 5
    assert len(info['files']) == len(old['files'])
    for new_file, old_file in zip(info['files'], old['files']):
        assert Path(new_file['path']).parts[-3:] == Path(old_file['path']).parts[-3:]
        for key in ['size', 'sha256', 'selected_trials', 'original_trial_indices']:
            assert new_file[key] == old_file[key], (sid, new_file['path'], key)
    with np.load(old_dir/'index.npz') as old_index, np.load(root/'cache'/f'S{sid:02d}'/'index.npz') as new_index:
        assert old_index.files == new_index.files
        for key in old_index.files:
            np.testing.assert_array_equal(old_index[key], new_index[key])
    save_json(root/'cache'/f'S{sid:02d}'/'baseline_verified.json',
              dict(status='verified', subject=sid, files=len(info['files']),
                   windows=info['shape'][0], source_sha256_and_window_indices_identical=True))


def suite(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    lock=(root/'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit(root)
    source=root/'source'
    source.mkdir(exist_ok=True)
    # Retain every executed source revision before updating the resumable suite.
    old_hashes = root/'source_sha256.json'
    if old_hashes.exists():
        snapshot = root/'source_revisions'/f'before_resume_{int(time.time())}'
        shutil.copytree(source, snapshot)
        shutil.copy2(old_hashes, snapshot/'source_sha256.json')
    for file in Path(__file__).parent.glob('*.py'):
        if file.resolve() != (source/file.name).resolve():
            shutil.copy2(file, source/file.name)
    save_json(root/'source_sha256.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')})
    pending=[21]+list(range(1,21))
    prep, active, complete, failures = {}, {}, [], {}
    train_pending=list(GROUPS)
    for procdir in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (procdir/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'worker' not in cmd or str(root).encode() not in cmd or b'--group' not in cmd:
            continue
        group = cmd[cmd.index(b'--group')+1].decode()
        if group not in train_pending:
            continue
        env = (procdir/'environ').read_bytes().split(b'\0')
        gpu = int(next(v.split(b'=', 1)[1] for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')))
        assert gpu in GPUS and gpu not in active
        active[gpu] = (group, ExistingWorker(int(procdir.name),root,group), (root/f'{group}.log').open('a'))
        train_pending.remove(group)
        print('Resuming supervision of existing worker',group,procdir.name,flush=True)
    started=time.time()
    try:
        while pending or prep or train_pending or active:
            for sid,(proc, log) in list(prep.items()):
                if proc.poll() is not None:
                    log.close()
                    del prep[sid]
                    if proc.returncode:
                        failures[f'prepare_S{sid:02d}']=proc.returncode
            if failures:
                raise RuntimeError(str(failures))
            while pending and len(prep)<4:
                sid=pending.pop(0)
                log=(root/f'prepare_S{sid:02d}.log').open('a')
                proc=subprocess.Popen([sys.executable,str(source/'run_scaling_residual.py'),'prepare','--root',str(root),'--subject',str(sid)],
                                      stdout=log,stderr=subprocess.STDOUT,env=env_for())
                prep[sid]=(proc,log)
            for gpu,(group,proc,log) in list(active.items()):
                if proc.poll() is not None:
                    log.close()
                    del active[gpu]
                    if proc.returncode:
                        failures[group]=proc.returncode
                    else:
                        complete.append(group)
            if failures:
                raise RuntimeError(str(failures))
            for gpu in GPUS:
                if gpu in active:
                    continue
                for group in list(train_pending):
                    required=GROUPS[group]+[21]
                    if all((root/'cache'/f'S{s:02d}'/'baseline_verified.json').exists() for s in required):
                        log=(root/f'{group}.log').open('a')
                        proc=subprocess.Popen([sys.executable,str(source/'run_scaling_residual.py'),'worker','--root',str(root),'--group',group],
                                              stdout=log,stderr=subprocess.STDOUT,env=env_for(gpu))
                        active[gpu]=(group,proc,log)
                        train_pending.remove(group)
                        break
            save_json(root/'suite.status.json',dict(status='running',started_unix=started,
                preparing=list(prep),pending_subjects=pending,training={str(k):v[0] for k,v in active.items()},
                pending_groups=train_pending,completed=complete,failures=failures))
            time.sleep(10)
        subprocess.run([sys.executable,str(source/'summarize_residual.py'),'--root',str(root)],check=True,env=env_for())
        save_json(root/'suite.status.json',dict(status='complete',groups=complete,elapsed_seconds=time.time()-started))
    except BaseException:
        save_json(root/'suite.status.json',dict(status='failed',traceback=traceback.format_exc(),
            preparing={str(s):p.pid for s,(p,l) in prep.items()},training={str(g):p.pid for g,(n,p,l) in active.items()},
            completed=complete,failures=failures))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','prepare','worker','train','suite'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--subject',type=int)
    p.add_argument('--group',choices=list(GROUPS))
    p.add_argument('--phase',choices=['pretrain','finetune'])
    a=p.parse_args()
    if a.command=='prepare':
        prepare_verified(a.root, a.subject)
    elif a.command=='worker':
        for phase in ['pretrain', 'finetune']:
            subprocess.run([sys.executable,__file__,'train','--root',str(a.root),'--group',a.group,'--phase',phase],check=True)
    elif a.command=='train':
        train(a.root,a.group,a.phase)
    elif a.command=='audit':
        audit(a.root)
    else:
        suite(a.root)
