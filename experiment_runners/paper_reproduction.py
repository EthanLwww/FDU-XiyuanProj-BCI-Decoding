"""Run one paper EEGNet condition with reproducible trial CV and fold checkpoints."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from experiment_runners.paper_pipeline import (PIPELINE_VERSION, make_folds, prepare_cache,
                                               save_json, sha256, vote_trials)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    # Set limits before importing TensorFlow. Only the externally assigned GPU is visible.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(False)
    from bci.EEGModels_tf import EEGNet
    if args.folds != 5 and not args.smoke:
        raise ValueError('Formal reproduction requires five folds')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(pipeline_version=PIPELINE_VERSION, subject=args.subject, task=args.task,
                  labels=args.labels, band=args.band, folds=args.folds, epochs=args.epochs,
                  batch_size=16, seed=args.seed, model='upstream EEGNet-8,2',
                  selection='maximum validation window accuracy, first maximum',
                  validation_role='CV held-out fold also used for callbacks; no independent test set',
                  split='StratifiedKFold over eligible trials, shuffle=True',
                  smoke=args.smoke, fold_limit=args.fold_limit,
                  tensorflow=tf.__version__, python=__import__('sys').version,
                  pipeline_sha256=sha256(Path(__file__).with_name('paper_pipeline.py')),
                  runner_sha256=sha256(__file__), model_source_sha256=sha256(Path(__file__).parents[1]/'bci/EEGModels_tf.py'))
    cfg_path = output/'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise ValueError(f'Refusing to mix configurations in {output}')
    save_json(cfg_path, config)
    cache = prepare_cache(args.cache, args.data_root, args.subject, args.task, tuple(args.band))
    metadata = json.loads((cache/'ready.json').read_text())
    index = np.load(cache/'index.npz')
    all_groups, raw_labels = index['trial_ids'], index['trial_labels']
    select = np.flatnonzero(np.isin(index['labels'], args.labels))
    groups = all_groups[select]
    raw_to_encoded = {label: i for i, label in enumerate(args.labels)}
    encoded = np.array([raw_to_encoded.get(int(x), -1) for x in raw_labels])
    folds = make_folds(raw_labels, groups, n_splits=args.folds, seed=args.seed)
    data = np.memmap(cache/'windows.dat', dtype=np.float32, mode='r', shape=tuple(metadata['shape']))
    save_json(output/'data_audit.json', dict(cache=str(cache), n_selected_trials=int(np.isin(raw_labels,args.labels).sum()),
                n_eligible_trials=len(np.unique(groups)), n_windows=len(groups),
                no_window_trials=[t for t in metadata['trials'] if t['label'] in args.labels and not t['n_windows']],
                short_trials=[t for t in metadata['trials'] if t['label'] in args.labels and t['valid_raw_samples']<5120],
                source_files=metadata['files'], exact_stride_seconds=128/1024,
                complete_trial_windows=33, cache_sha256=sha256(cache/'ready.json')))

    def dataset(window_positions, training, fold_seed):
        epoch = [0]
        def generate():
            order = np.asarray(window_positions).copy()
            if training:
                np.random.RandomState(fold_seed+epoch[0]).shuffle(order)
            epoch[0] += 1
            for start in range(0, len(order), 16):
                positions = order[start:start+16]
                global_idx = select[positions]
                yield np.asarray(data[global_idx]), np.eye(len(args.labels), dtype=np.float32)[encoded[groups[positions]]]
        ds = tf.data.Dataset.from_generator(generate, output_signature=(
            tf.TensorSpec((None,128,100,1),tf.float32), tf.TensorSpec((None,len(args.labels)),tf.float32)))
        ds = ds.apply(tf.data.experimental.assert_cardinality((len(window_positions)+15)//16))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        options.threading.max_intra_op_parallelism = 1
        options.experimental_deterministic = True
        return ds.with_options(options).prefetch(1)

    rows = []
    for split in folds[:args.fold_limit]:
        fold = split['fold']
        directory = output/f'fold{fold}'
        directory.mkdir(exist_ok=True)
        result_path = directory/'result.json'
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved['checkpoint_sha256'] != sha256(directory/'best.weights.h5'):
                raise ValueError('Completed checkpoint digest mismatch')
            rows.append(saved)
            continue
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed+fold-1)
        model = EEGNet(nb_classes=len(args.labels), Chans=128, Samples=100,
                       dropoutRate=.5, kernLength=32, F1=8, D=2, F2=16, dropoutType='Dropout')
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=.001),
                      metrics=['accuracy'], jit_compile=False)
        save_json(directory/'split.json', {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in split.items()})
        checkpoint = directory/'best.weights.h5'
        started = time.monotonic()

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                save_json(output/'progress.json', dict(state='training', subject=args.subject, task=args.task,
                    fold=fold, epoch=epoch+1, max_epochs=args.epochs,
                    elapsed_fold_seconds=time.monotonic()-started,
                    metrics={k:float(v) for k,v in (logs or {}).items()}))

        train_ds = dataset(split['train_windows'], True, args.seed+fold-1)
        val_ds = dataset(split['val_windows'], False, args.seed+fold-1)
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=0, callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_accuracy', mode='max',
                                               save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=False),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=30),
            tf.keras.callbacks.CSVLogger(str(directory/'history.csv')), Progress(),
            tf.keras.callbacks.TerminateOnNaN()])
        if not all(np.isfinite(v).all() for v in history.history.values()):
            raise ValueError('Nonfinite training history')
        # Evaluate exactly the weights written by the original val_accuracy checkpoint policy.
        model.load_weights(str(checkpoint))
        evaluated_loss, evaluated_acc = model.evaluate(val_ds, verbose=0)
        probabilities = model.predict(val_ds.map(lambda x,y:x), verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        val_groups = groups[split['val_windows']]
        votes = vote_trials(predictions, val_groups, encoded, args.labels)
        for vote in votes:
            vote['trial_id'] = metadata['trials'][vote['trial_index']]['trial_id']
        best_logged = float(np.max(history.history['val_accuracy']))
        actual_acc = float(np.mean(predictions == encoded[val_groups]))
        if not (np.isclose(actual_acc,best_logged,atol=1e-6) and np.isclose(actual_acc,evaluated_acc,atol=1e-6)):
            raise ValueError(f'Checkpoint validation mismatch: {actual_acc}, {best_logged}, {evaluated_acc}')
        if len(votes) != len(split['val_trials']):
            raise ValueError('A held-out trial lost all windows')
        np.savez_compressed(directory/'predictions.npz', probabilities=probabilities, trial_indices=val_groups,
                            raw_starts=index['starts'][select[split['val_windows']]],
                            true_encoded_labels=encoded[val_groups])
        write_csv(directory/'trial_predictions.csv', votes)
        result = dict(fold=fold, n_train_trials=len(split['train_trials']), n_val_trials=len(votes),
                      n_train_windows=len(split['train_windows']), n_val_windows=len(predictions),
                      trial_accuracy=float(np.mean([v['correct'] for v in votes])),
                      n_correct_trials=sum(v['correct'] for v in votes),
                      window_accuracy=actual_acc, loss=float(evaluated_loss),
                      checkpoint_epoch=int(np.argmax(history.history['val_accuracy']))+1,
                      epochs_ran=len(history.history['loss']), best_val_window_accuracy=best_logged,
                      checkpoint_sha256=sha256(checkpoint), elapsed_seconds=time.monotonic()-started,
                      trial_leakage=False, checkpoint_reloaded_and_verified=True,
                      parameters=model.count_params(), ties=sum(v['tie'] for v in votes))
        save_json(result_path, result)
        rows.append(result)
        print(json.dumps(result), flush=True)
    summary = dict(config=config, n_eligible_trials=len(np.unique(groups)), n_windows=len(groups),
                   completed_folds=len(rows), mean_trial_accuracy=float(np.mean([r['trial_accuracy'] for r in rows])),
                   pooled_trial_accuracy=sum(r['n_correct_trials'] for r in rows)/sum(r['n_val_trials'] for r in rows),
                   mean_window_accuracy=float(np.mean([r['window_accuracy'] for r in rows])),
                   std_trial_accuracy=float(np.std([r['trial_accuracy'] for r in rows])), folds=rows)
    save_json(output/'result.json', summary)
    save_json(output/'progress.json', dict(state='complete', completed_folds=len(rows)))
    write_csv(output/'folds.csv', rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--subject', required=True, type=int)
    parser.add_argument('--task', choices=['ME','MI'], default='ME')
    parser.add_argument('--labels', type=int, nargs='+', default=[1,4])
    parser.add_argument('--band', type=float, nargs=2, default=[4,40])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--fold-limit', type=int, default=5)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.labels != sorted(set(args.labels)) or len(args.labels)<2 or not set(args.labels)<=set(range(1,5)):
        parser.error('labels must be an increasing subset of 1,2,3,4')
    if (args.epochs != 300 or args.fold_limit != 5) and not args.smoke:
        parser.error('Use --smoke for reduced training; formal training is 300 epochs maximum and 5 folds')
    try:
        run(args)
    except Exception as exc:
        save_json(Path(args.out)/'error.json', dict(error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
