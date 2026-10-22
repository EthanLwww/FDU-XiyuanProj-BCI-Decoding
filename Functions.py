import numpy as np
import tensorflow as tf

import glob

import scipy
import scipy.io
from scipy.signal import resample
import matplotlib.pyplot as plt

import os

# EEGNet-specific imports
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K


###### data segmenting and relabeling functions ######
def segment_data(data, labels, segment_size, step_size):
    if segment_size <= 0 or step_size <= 0:
        raise ValueError("segment_size and step_size must be positive.")

    num_trials, num_channels, num_samples = data.shape
    segments = []

    for start in range(0, num_samples - segment_size + 1, step_size):
        end = start + segment_size
        segments.append(data[:, :, start:end])

    segmented_data = np.concatenate(segments, axis=0)
    # repeat labels 
    repeated_labels = np.tile(labels, len(segments))
    trial_indices = range(num_trials)
    repeated_indices = np.tile(trial_indices, len(segments))

    repeated_labels = repeated_labels[~np.isnan(segmented_data).any(axis=(1,2))]
    repeated_indices = repeated_indices[~np.isnan(segmented_data).any(axis=(1,2))]
    segmented_data = segmented_data[~np.isnan(segmented_data).any(axis=(1,2)),:,:]

    return segmented_data, repeated_labels, repeated_indices


def filter_and_relabel(data, label, keep_labels, new_labels):
    filtered_label = label[np.isin(label,keep_labels)]
    filtered_data = data[np.isin(label,keep_labels)]
    filtered_label = np.array([new_labels[l] for l in filtered_label])
    return filtered_data, filtered_label

def generate_paths(subj_id, task, nclass, session_num, model_type, data_folder):
    # get the file paths to the training data
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')

    if task == 'MI':
        prefix = '*Imagery'
    else:
        prefix = '*Movement'
    
    if model_type == 'Finetune':
        prefix_online = f'{prefix}_Sess{session_num:02}'
        if nclass == 3:
            suffix = f'{nclass}class_Base' # 3-class model is fine-tuned on 3-class same day data
        else:
            suffix = 'Base' # 2-class model is fine-tuned on both 2-class and 3-class same day data
    
        pattern = os.path.join(subject_folder, f'{prefix_online}*{suffix}')
        data_paths = sorted(glob.glob(pattern))
    else:
        # load the offline session data
        offline_pattern = os.path.join(subject_folder, prefix) 
        data_paths = sorted(glob.glob(offline_pattern))

        # load the prior online sessions 
        for session in range(1,session_num+1):
            prefix_online = f'{prefix}_Sess{session:02}'
            online_pattern = os.path.join(subject_folder, f'{prefix_online}*')
            data_paths.extend(sorted(glob.glob(online_pattern)))
    return data_paths


def load_and_filter_data(data_paths, params):
    params['data_files'] = []
    label = [] #nTrials
    data = []#nTrials, nChannels, nSamples

    if params['nclass'] == 2:
        keep_labels = [1,4] # thumb; pinky
        new_labels = {1: 1, 4: 2}
    elif params['nclass'] == 3:
        keep_labels = [1,2,4] # thumb; index; pinky
        new_labels = {1: 1, 2: 2, 4: 3}
    else:
        raise ValueError("nclass must be either 2 or 3.")

    for filepath in data_paths:
        if not os.path.isdir(filepath):
            print(f"Skipping non-directory path: {filepath}")
            continue
            
        files = [f for f in os.listdir(filepath) if f.endswith('.mat')]
        for filename in sorted(files):
            cur_data = []

            file_path = os.path.join(filepath, filename)
            print(f"Processing file: {file_path}")

            try:
                mat = scipy.io.loadmat(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
            eeg = mat['eeg']
            event = mat['event']

            signals = eeg['data'][0][0]
            file_srate = eeg['fsample'][0][0][0][0]
            if params['data_files'] and file_srate != params['srate']:
                raise ValueError('All files must have the same sampling rate.')
            params['srate'] = file_srate
            start_idx, end_idx, target = [], [], []

            # Iterate through events
            for i in range(event.shape[1]):
                evt = event[0, i]
                event_type = evt['type'][0]
                sample = evt['sample'][0][0]
                value = evt['value'][0][0]

                if event_type == 'Target':
                    start_idx.append(sample-1) # 0-index
                    target.append(value)
                elif event_type == 'TrialEnd':
                    end_idx.append(sample-1) # 0-index

            cur_label = target

            for i in range(len(start_idx)):
                tmp = signals[:,int(start_idx[i]):int(end_idx[i])]
                tmp = tmp[:,:min(np.size(tmp,1), int(params['maxtriallen']*params['srate']))]
                tmp = np.pad(tmp,((0,0),(0,int(params['maxtriallen']*params['srate'])-np.size(tmp,1))), 'constant', constant_values=np.nan)
        
                cur_data.append(tmp)
            cur_data = np.array(cur_data)

            # CAR
            cur_data = cur_data-cur_data.mean(axis=1, keepdims=True)
            
            data.append(cur_data)
            label.append(cur_label) #nTrials
            stat = os.stat(file_path)
            params['data_files'].append(dict(path=os.path.abspath(file_path),
                                             size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                             trials_before_label_filter=len(cur_label)))

    #### Preprocessing ####
    if not data:
        raise ValueError(f"No data loaded from paths: {data_paths}. Please check if the directories contain .mat files with valid EEG data and trials.")

    data = np.concatenate(data,axis=0)
    label = np.concatenate(label,axis=0)
    label = label.flatten()
    print(data.shape)
    print(label.shape)

    # relabel the data
    data, label = filter_and_relabel(data, label, keep_labels, new_labels)
    return data, label, params

def configure_finetuning(model, policy="legacy_last12"):
    """Match the original Conformer trainable weights across frontend variants."""
    if policy == "legacy_last12":
        for layer in model.layers[:-12]:
            layer.trainable = False
    elif policy == "conformer_baseline":
        dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        if len(dense) < 5 or [layer.units for layer in dense[-4:-1]] != [40, 256, 32]:
            raise ValueError("conformer_baseline requires the expected Conformer head.")
        for layer in model.layers:
            layer.trainable = layer in dense[-4:]
    else:
        raise ValueError("Unknown finetune policy: " + policy)
    for layer in model.layers:
        if layer.name in ("band_fusion", "broadband_residual"):
            layer.trainable = True


def train_models(data, label, save_name, params):
    """Train with trial-level splitting and an optional independent-band frontend."""
    from pathlib import Path
    from filter_bank import DEFAULT_BANDS, preprocess_windows, split_trial_indices, validate_residual_mode
    from FilterBankModels_tf import build_model, get_band_weights, get_residual_info
    from training_artifacts import write_json, validate_checkpoint, checkpoint_hash
    from training_callbacks import BestAccuracyCheckpoint, EpochHistory

    params = dict(params)
    mode = params.get('filter_bank', 'off')
    residual_mode = params.get('broadband_residual', 'off')
    validate_residual_mode(mode, residual_mode)
    seed = params.get('seed')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    K.set_image_data_format('channels_last')
    nTrial, nChan = len(data), np.size(data, axis=1)
    train_idx, val_idx = split_trial_indices(nTrial, seed)
    samples = int(params['windowlen'] * params['downsrate'])
    fine_tune = 'modelpath' in params
    epochs = params.get('epochs', 100 if fine_tune else 300)
    batch_size = params.get('batch_size', 32)
    verbose = params.get('verbose', 2)
    dropout = .65 if fine_tune else .5
    arch = params.get('model_arch', 'EEGNet')
    depth = params.get('depth', 3)
    save_path = Path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(params, filter_bank=mode, bands=[list(b) for b in DEFAULT_BANDS] if mode != 'off' else [],
                  model_arch=arch, chans=nChan, samples=samples, depth=depth,
                  frontend_version=1, broadband_residual=residual_mode,
                  input_shape=[nChan, samples, 1 if mode == 'off' else 4 + int(residual_mode != 'off')],
                  epochs_requested=epochs, batch_size=batch_size, dropout=dropout,
                  tensorflow_version=tf.__version__, status='running', n_trials=nTrial)
    if fine_tune:
        validate_checkpoint(params['modelpath'], config)

    segment_size = int(params['windowlen'] * params['srate'])
    step_size = int(params.get('block_size', 128))
    if segment_size > data.shape[2]:
        raise ValueError('Window length exceeds trial length.')
    if params.get('preprocessed_cache'):
        from experiment_window_cache import load_cached_split
        X_train, Y_train, I_train = load_cached_split(
            params['preprocessed_cache'], data, label, params, train_idx, mode, residual_mode)
        X_validate, Y_validate, I_validate = load_cached_split(
            params['preprocessed_cache'], data, label, params, val_idx, mode, residual_mode)
    else:
        X_train, Y_train, I_train = segment_data(data[train_idx], label[train_idx], segment_size, step_size)
        X_validate, Y_validate, I_validate = segment_data(data[val_idx], label[val_idx], segment_size, step_size)
    if len(X_train) == 0 or len(X_validate) == 0:
        raise ValueError('Training and validation each need at least one complete, finite window.')
    if not params.get('preprocessed_cache'):
        X_train = resample(X_train, samples, axis=2)
        X_validate = resample(X_validate, samples, axis=2)
        X_train = preprocess_windows(X_train, params['downsrate'], mode,
                                     bandpass=params['bandpass_filt'], broadband_residual=residual_mode)
        X_validate = preprocess_windows(X_validate, params['downsrate'], mode,
                                        bandpass=params['bandpass_filt'], broadband_residual=residual_mode)
    if not np.isfinite(X_train).all() or not np.isfinite(X_validate).all():
        raise ValueError('Non-finite preprocessed EEG; check constant channels or corrupt trials.')
    Y_train = np_utils.to_categorical(Y_train - 1, num_classes=params['nclass'])
    Y_validate = np_utils.to_categorical(Y_validate - 1, num_classes=params['nclass'])
    model = build_model(arch, params['nclass'], nChan, samples, mode=mode, dropout=dropout, depth=depth,
                        broadband_residual=residual_mode)
    if fine_tune:
        model.load_weights(params['modelpath'])
        # Trainability must be set BEFORE compile.
        configure_finetuning(model, params.get('finetune_policy', 'legacy_last12'))
    lr = ((2e-4 if arch == 'EEGNet' else 5e-5) if fine_tune
          else (.005 if arch == 'EEGNet' else .0005))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.1),
                  metrics=['accuracy'])
    config.update(learning_rate=lr, parameter_count=model.count_params(),
                  trainable_parameter_count=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
                  trainable_weight_layers=[layer.name for layer in model.layers if layer.trainable_weights],
                  n_train_windows=len(X_train), n_validation_windows=len(X_validate))
    write_json(save_path.with_suffix('.config.json'), config)
    np.savez_compressed(save_path.with_suffix('.split.npz'),
                        train_trial_indices=train_idx, val_trial_indices=val_idx,
                        train_window_trial_indices=train_idx[I_train],
                        val_window_trial_indices=val_idx[I_validate], trial_labels=label)
    if verbose:
        print('Filter-bank mode:', mode, 'Broadband residual:', residual_mode, 'Input:', X_train.shape)
        model.summary()
    callbacks = [
        BestAccuracyCheckpoint(str(save_path), verbose=1 if verbose else 0),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80),
        ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=30),
        EpochHistory(save_path),
    ]
    # Keep complete datasets in host RAM; only individual batches go to the GPU.
    with tf.device('/CPU:0'):
        X_train, Y_train = tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train)
        X_validate, Y_validate = tf.convert_to_tensor(X_validate), tf.convert_to_tensor(Y_validate)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_validate, Y_validate), verbose=verbose,
                        callbacks=callbacks,
                        class_weight={i: 1 for i in range(params['nclass'])})
    # Report coefficients from the checkpoint used for evaluation, not the last epoch.
    model.load_weights(save_path)
    weights = get_band_weights(model)
    residual_info = get_residual_info(model)
    if residual_info is not None:
        write_json(save_path.with_suffix('.residual.json'),
                   dict(residual_info, checkpoint=str(save_path.resolve()),
                        scope='global residual coefficient; not a percentage or causal importance'))
    write_json(save_path.with_suffix('.history.json'), history.history)
    if weights is not None:
        write_json(save_path.with_suffix('.band_weights.json'),
                   dict(mode=mode, bands=config['bands'], weights=weights,
                        scope='global model coefficients; not class-specific or causal importance',
                        checkpoint=str(save_path.resolve())))
    config.update(status='complete', epochs_completed=len(history.epoch),
                  best_val_accuracy=max(history.history['val_accuracy']),
                  checkpoint_sha256=checkpoint_hash(save_path))
    write_json(save_path.with_suffix('.config.json'), config)
    plot_training_history(history, str(save_path))
    print('Training finished. Best checkpoint:', save_path)
    return str(save_path)


def plot_training_history(history, save_name):
    """
    Plots the training and validation loss and accuracy.
    """
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plot_save_path = save_name.replace('.h5', '_history.png')
    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"Training history plot saved to {plot_save_path}")
    plt.close()
