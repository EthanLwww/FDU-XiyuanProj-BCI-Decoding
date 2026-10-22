# Filter-bank Implementation Plan

**Goal:** Add an optional four-band frontend to EEGNet and Conformer for controlled experiments.
**Architecture:** Pure NumPy/SciPy preprocessing; registered Keras band layers and builders; shared training CLI/configuration.
**Tech Stack:** Python, NumPy, SciPy, TensorFlow/Keras, unittest.
**Spec:** ../specs/2026-10-20-filter-bank-design.md
**Execution:** Inline in an isolated copy because the target project is not a Git checkout. Back up changed remote files before deployment.

## Task 1: Signal preprocessing and reproducible splitting
Files: filter_bank.py, tests/test_filter_bank.py.
API: preprocess_windows(data, sampling_rate=100, mode="off", bands=DEFAULT_BANDS, bandpass=(4,40), chunk_size=128); filter_bank_transform(data, sampling_rate, bands, standardize=True, chunk_size=128); split_trial_indices(n_trials, seed=None).
- [x] Write and run failing tests for four sine frequencies, constant signals, off numerical regression, and deterministic trial splitting.
- [x] Implement shape/band/mode validation and chunked causal SOS filtering; retain legacy off calculations.
- [x] Run these tests to green.

## Task 2: Independent branches and fusion
Files: FilterBankModels_tf.py, tests/test_filter_bank_models.py; existing EEGModels_tf.py stays unchanged.
API: build_model(model_arch, nb_classes, Chans, Samples, mode="off", dropout=0.5, depth=3); get_band_weights(model).
- [x] Write failing tests for per-band independence, nonzero finite gradients, disabled compatibility, and serialization.
- [x] Implement registered BandSlice and BandFusion, EEGNet/Conformer frontends, and shared-size classification backends.
- [x] Run model tests for off/mean/weighted/concat with two and three classes.

## Task 3: Training, metadata and command-line switches
Files: training_cli.py, Functions.py, main_model_training.py, main_multi_subject_training.py, tests/test_training_integration.py.
- [x] Write tests for old CLI compatibility, mode/seed output separation, checkpoint compatibility, and a one-epoch fit/finetune.
- [x] Connect preprocessing and model factory; expose --filter-bank, --seed, --epochs, --batch-size, --data-dir, --save-dir, --init-model.
- [x] Save JSON configuration/history, split NPZ and best-checkpoint band weights. Freeze before compile; preserve trainable band fusion.
- [x] Execute smoke tests on synthetic EEG without reading the full dataset.

## Task 4: Documentation, review and delivery
Files: FILTER_BANK.md, README.md, requirements-filterbank.txt.
- [x] Document matched-seed commands, unchanged upstream baseline, interpretation limits, memory behavior and online support boundary.
- [x] Run complete tests and inspect the diff against the copied remote original.
- [x] Back up old files, verify remote source hashes have not changed, deploy only reviewed files, and verify deployed tests and baseline Git cleanliness.
