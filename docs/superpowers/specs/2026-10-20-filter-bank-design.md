# Switchable filter-bank frontend

The user requested a four-band frontend integrated into the existing remote project at /home/kxy/LYC/Finger-BCI-Decoding-main. Keep the upstream baseline untouched.

## Data and models
Modes: off (default), mean, weighted, concat. Bands in Hz: [4,8], [8,13], [13,30], [30,40]. Input is (N,C,T,4) when enabled; channels and window length remain data/config driven. Split trials before windowing. Use fourth-order causal Butterworth SOS filters for the new frontend, 100 zero samples of padding on either side, independent temporal z-score per band/channel, float32 output and chunked processing. Off follows the existing single-band lfilter preprocessing.

Each band has its own temporal convolution and depthwise spatial convolution. EEGNet branches keep their separable temporal block; Conformer branches project to 40-dimensional tokens, followed by one shared existing-size encoder/head. Normalize branch features before fusion. Weighted fusion uses four globally learned softmax logits initialized uniformly; mean is its untrained control. Concat concatenates features and, for Conformer, projects back to embedding width. No positional encoding or extra Transformer depth is introduced.

## Experiments and compatibility
Expose the same flags in single/multi-subject training, preserving existing positional arguments. Seed is optional; supplied seeds fix trial splitting and initialization. Save trial split indices, resolved configuration, training history, and best-checkpoint band weights. Filenames include mode and optional seed, preventing off/mean/weighted/concat collisions. Retain original filename when no new flags are supplied. Include actual subject IDs in multi-subject filenames to avoid confusing different cohorts of the same size.

Finetuning must select a compatible checkpoint; reject incompatible filter settings and tensor shapes. Freeze early layers before compile; allow the weighted fusion logits to adapt during finetuning. New filter-bank checkpoints need FilterBankModels_tf imported for custom-layer registration. Existing online BCPy2000 integration remains EEGNet single-band only; documentation must explicitly state this boundary.

## Interpretation
Weights are global model preferences, not class-specific/subject-specific causal importance. Per-band z-score removes absolute band power. Filter-bank alone does not multiply sample count. Model parameters and activations grow in the frontend, not in the shared Conformer encoder. No claim of accuracy improvement without matched-split training.

## Verification
Synthetic sine-wave band selectivity, finite output on constant input, unchanged off preprocessing/model, no cross-band mixing before fusion, normalized and trainable weights, all modes on both models, compatible serialization, deterministic trial splitting, and a one-epoch training/finetuning smoke test. Use CPU and bounded thread counts; do not start a full dataset run.
