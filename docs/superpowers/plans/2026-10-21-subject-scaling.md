# EEGConformer subject scaling implementation plan

**Goal:** Run one seed (42) for S01–05, S01–10 and S01–20 pretraining, and fine-tune each on S21. Reuse the verified S01/S02/S05 baseline pretraining checkpoint and fine-tune it on the same S21 partition.

**Architecture:** Keep the verified model and signal transform unchanged. Read one MAT file at a time, write float32 window shards per subject, and use memory mapping to assemble only the current training batch. Separate this experiment in `scaling_experiments/` and a new remote results directory.

**Tech stack:** Existing Python, NumPy/SciPy, TensorFlow 2.17, HDF5 checkpoints, Matplotlib.

**Authorized design:** User requested the three cohorts, no repeats, S21 transfer, reuse of the original three-subject reference and batching where necessary. User confirmed baseline model only. Data root is `/Dataset2/dataset_bci_decode_kxy`.

## Constraints

- ME thumb/pinky, labels 1/4 remapped to 1/2, 128 channels, depth 1, 3 seconds, 100 Hz, step 128 raw samples, 4–40 Hz, original CAR and per-window standardization.
- Pretraining includes OfflineMovement and both online sessions, Base and Finetune. S21 uses Session 2 Base, both two/three-class files.
- Seed 42, trial split 80/20 before windows. Same S21 split in all four groups. S21 is excluded from every pretraining cohort.
- Batch 32, Adam LR .0005/.00005, original dropout .5/.65, 300/100 epoch ceilings, loss early stop 80, LR patience 30/factor .5, first highest validation accuracy checkpoint.
- No corrupt file is silently skipped. Record actual counts and differing run counts for S09/S13/S19. Record each source size, mtime, and hash as it is read.
- Preserve old runs. Reused three-subject results are one seed, not an average of previous repeats. Report validation performance and single-run limitations.

## Tasks

- [x] Verify streaming preprocessing against the original pipeline on synthetic and actual MAT data. Test short trial exclusion, label alignment, cross-shard batch ordering and complete single-epoch coverage.
- [x] Implement file-by-file preprocessing with atomic completion markers, source manifests, disk shards and a bounded batch reader. Check finite values and trial partition leakage.
- [x] Add a baseline-only runner with bounded prefetch, GPU allocation cap, source snapshots, history/checkpoint/config/split output and full checkpoint evaluation.
- [x] Audit and prepare S01–S21. Reuse the original pretraining checkpoint by verified hash. Launch the three new pretraining runs and all four S21 fine-tunes on available GPUs.
- [x] Verify all outputs, aggregate loss/accuracy curves and actual dataset-size versus S21 accuracy, write a Chinese report, and copy final results to the local workspace.

## Completion evidence

All four S21 runs and all four pretraining checkpoints were verified by full validation evaluation. The three-subject pretraining checkpoint was reused. S21 best validation accuracies: 55.88%, 58.82%, 64.71%, 67.65%. All S21 partitions match. No OOM occurred. New pretraining peak RSS: 7.80/12.63/22.14 GiB for 5/10/20 subjects. Source files and all eight checkpoints verified locally after archive transfer.

One orphan TrialEnd in S09 was identified by a complete 1,924-file event scan. Event-order pairing records and ignores only that orphan end, preventing subsequent trial boundary misalignment. All ordinary-file preprocessing remains numerically equivalent to the original. Source revisions before the loader correction are retained in the experiment bundle.
