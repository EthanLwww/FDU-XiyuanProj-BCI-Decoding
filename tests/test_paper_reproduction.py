"""Regression checks for the paper's raw-window preprocessing and CV accounting."""
import tempfile
import unittest
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, resample
from scipy.stats import zscore


class PaperPipelineTests(unittest.TestCase):
    def setUp(self):
        from experiment_runners import paper_pipeline
        self.p = paper_pipeline

    def test_raw_stride_and_short_trials_preserve_all_valid_windows(self):
        x = np.random.default_rng(13).normal(size=(3, 3, 5120))
        x[1, :, 3072:] = np.nan
        x[2, :, 900:] = np.nan
        windows, ids, starts = self.p.preprocess_trials(x)
        self.assertEqual(windows.shape, (50, 3, 100, 1))
        self.assertEqual(np.bincount(ids, minlength=3).tolist(), [33, 17, 0])
        self.assertEqual(starts[ids == 0].tolist(), list(range(0, 4097, 128)))
        self.assertEqual(starts[ids == 1][-1], 2048)
        self.assertTrue(np.isfinite(windows).all())

    def test_windows_match_upstream_numeric_reference_for_every_start(self):
        x = np.random.default_rng(7).normal(size=(2, 3, 5120))
        x[1, :, 3072:] = np.nan
        expected, trial_ids, positions = [], [], []
        b, a = butter(4, [4, 40], btype='bandpass', fs=100)
        for start in range(0, 4097, 128):
            for trial in range(2):
                raw = x[trial, :, start:start+1024]
                if np.isnan(raw).any():
                    continue
                down = resample(raw, 100, axis=-1)
                pad = np.pad(down, ((0, 0), (100, 100)))
                causal = lfilter(b, a, pad, axis=-1)[:, 100:-100]
                expected.append(zscore(causal, axis=-1)[..., None].astype(np.float32))
                trial_ids.append(trial)
                positions.append(start)
        actual, ids, starts = self.p.preprocess_trials(x)
        np.testing.assert_array_equal(actual, np.array(expected))
        np.testing.assert_array_equal(ids, trial_ids)
        np.testing.assert_array_equal(starts, positions)

    def test_non_1024_rate_is_rejected_instead_of_silently_hardcoded(self):
        with self.assertRaises(ValueError):
            self.p.preprocess_trials(np.ones((1, 2, 5120)), fs=1000)

    def test_constant_channel_cannot_be_silently_replaced_with_zero(self):
        x = np.random.default_rng(1).normal(size=(1, 2, 5120))
        x[:, 0, :] = 0
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.p.preprocess_trials(x)

    def test_oof_split_covers_every_eligible_trial_once_without_window_leakage(self):
        labels = np.array([1, 4] * 16)
        groups = np.repeat(np.arange(30), 3)
        folds = self.p.make_folds(labels, groups, n_splits=5, seed=42)
        held_out = []
        for split in folds:
            train, val = split['train_trials'], split['val_trials']
            self.assertFalse(set(train) & set(val))
            self.assertFalse(set(groups[split['train_windows']]) & set(groups[split['val_windows']]))
            held_out.extend(val.tolist())
        self.assertEqual(sorted(held_out), list(range(30)))

    def test_majority_has_one_vote_result_per_trial_with_explicit_tie_rule(self):
        rows = self.p.vote_trials(np.array([0, 1, 0, 1, 1]), np.array([3, 3, 3, 8, 8]),
                                  np.array([0]*8 + [1]), [1, 4])
        self.assertEqual([(r['trial_index'], r['pred_label'], r['n_windows']) for r in rows],
                         [(3, 1, 3), (8, 4, 2)])
        tie = self.p.vote_trials(np.array([1, 0]), np.array([3, 3]), np.array([0]*4), [1, 4])
        self.assertTrue(tie[0]['tie'])
        self.assertEqual(tie[0]['pred_label'], 1)

    def test_predictions_and_labels_must_have_matching_window_lengths(self):
        with self.assertRaises(ValueError):
            self.p.vote_trials(np.array([0, 1]), np.array([3]), np.array([0]*4), [1, 4])

    def test_atomic_json_rejects_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self.p.save_json(Path(td)/'result.json', {'accuracy': float('nan')})
            self.assertFalse((Path(td)/'result.json').exists())


class PaperResultAuditTests(unittest.TestCase):
    """A plausible summary must not hide corrupted per-fold evaluation records."""

    def setUp(self):
        from experiment_runners.paper_pipeline import save_json, sha256
        from experiment_runners.paper_report import audit_condition
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.save = save_json
        self.audit = audit_condition
        folds = []
        all_trials = set(range(10))
        for fold in range(1, 6):
            directory = self.root/f'fold{fold}'
            directory.mkdir()
            val = [2*(fold-1), 2*(fold-1)+1]
            train = sorted(all_trials-set(val))
            windows = [2*t+i for t in val for i in (0, 1)]
            self.save(directory/'split.json', dict(fold=fold, train_trials=train, val_trials=val,
                train_windows=[2*t+i for t in train for i in (0, 1)], val_windows=windows))
            (directory/'best.weights.h5').write_bytes(b'synthetic checkpoint for integrity audit')
            np.savez(directory/'predictions.npz', probabilities=[[.9,.1],[.8,.2],[.2,.8],[.1,.9]],
                     trial_indices=np.repeat(val, 2), raw_starts=[0,128,0,128], true_encoded_labels=[0,0,1,1])
            with (directory/'trial_predictions.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=['trial_index','true_label','pred_label','correct','n_windows','tie','votes'])
                writer.writeheader()
                for trial,label,votes in zip(val,[1,4],[[2,0],[0,2]]):
                    writer.writerow(dict(trial_index=trial,true_label=label,pred_label=label,
                                         correct=True,n_windows=2,tie=False,votes=votes))
            (directory/'history.csv').write_text('epoch,loss,accuracy,val_loss,val_accuracy,lr\n0,.3,.9,.4,1.0,.001\n1,.2,1.0,.6,.5,.001\n')
            folds.append(dict(fold=fold,n_train_trials=8,n_val_trials=2,n_train_windows=16,n_val_windows=4,
                trial_accuracy=1.,n_correct_trials=2,window_accuracy=1.,loss=.4,checkpoint_epoch=1,
                epochs_ran=2,best_val_window_accuracy=1.,checkpoint_sha256=sha256(directory/'best.weights.h5'),
                checkpoint_reloaded_and_verified=True,trial_leakage=False,ties=0))
        self.summary = dict(config=dict(smoke=False,folds=5,labels=[1,4]),completed_folds=5,
            n_eligible_trials=10,n_windows=20,mean_trial_accuracy=1.,pooled_trial_accuracy=1.,
            mean_window_accuracy=1.,std_trial_accuracy=0.,folds=folds)
        self.save(self.root/'result.json',self.summary)

    def test_valid_five_fold_outputs_can_be_audited_without_original_training_cache(self):
        self.assertEqual(self.audit(self.root)['pooled_trial_accuracy'],1.)

    def test_incorrect_summary_mean_is_rejected_before_plotting(self):
        self.summary['mean_trial_accuracy'] = .65
        self.save(self.root/'result.json',self.summary)
        with self.assertRaisesRegex(ValueError,'mean_trial_accuracy'):
            self.audit(self.root)

    def test_duplicate_trial_vote_cannot_replace_another_trial(self):
        path = self.root/'fold1/trial_predictions.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        with path.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows([rows[0],rows[0]])
        with self.assertRaisesRegex(ValueError,'Vote trial coverage'):
            self.audit(self.root)

    def test_checkpoint_selection_must_match_the_logged_best_epoch(self):
        self.summary['folds'][0]['checkpoint_epoch'] = 2
        self.save(self.root/'result.json',self.summary)
        with self.assertRaisesRegex(ValueError,'Checkpoint selection'):
            self.audit(self.root)


if __name__ == '__main__':
    unittest.main()
