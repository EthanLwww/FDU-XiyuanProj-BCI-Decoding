import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import signal, stats


class StreamingTests(unittest.TestCase):
    def test_orphan_trial_end_cannot_shift_subsequent_trial_boundaries(self):
        from scaling_experiments.streaming_data import pair_trial_events
        events = [('Target', 10, 4, 0), ('TrialEnd', 20, 1, 1),
                  ('TrialEnd', 22, 1, 2), ('Target', 30, 1, 3), ('TrialEnd', 40, 1, 4)]
        trials, ignored = pair_trial_events(events)
        self.assertEqual(trials, [(10, 20, 4, 0), (30, 40, 1, 1)])
        self.assertEqual(ignored, [dict(event_index=2, sample_zero_based=22, reason='TrialEnd without an open Target')])

    def test_chunk_transform_equals_original_and_excludes_short_trials(self):
        from scaling_experiments.streaming_data import transform_trials
        x = np.random.default_rng(4).normal(size=(3, 5, 5120))
        x[1, :, 3000:] = np.nan
        x[2, :, 4000:] = np.nan
        labels = np.array([1, 2, 1])
        expected, trial_ids, starts = [], [], []
        for start in range(0, 5120 - 3072 + 1, 128):
            for trial in range(3):
                raw = x[trial, :, start:start + 3072]
                if np.isnan(raw).any():
                    continue
                y = signal.resample(raw, 300, axis=1)
                b, a = signal.butter(4, [4, 40], btype='bandpass', fs=100)
                y = signal.lfilter(b, a, np.pad(y, ((0, 0), (100, 100))), axis=-1)[:, 100:-100]
                expected.append(stats.zscore(y, axis=1)[..., None].astype(np.float32))
                trial_ids.append(trial)
                starts.append(start)
        chunks = list(transform_trials(x, labels, chunk_size=2))
        actual = np.concatenate([c[0] for c in chunks])
        np.testing.assert_array_equal(actual, np.array(expected))
        np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), trial_ids)
        np.testing.assert_array_equal(np.concatenate([c[2] for c in chunks]), starts)

    def test_disk_batch_preserves_order_duplicates_and_incomplete_last_batch(self):
        from scaling_experiments.streaming_data import WindowStore, epoch_batches, save_json
        with tempfile.TemporaryDirectory() as td:
            paths = []
            truth = np.arange(7*2*3, dtype=np.float32).reshape(7, 2, 3, 1)
            for sid, (first, stop) in enumerate([(0, 3), (3, 7)]):
                p = Path(td)/str(sid)
                p.mkdir()
                truth[first:stop].tofile(p/'windows.dat')
                np.savez(p/'index.npz', trial_ids=np.arange(stop-first), starts=np.zeros(stop-first, int), labels=np.ones(stop-first, int))
                save_json(p/'ready.json', dict(shape=list(truth[first:stop].shape), n_trials=stop-first, trial_labels=[1]*(stop-first)))
                paths.append(p)
            store = WindowStore(paths)
            ids = np.array([6, 0, 4, 4, 2])
            np.testing.assert_array_equal(store.read(ids), truth[ids])
            batches = list(epoch_batches(np.arange(7), 3, seed=42))
            self.assertEqual([len(b) for b in batches], [3, 3, 1])
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(7))
            a, b = store.split(42)
            self.assertFalse(set(store.trial_ids[a]) & set(store.trial_ids[b]))
            self.assertEqual(len(a)+len(b), 7)
            del store

    def test_residual_transform_matches_filter_bank_preprocessing_and_keeps_five_planes(self):
        from scaling_experiments.streaming_data import transform_trials
        from bci.filter_bank import preprocess_windows
        x = np.random.default_rng(12).normal(size=(2, 128, 5120))
        labels = np.array([1, 2])
        chunks = list(transform_trials(x, labels, chunk_size=1, mode='weighted',
                                        residual_mode='gated'))
        actual = np.concatenate([c[0] for c in chunks])
        expected, starts = [], []
        for start in range(0, 5120 - 3072 + 1, 128):
            raw = x[:, :, start:start + 3072]
            expected.append(preprocess_windows(signal.resample(raw, 300, axis=2), 100,
                                               'weighted', bandpass=[4, 40],
                                               broadband_residual='gated'))
            starts.extend([start] * 2)
        expected = np.concatenate(expected).astype(np.float32)
        np.testing.assert_equal(actual.shape, (34, 128, 300, 5))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_residual_window_store_reads_five_planes_and_keeps_tail_batch(self):
        from scaling_experiments.streaming_data import WindowStore, epoch_batches, save_json
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'S01'
            p.mkdir()
            truth = np.arange(5 * 2 * 3 * 5, dtype=np.float32).reshape(5, 2, 3, 5)
            truth.tofile(p / 'windows.dat')
            np.savez(p / 'index.npz', trial_ids=np.arange(5), starts=np.zeros(5, int),
                     labels=np.ones(5, int))
            save_json(p / 'ready.json', dict(shape=list(truth.shape), n_trials=5,
                                             trial_labels=[1, 1, 1, 1, 1]))
            store = WindowStore([p])
            np.testing.assert_equal(store.shape, truth.shape)
            np.testing.assert_array_equal(store.read(np.array([4, 0])), truth[[4, 0]])
            batches = list(epoch_batches(np.arange(5), 2, seed=42))
            self.assertEqual([len(b) for b in batches], [2, 2, 1])
            del store


if __name__ == '__main__':
    unittest.main()
