"""Real-data, dataset-tail, serialization and finetuning checks before training."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from run_scaling_residual import ROOT, env_for, prepare_verified

if '--configured' not in sys.argv:
    raise SystemExit(subprocess.call([sys.executable, __file__, '--configured'], env=env_for(0)))

import numpy as np
import tensorflow as tf
for device in tf.config.list_physical_devices('GPU'):
    tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=16384)])
from Functions import configure_finetuning
from FilterBankModels_tf import build_model
from filter_bank import preprocess_windows
from streaming_data import DATA_ROOT, WindowStore, decode_mat, transform_trials, tf_dataset, save_json
from scipy.signal import resample

def main():
    prepare_verified(ROOT, 21)
    equivalence=[]
    for rel in ['S01/OfflineMovement/S01_OfflineMovement_R01.mat',
                'S21/OnlineMovement_Sess02_2class_Base/S21_OnlineMovement_Sess02_2class_Base_R01.mat']:
        raw, labels, meta = decode_mat(DATA_ROOT/rel)
        windows, ids = [], []
        for start in range(0, 5120-3072+1, 128):
            part=raw[:3, :, start:start+3072]
            valid=~np.isnan(part).any(axis=(1,2))
            windows.extend(part[valid])
            ids.extend(np.flatnonzero(valid))
        expected=preprocess_windows(resample(np.array(windows),300,axis=2),100,'weighted',
                                    broadband_residual='gated')
        chunks=list(transform_trials(raw[:3],labels[:3],chunk_size=2,mode='weighted',residual_mode='gated'))
        actual=np.concatenate([chunk[0] for chunk in chunks])
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(np.concatenate([chunk[1] for chunk in chunks]),ids)
        equivalence.append(dict(file=rel,sha256=meta['sha256'],windows=len(actual),max_absolute_error=0.0))
    store=WindowStore([ROOT/'cache'/'S21'])
    train_ids,val_ids=store.split(42)
    ds=tf_dataset(store,train_ids,shuffle=True)
    sizes=[len(x) for x,y in ds]
    assert sizes==[32]*8+[6],sizes
    tf.keras.utils.set_random_seed(42)
    model=build_model('EEGConformer',2,128,300,mode='weighted',depth=1,broadband_residual='gated')
    assert model.input_shape==(None,128,300,5) and model.count_params()==412751
    model.compile(optimizer=tf.keras.optimizers.Adam(.0005),loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=.1))
    x=store.read(train_ids[:2]); y=np.eye(2)[store.labels[train_ids[:2]]-1]
    loss=float(model.train_on_batch(x,y))
    assert np.isfinite(loss)
    before=model(x,training=False).numpy()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'smoke.h5'
        model.save(path)
        loaded=tf.keras.models.load_model(path,compile=False)
        np.testing.assert_allclose(before,loaded(x,training=False).numpy(),rtol=1e-6,atol=1e-7)
        configure_finetuning(loaded,'conformer_baseline')
        trainable=[layer.name for layer in loaded.layers if layer.trainable_weights]
        assert len(trainable)==6 and {'band_fusion','broadband_residual'}.issubset(trainable)
    result=dict(status='verified',data_root=str(DATA_ROOT),input_shape=[128,300,5],
                parameter_count=412751,real_data_equivalence=equivalence,
                training_batch_sizes=sizes,serialization_prediction_match=True,
                trainable_layers_finetune=trainable,smoke_loss=loss)
    save_json(ROOT/'preflight.json',result)
    print(result,flush=True)

if __name__=='__main__':
    main()
