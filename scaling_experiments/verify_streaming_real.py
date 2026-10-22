"""Compare the new loader against the original pipeline on real MAT files."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from scipy.signal import resample
from bci.Functions import load_and_filter_data, segment_data
from bci.filter_bank import preprocess_windows
from scaling_experiments.streaming_data import DATA_ROOT, decode_mat, transform_trials, save_json

results=[]
for rel in ['S01/OfflineMovement/S01_OfflineMovement_R01.mat',
            'S09/OnlineMovement_Sess01_3class_Base/S09_OnlineMovement_Sess01_3class_Base_R10.mat',
            'S21/OnlineMovement_Sess02_2class_Base/S21_OnlineMovement_Sess02_2class_Base_R01.mat']:
    path=DATA_ROOT/rel
    x,labels,info=decode_mat(path)
    with tempfile.TemporaryDirectory() as td:
        (Path(td)/path.name).symlink_to(path)
        with contextlib.redirect_stdout(io.StringIO()):
            old,old_labels,_=load_and_filter_data([td],dict(nclass=2,maxtriallen=5))
    np.testing.assert_array_equal(x,old)
    np.testing.assert_array_equal(labels,old_labels)
    old_windows,old_y,old_i=segment_data(old[:3],old_labels[:3],3072,128)
    if len(old_windows):
        expected=preprocess_windows(resample(old_windows,300,axis=2),100,'off').astype(np.float32)
        chunks=list(transform_trials(x[:3],labels[:3],chunk_size=2))
        actual=np.concatenate([c[0] for c in chunks])
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]),old_i)
        max_error=float(np.max(np.abs(actual-expected)))
    else:
        raise ValueError('Audit selected no windows')
    results.append(dict(file=rel,sha256=info['sha256'],trials=len(x),windows_compared=len(actual),max_error=max_error))
save_json('/home/kxy/LYC/experiments/conformer_scaling_S21_20261021/preprocessing_equivalence.json',results)
print(json.dumps(results,indent=2))
