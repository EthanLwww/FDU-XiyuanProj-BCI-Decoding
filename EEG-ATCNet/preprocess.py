""" 
Copyright (C) 2022 King Saud University, Saudi Arabia 
SPDX-License-Identifier: Apache-2.0 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the 
License at

http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. 

Author:  Hamdi Altaheri 
"""

# Dataset BCI Competition IV-2a is available at 
# http://bnci-horizon-2020.eu/database/data-sets

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, resample_poly
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# %%
def _event_text(value):
    """Convert MATLAB string/bytes values to ordinary Python text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def load_custom_file(mat_file, target_fs=250, tmin=1.0, tmax=4.0,
                     lowcut=4.0, highcut=40.0, label_map=None):
    """Load one OfflineImagery MAT file.

    Parameters
    ----------
    mat_file : str or Path
        Path to a file containing ``eeg`` and ``event`` fields.
    target_fs : int
        Sampling rate returned to the model.
    tmin, tmax : float
        Trial window in seconds after each Target event. The default is
        Target + 1 s through Target + 4 s (3 seconds).
    lowcut, highcut : float
        Band-pass filter limits in Hz. Set either to None to disable filtering.
    label_map : dict, optional
        Mapping from original labels to zero-based labels. Defaults to
        ``{1: 0, 2: 1, 3: 2, 4: 3}``.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    y : ndarray, shape (n_trials,)
    """
    if label_map is None:
        label_map = {1: 0, 2: 1, 3: 2, 4: 3}

    mat_file = Path(mat_file)
    mat = sio.loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
    if "eeg" not in mat or "event" not in mat:
        raise KeyError(f"{mat_file} 必须包含 eeg 和 event 字段")

    eeg = mat["eeg"]
    data = np.asarray(eeg.data, dtype=np.float32)
    fs = float(np.asarray(eeg.fsample).squeeze())
    if data.ndim != 2:
        raise ValueError(f"{mat_file} 的 eeg.data 应为二维数组，实际为 {data.shape}")
    # The dataset is channels x time. Transpose if a file happens to be time x channels.
    n_channels = int(np.asarray(eeg.nChans).squeeze())
    if data.shape[0] != n_channels and data.shape[1] == n_channels:
        data = data.T
    if data.shape[0] != n_channels:
        raise ValueError(f"通道数不一致：eeg.nChans={n_channels}, data.shape={data.shape}")

    if lowcut is not None and highcut is not None:
        sos = butter(4, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
        data = sosfiltfilt(sos, data, axis=1).astype(np.float32)

    events = np.asarray(mat["event"], dtype=object).reshape(-1)
    window_start = int(round(tmin * fs))
    window_stop = int(round(tmax * fs))
    trials, labels = [], []
    for event in events:
        event_type = _event_text(getattr(event, "type", ""))
        if event_type != "Target":
            continue
        original_label = int(np.asarray(getattr(event, "value")).squeeze())
        if original_label not in label_map:
            continue
        # MATLAB sample indices are 1-based; Python array indices are 0-based.
        target_sample = int(np.asarray(getattr(event, "sample")).squeeze()) - 1
        start = target_sample + window_start
        stop = target_sample + window_stop
        if start < 0 or stop > data.shape[1]:
            continue
        trial = data[:, start:stop]
        if target_fs != int(round(fs)):
            trial = resample_poly(trial, target_fs, int(round(fs)), axis=1)
        trials.append(trial.astype(np.float32))
        labels.append(label_map[original_label])

    if not trials:
        raise ValueError(f"{mat_file} 中没有可用的 Target trial")
    return np.stack(trials), np.asarray(labels, dtype=np.int64)


def load_custom_subject(data_path, subject, **kwargs):
    """Load and concatenate all OfflineImagery MAT files for one subject."""
    subject_name = f"S{int(subject):02d}"
    folder = Path(data_path) / subject_name / "OfflineImagery"
    files = sorted(folder.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"未找到 {folder} 下的 MAT 文件")
    all_x, all_y = zip(*(load_custom_file(file, **kwargs) for file in files))
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


def load_custom_loso(data_path, test_subject, subject_ids=None, **kwargs):
    """Load all subjects, keeping one subject entirely for testing.

    ``test_subject`` and ``subject_ids`` use the zero-based indexing expected
    by the training scripts; the on-disk folders are named S01, S02, etc.
    """
    root = Path(data_path)
    if subject_ids is None:
        subject_ids = sorted(
            int(folder.name[1:])
            for folder in root.glob("S*")
            if folder.is_dir() and folder.name[1:].isdigit()
            and (folder / "OfflineImagery").is_dir()
        )
    if not subject_ids:
        raise FileNotFoundError(f"在 {root} 下没有找到 Sxx/Sxx/OfflineImagery 数据目录")
    if test_subject < 0 or test_subject >= len(subject_ids):
        raise IndexError(
            f"LOSO 被试索引 {test_subject} 超出范围；数据中有 {len(subject_ids)} 个被试"
        )

    train_x, train_y = [], []
    for index, subject_id in enumerate(subject_ids):
        X, y = load_custom_subject(root, subject_id, **kwargs)
        if index == test_subject:
            X_test, y_test = X, y
        else:
            train_x.append(X)
            train_y.append(y)
    if not train_x:
        raise ValueError("LOSO 至少需要两个被试")
    return (np.concatenate(train_x), np.concatenate(train_y), X_test, y_test)


def load_custom_subject_split(data_path, subject, train_ratio=0.8, **kwargs):
    """Split one subject by run files, avoiding trial-level run leakage."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    subject_name = f"S{int(subject):02d}"
    folder = Path(data_path) / subject_name / "OfflineImagery"
    files = sorted(folder.glob("*.mat"))
    if len(files) < 2:
        raise ValueError(f"{folder} 至少需要两个 run 文件才能划分训练集和测试集")
    split = int(len(files) * train_ratio)
    split = min(max(split, 1), len(files) - 1)

    def load_files(selected_files):
        xy = [load_custom_file(file, **kwargs) for file in selected_files]
        return np.concatenate([item[0] for item in xy]), np.concatenate([item[1] for item in xy])

    X_train, y_train = load_files(files[:split])
    X_test, y_test = load_files(files[split:])
    return X_train, y_train, X_test, y_test

# We need the following function to load and preprocess the High Gamma Dataset
# from preprocess_HGD import load_HGD_data

#%%
def load_data_LOSO (data_path, subject, dataset): 
    """ Loading and Dividing of the data set based on the 
    'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the 
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that 
    separate subjects (not visible in the training data) are usedto evaluate 
    the model.
    
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available at 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9/14]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    X_train, y_train = [], []
    for sub in range (0,9):
        path = data_path+'s' + str(sub+1) + '/'
        
        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub+1, True)
            X2, y2 = load_BCI2a_data(path, sub+1, False)
        elif (dataset == 'CS2R'):
            X1, y1, _, _, _  = load_CS2R_data_v2(path, sub, True)
            X2, y2, _, _, _  = load_CS2R_data_v2(path, sub, False)
        # elif (dataset == 'HGD'):
        #     X1, y1 = load_HGD_data(path, sub+1, True)
        #     X2, y2 = load_HGD_data(path, sub+1, False)
        
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (sub == subject):
            X_test = X
            y_test = y
        elif len(X_train) == 0:  
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


#%%
def load_BCI2a_data(data_path, subject, training, all_trials = True):
    """ Loading and Dividing of the data set based on the subject-specific 
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training, 
    and 288 x 9 trials in session 2 for testing.  
   
        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on 
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts 
    """
    
    # Define MI-trials parameters
    n_channels = 22
    n_tests = 6*48     
    window_Length = 7*250 
    
    # Define MI trial window 
    fs = 250          # sampling rate
    t1 = int(1.5*fs)  # start time_point
    t2 = int(6*fs)    # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1        
    

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return



#%%
import json
import glob as glob
from datetime import datetime

def load_CS2R_data_v2(data_path, subject, training, 
                      classes_labels =  ['Fingers', 'Wrist','Elbow','Rest'], 
                      all_trials = True):
    """ Loading training/testing data for the CS2R motor imagery dataset
    for a specific subject        
   
        Parameters
        ----------
        data_path: string
            dataset path
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training data
            if False, load testing data
        classes_labels: tuple
            classes of motor imagery returned by the method (default: all) 
    """
    # Keep these optional CS2R-only dependencies out of module import time so
    # the MAT-based datasets can be loaded without installing MNE.
    from mne.io import read_raw_edf
    from dateutil.parser import parse
    
    # Get all subjects files with .edf format.
    subjectFiles = glob.glob(data_path + 'S_*/')
    
    # Get all subjects numbers sorted without duplicates.
    subjectNo = list(dict.fromkeys(sorted([x[len(x)-4:len(x)-1] for x in subjectFiles])))
    # print(SubjectNo[subject].zfill(3))
    
    if training:  session = 1
    else:         session = 2
    
    num_runs = 5
    sfreq = 250 #250
    mi_duration = 4.5 #4.5

    data = np.zeros([num_runs*51, 32, int(mi_duration*sfreq)])
    classes = np.zeros(num_runs * 51)
    valid_trails = 0
    
    onset = np.zeros([num_runs, 51])
    duration = np.zeros([num_runs, 51])
    description = np.zeros([num_runs, 51])

    #Loop to the first 4 runs.
    CheckFiles = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/*.edf')
    if not CheckFiles:
        return 
    
    for runNo in range(num_runs): 
        valid_trails_in_run = 0
        #Get .edf and .json file for following subject and run.
        EDFfile = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.edf')
        JSONfile = glob.glob(data_path + 'S_'+subjectNo[subject].zfill(3) + '/S'+ str(session) +'/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.json')
    
        #Check if EDFfile list is empty
        if not EDFfile:
          continue
    
        # We use mne.read_raw_edf to read in the .edf EEG files
        raw = read_raw_edf(str(EDFfile[0]), preload=True, verbose=False)
        
        # Opening JSON file of the current RUN.
        f = open(JSONfile[0],) 
    
        # returns JSON object as a dictionary 
        JSON = json.load(f) 
    
        #Number of Keystrokes Markers
        keyStrokes = np.min([len(JSON['Markers']), 51]) #len(JSON['Markers']), to avoid extra markers by accident
        # MarkerStart = JSON['Markers'][0]['startDatetime']
           
        #Get Start time of marker
        date_string = EDFfile[0][-21:-4]
        datetime_format = "%d.%m.%y_%H.%M.%S"
        startRecordTime = datetime.strptime(date_string, datetime_format).astimezone()
    
        currentTrialNo = 0 # 1 = fingers, 2 = Wrist, 3 = Elbow, 4 = rest
        if(runNo == 4): 
            currentTrialNo = 4
    
        ch_names = raw.info['ch_names'][4:36]
             
        # filter the data 
        raw.filter(4., 50., fir_design='firwin')  
        
        raw = raw.copy().pick_channels(ch_names = ch_names)
        raw = raw.copy().resample(sfreq = sfreq)
        fs = raw.info['sfreq']

        for trail in range(keyStrokes):
            
            # class for current trial
            if(runNo == 4 ):               # In Run 5 all trials are 'reset'
                currentTrialNo = 4
            elif (currentTrialNo == 3):    # Set the class of current trial to 1 'Fingers'
                currentTrialNo = 1   
            else:                          # In Runs 1-4, 1st trial is 1 'Fingers', 2nd trial is 2 'Wrist', and 3rd trial is 'Elbow', and repeat ('Fingers', 'Wrist', 'Elbow', ..)
                currentTrialNo = currentTrialNo + 1
                
            trailDuration = 8
            
            trailTime = parse(JSON['Markers'][trail]['startDatetime'])
            trailStart = trailTime - startRecordTime
            trailStart = trailStart.seconds 
            start = trailStart + (6 - mi_duration)
            stop = trailStart + 6

            if (trail < keyStrokes-1):
                trailDuration = parse(JSON['Markers'][trail+1]['startDatetime']) - parse(JSON['Markers'][trail]['startDatetime'])
                trailDuration =  trailDuration.seconds + (trailDuration.microseconds/1000000)
                if (trailDuration < 7.5) or (trailDuration > 8.5):
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    if (trailDuration > 14 and trailDuration < 18):
                        if (currentTrialNo == 3):   currentTrialNo = 1   
                        else:                       currentTrialNo = currentTrialNo + 1
                    continue
                
            elif (trail == keyStrokes-1):
                trailDuration = raw[0, int(trailStart*int(fs)):int((trailStart+8)*int(fs))][0].shape[1]/fs
                if (trailDuration < 7.8) :
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    continue

            MITrail = raw[:32, int(start*int(fs)):int(stop*int(fs))][0]
            if (MITrail.shape[1] != data.shape[2]):
                print('Error in Session: {} - Run: {}, Trail no: {} due to the lost of data'.format(session, (runNo+1), (trail+1)))
                return
            
            # select some specific classes
            if ((('Fingers' in classes_labels) and (currentTrialNo==1)) or 
            (('Wrist' in classes_labels) and (currentTrialNo==2)) or 
            (('Elbow' in classes_labels) and (currentTrialNo==3)) or 
            (('Rest' in classes_labels) and (currentTrialNo==4))):
                data[valid_trails] = MITrail
                classes[valid_trails] =  currentTrialNo
                
                # For Annotations
                onset[runNo, valid_trails_in_run]  = start
                duration[runNo, valid_trails_in_run] = trailDuration - (6 - mi_duration)
                description[runNo, valid_trails_in_run] = currentTrialNo
                valid_trails += 1
                valid_trails_in_run += 1
                         
    data = data[0:valid_trails, :, :]
    classes = classes[0:valid_trails]
    classes = (classes-1).astype(int)

    return data, classes, onset, duration, description


#%%
def standardize_data(X_train, X_test, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


#%%
def get_data(path, subject, dataset = 'BCI2a', classes_labels = 'all', LOSO = False, isStandard = True, isShuffle = True, selected_classes=None):
    # Custom: retain selected original event codes and remap to contiguous indices.
    # The order also determines one-hot columns and confusion-matrix labels.
    custom_label_map = None
    if dataset == 'Custom':
        selected_classes = list((1, 2, 3, 4) if selected_classes is None else selected_classes)
        if (not 2 <= len(selected_classes) <= 4
                or len(set(selected_classes)) != len(selected_classes)
                or any(label not in (1, 2, 3, 4) for label in selected_classes)):
            raise ValueError('selected_classes must contain 2-4 unique event codes from 1, 2, 3, 4')
        custom_label_map = {label: index for index, label in enumerate(selected_classes)}
    
    # Load and split the dataset into training and testing 
    if LOSO:
        """ Loading and Dividing of the dataset based on the 
        'Leave One Subject Out' (LOSO) evaluation approach. """ 
        if dataset == 'Custom':
            X_train, y_train, X_test, y_test = load_custom_loso(
                path, subject, label_map=custom_label_map)
        else:
            X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        """ Loading and Dividing of the data set based on the subject-specific 
        (subject-dependent) approach.
        In this approach, we used the same training and testing data as the original
        competition, i.e., for BCI Competition IV-2a, 288 x 9 trials in session 1 
        for training, and 288 x 9 trials in session 2 for testing.  
        """
        if (dataset == 'BCI2a'):
            path = path + 's{:}/'.format(subject+1)
            X_train, y_train = load_BCI2a_data(path, subject+1, True)
            X_test, y_test = load_BCI2a_data(path, subject+1, False)
        elif (dataset == 'CS2R'):
            X_train, y_train, _, _, _ = load_CS2R_data_v2(path, subject, True, classes_labels)
            X_test, y_test, _, _, _ = load_CS2R_data_v2(path, subject, False, classes_labels)
        elif dataset == 'Custom':
            # The training scripts pass zero-based subject indices, while the
            # dataset directories are numbered from S01.
            X_train, y_train, X_test, y_test = load_custom_subject_split(
                path, subject + 1, label_map=custom_label_map
            )
        # elif (dataset == 'HGD'):
        #     X_train, y_train = load_HGD_data(path, subject+1, True)
        #     X_test, y_test = load_HGD_data(path, subject+1, False)
        else:
            raise Exception("'{}' dataset is not supported yet!".format(dataset))

    if dataset == 'Custom':
        expected = set(range(len(selected_classes)))
        for split_name, split_labels in [('train', y_train), ('test', y_test)]:
            if set(np.unique(split_labels)) != expected:
                raise ValueError(f'Custom subject {subject + 1} {split_name}: selected classes are missing')
    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    n_classes = len(selected_classes) if dataset == 'Custom' else None
    y_train_onehot = to_categorical(y_train, num_classes=n_classes)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test, num_classes=n_classes)
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

