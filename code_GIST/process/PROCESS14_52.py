'''''

HGD is processed for GIST
'''''
import logging
import os.path
from collections import OrderedDict
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from os.path import join as pjoin
import h5py
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, lfilter
from scipy import signal
from braindecode.datautil.signalproc import highpass_cnt

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b,a,data,axis=2)
    return y


def load_bbci_data(filename, low_cut_hz,debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']

    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()
    print(cnt.info["sfreq"])

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2])])

    # pick sensors
    C_sensors = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3',
                 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors,ordered=True)

    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=4, axis=1),
        cnt)

    # select time period
    ival = [0, 3000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X
    dataset.y = dataset.y
    return dataset


if __name__ == '__main__':
    # generate data path in device
    out = './DATA14_52'
    low_cut_hz = 0.5
    # source data path in device
    data_folder = '../../datasets/HGD'
    with h5py.File(pjoin(out, 'ku_mi_smt.h5'), 'w') as f:
        for num in tqdm(range(1, 15)):
            subject_id = num
            train_filename = os.path.join(
                data_folder, 'train/{:d}.mat'.format(subject_id))
            test_filename = os.path.join(
                data_folder, 'test/{:d}.mat'.format(subject_id))
            dataset1 = load_bbci_data(train_filename, low_cut_hz,  debug=False)
            dataset2 = load_bbci_data(test_filename, low_cut_hz,  debug=False)
            X1 = dataset1.X
            print(X1.shape)
            y1 = dataset1.y
            X2 = dataset2.X
            y2 = dataset2.y
            NUMS1 = X1.shape[0]
            NUMS2 = X2.shape[0]
            # CAR
            for i in range(NUMS1):
                for j in range(1500):
                    X1[i, :, j] = X1[i, :, j] - np.mean(X1[i, :, j])
            for i in range(NUMS2):
                for j in range(1500):
                    X2[i, :, j] = X2[i, :, j] - np.mean(X2[i, :, j])
            # bandpass filter
            X1 = butter_bandpass_filter(X1, 8, 30, 500)
            X2 = butter_bandpass_filter(X2, 8, 30, 500)
            X = np.concatenate((X1, X2), axis=0)
            Y = np.concatenate((y1, y2), axis=0)
            X = X.astype(np.float32)
            Y = Y.astype(np.int64)
            f.create_dataset('s' + str(num) + '/X', data=X)
            f.create_dataset('s' + str(num) + '/Y', data=Y)
