import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import os
import numpy as np
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from autoreject import (AutoReject, get_rejection_threshold)



import mne
from mne.io import concatenate_raws, read_raw_edf
from glob import glob
data_name = []
data_time = []
raws = []


def rename_channel_to_standard(orig_name: str) -> str:
    new_name = (orig_name
                .replace('EEG ', '')
                .replace('-REF', ''))
    return new_name

path = 'Humanoid EDF raw'
folderpath = os.listdir(path)
for folder in folderpath:
    data_path = glob( f"{path}/{folder}" + '/*.edf')
    data = concatenate_raws([read_raw_edf(file, preload=True) for file in data_path])
    data.drop_channels(['RR', 'IBI', 'Bursts', 'Suppr', 'EEG A1-REF', 'EEG A2-REF'])
    data.rename_channels(mapping=rename_channel_to_standard)
    data.set_channel_types({'EOGdx':'eog', 'EOGsin':'eog', 'ECG':'ecg', 'EMG':'emg'})
    montage = mne.channels.make_standard_montage('standard_1020')
    data.set_montage(montage, on_missing='ignore')
    data_time.append(len(data.times)/1000)
    data_name.append(folder)
    raws.append(data)
    

raw_copy = raws.copy()

def eeg_prep(raw, epoch_length):

    raw_eeg_filtered = raw.filter(l_freq=0.1, h_freq=40)
    raw_epochs = mne.make_fixed_length_epochs(raw_eeg_filtered, duration=epoch_length, preload=True, reject_by_annotation=False)
    filt_epo = raw_epochs.copy().filter(l_freq=1., h_freq=None)

    # Autoreject (local) epochs to benefit ICA
    auto_reject_pre_ica   = AutoReject(random_state = 100).fit(raw_epochs)
    epochs_ar, reject_log = auto_reject_pre_ica.transform(raw_epochs, return_log = True)

    # Fit ICA on non-artifactual epochs 
    ica = mne.preprocessing.ICA(random_state = 100).fit(filt_epo[~reject_log.bad_epochs], decim=11)

    # Exclude blink artifact components
    epochs_eog    = mne.preprocessing.create_eog_epochs(raw = raw) 
    _, eog_scores = ica.find_bads_eog(epochs_eog, measure = 'zscore')
    ica.exclude   = np.argwhere(np.abs(eog_scores) > 0.5).ravel().tolist()
                
    # Apply ICA
    epochs_clean = ica.apply(raw_epochs.copy())
                
    # Autoreject (local) on blink-artifact-free epochs
    auto_reject_post_ica = AutoReject(random_state = 100).fit(epochs_clean)
    epochs_clean         = auto_reject_post_ica.transform(epochs_clean)

    return raw_eeg_filtered, raw_epochs, epochs_clean
