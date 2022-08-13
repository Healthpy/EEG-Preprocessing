# EEG-Preprocessing
Implementation of raw eeg preprocessing pipeline using MNE-Python with autoreject

The pipeline is implemented in two main steps. 
First is structuring and alignment of the raw eeg data. It involves the concatenation of unique subjects data, renaming the files to commonly used names, dropping of non-relevant channels, assigning channel types to electrode signals and setting of individual subject montages.
The second stage processes these structed data to clean data. Filtering to remove undesired frequency component of the signal, division into trials (epochs) of 30 seconds, Autoreject algorithm, Independent component analyses implementation to remove prevalent artifacts scoring more than 0.5 and finally implementaion of Autoreject again to have a clean eeg data.
