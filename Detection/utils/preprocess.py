import datasets
from datasets import concatenate_datasets, DatasetDict,  Audio, load_from_disk, Dataset, Features, Value, Array2D
import pandas as pd
import os
import numpy as np
from tqdm import tqdm, tqdm_notebook
import datetime
import h5py
import time

from multiprocessing import Pool, Manager
from scipy.signal import resample, resample_poly

import matplotlib.pyplot as plt

import librosa
import librosa.display as ldisplay

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import WhisperFeatureExtractor, AdamW

import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper.model import Whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, accuracy_score


os.chdir('/workspace/ligo_data/Binary_classification_tests/') # Change this to the path of the data folder
print(os.getcwd())

file_name = 'Test_Whisper_SNR-10.hdf' # Change this to the path of the HDF5 file

def resample_timeseries(data):
    """
    Function to resample a single timeseries sample from 2048 Hz to 16000 Hz.
    """
    original_sampling_rate = 2048
    target_sampling_rate = 16000
    target_length = len(data) * target_sampling_rate // original_sampling_rate
    return resample(data, target_length)

def process_sample(sample):
    """
    Wrapper function to resample timeseries data using multiprocessing.
    """
    return resample_timeseries(sample)

# Open the HDF5 file
with h5py.File(file_name, 'r') as f:
    num_samples = f['injection_samples']['l1_strain'].shape[0]
    print(f"Number of samples in the file: {num_samples}")

    # Use all the samples
    i = int(num_samples)

    # Load timeseries data
    h_1_injection_samples = f['injection_samples']['h1_strain'][:i]
    l_1_injection_samples = f['injection_samples']['l1_strain'][:i]
    h1_noise_samples = f['noise_samples']['h1_strain'][:i]
    l1_noise_samples = f['noise_samples']['l1_strain'][:i]
    
    # Load parameters for injections
    parameters_injection = {key: f['injection_parameters'][key][:i] for key in f['injection_parameters'].keys()}

    parameter_shapes = {key: value.shape[1:] for key, value in parameters_injection.items()}  # Assuming the first dimension is the length
    
    # The length of noise samples
    noise_length = len(l1_noise_samples)
    
    # Create parameters_noise with shapes matching parameter_shapes but lengths matching the noise samples
    parameters_noise = {key: np.zeros((noise_length,) + shape) for key, shape in parameter_shapes.items()}
    
    # Combine parameters_injection and parameters_noise
    # Assuming you want to append parameters_noise after parameters_injection
    parameters_combined = {}
    for key in parameters_injection.keys():
        combined = np.concatenate([parameters_injection[key], parameters_noise[key]], axis=0)
        parameters_combined[key] = combined.astype(np.float32)
 
start_time = time.time()

# Resample timeseries data with multiprocessing
with Pool(processes=16) as pool:  # Adjust the number of processes as needed
    resample_h1 = np.array(pool.map(process_sample, h_1_injection_samples), dtype=np.float32)
    resample_l1 = np.array(pool.map(process_sample, l_1_injection_samples), dtype=np.float32)
    resample_h1_noise = np.array(pool.map(process_sample, h1_noise_samples), dtype=np.float32)
    resample_l1_noise = np.array(pool.map(process_sample, l1_noise_samples), dtype=np.float32)
    

# Prepare the dataset as before, using resampled_injection_samples and resampled_noise_samples
injection_labels = np.ones(len(resample_l1))
noise_labels = np.zeros(len(resample_l1_noise))

labels = np.concatenate([injection_labels, noise_labels])

h1_time_series = np.concatenate([resample_h1, resample_h1_noise])
l1_time_series = np.concatenate([resample_l1, resample_l1_noise])


data = {'h1_timeseries': h1_time_series, 'l1_timeseries': l1_time_series, 'labels': labels}

data.update(parameters_combined)

end_time = time.time()
print(f"Time to resample: {end_time - start_time} seconds")


sizes = set()
for value in data.values():
    sizes.add(len(value))

if len(sizes) == 1:
    print("All values have the same size.")
else:
    print("Values have different sizes.")
    for key, value in data.items():
        print(f"Size of {key}: {len(value)}")

ds = datasets.Dataset.from_dict(data)

ds.save_to_disk(f"{file_name.split('.')[0]}_resampled")
