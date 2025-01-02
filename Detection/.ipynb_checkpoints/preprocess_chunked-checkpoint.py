import datasets
from datasets import Dataset
import h5py
import numpy as np
from scipy.signal import resample
from multiprocessing import Pool
import time
import os

# Function to resample a single timeseries sample from 2048 Hz to 16000 Hz.
def resample_timeseries(data):
    original_sampling_rate = 2048
    target_sampling_rate = 16000
    target_length = len(data) * target_sampling_rate // original_sampling_rate
    return resample(data, target_length)

# Wrapper function to resample timeseries data using multiprocessing.
def process_sample(sample):
    return resample_timeseries(sample)

os.chdir('/workspace/ligo_data/data_ts/') # Change this to the path of the data folder
print(os.getcwd())

# Load and process the data from the HDF5 file
file_name = 'Whisper_train_mass-8to100.hdf'  # Update with the actual file path
with h5py.File(file_name, 'r') as f:
    num_samples = f['injection_samples']['l1_strain'].shape[0]
    print(f"Number of samples in the file: {num_samples}")
    i = int(num_samples)  # Use all samples

    # Load timeseries data
    h_1_injection_samples = f['injection_samples']['h1_strain'][:i]
    l_1_injection_samples = f['injection_samples']['l1_strain'][:i]
    h1_noise_samples = f['noise_samples']['h1_strain'][:i]
    l1_noise_samples = f['noise_samples']['l1_strain'][:i]
    
    # Load parameters for injections
    parameters_injection = {key: f['injection_parameters'][key][:i] for key in f['injection_parameters'].keys()}
    parameter_shapes = {key: value.shape[1:] for key, value in parameters_injection.items()}
    noise_length = len(l1_noise_samples)
    parameters_noise = {key: np.zeros((noise_length,) + shape) for key, shape in parameter_shapes.items()}
    parameters_combined = {key: np.concatenate([parameters_injection[key], parameters_noise[key]], axis=0).astype(np.float32) for key in parameters_injection.keys()}

# Resample timeseries data with multiprocessing
start_time = time.time()
with Pool(processes=16) as pool:
    resample_h1 = np.array(pool.map(process_sample, h_1_injection_samples), dtype=np.float32)
    resample_l1 = np.array(pool.map(process_sample, l_1_injection_samples), dtype=np.float32)
    resample_h1_noise = np.array(pool.map(process_sample, h1_noise_samples), dtype=np.float32)
    resample_l1_noise = np.array(pool.map(process_sample, l1_noise_samples), dtype=np.float32)

# Concatenate labels and timeseries
injection_labels = np.ones(len(resample_l1))
noise_labels = np.zeros(len(resample_l1_noise))
labels = np.concatenate([injection_labels, noise_labels])
h1_time_series = np.concatenate([resample_h1, resample_h1_noise])
l1_time_series = np.concatenate([resample_l1, resample_l1_noise])
data = {'h1_timeseries': h1_time_series, 'l1_timeseries': l1_time_series, 'labels': labels}
data.update(parameters_combined)

end_time = time.time()
print(f"Time to resample: {end_time - start_time} seconds")

# Function to save the dataset in chunks
def save_dataset_in_chunks(data, chunk_size, file_name):
    num_chunks = len(data['labels']) // chunk_size + int(len(data['labels']) % chunk_size != 0)
    os.makedirs(file_name, exist_ok=True)  # Create a directory to store the chunks
    
    for i in range(num_chunks):
        # Select data for the current chunk
        chunk_data = {key: value[i * chunk_size:(i + 1) * chunk_size] for key, value in data.items()}
        
        # Create a dataset from the chunk and save it
        chunk_ds = Dataset.from_dict(chunk_data)
        chunk_ds.save_to_disk(os.path.join(file_name, f"{file_name}_chunk_{i}"))
        print(f"Saved chunk {i+1}/{num_chunks}")

# Define the chunk size (adjust based on available memory)
chunk_size = 10000  # Adjust as needed
save_dataset_in_chunks(data, chunk_size, file_name.split('.')[0] + "_resampled")
