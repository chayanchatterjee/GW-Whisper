import os
import h5py
import numpy as np
from multiprocessing import Pool
from datasets import Dataset
import datasets
from scipy.signal import resample, resample_poly
import argparse

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

def extract_label_from_filename(file_name):
    """
    Extracts the label from the filename between 'gspy' and 'O3b' or 'O3a'.
    """
    if 'O3b' in file_name:
        label = file_name.split('gspy_')[1].split('_O3b')[0]
    elif 'O3a' in file_name:
        label = file_name.split('gspy_')[1].split('_O3a')[0]
    elif 'all' in file_name:
        label = file_name.split('gspy_')[1].split('_all')[0]
    else:
        raise ValueError("Filename does not contain 'O3b' or 'O3a' or 'all'.")
    return label

def process_file(file_path, label):
    """
    Process a single HDF5 file and return the processed data.
    """
    print(file_path)
    f = h5py.File(file_path, 'r')
    if 'injection_samples' in f.keys():
        num_samples = f['injection_samples']['l1_strain'].shape[0]
    elif 'Strain' in f.keys():
        num_samples = f['Strain'].shape[0]
            
    print(f"Processing file: {file_path} with {num_samples} samples")
        
    # Use all samples
    i = int(num_samples)

    if 'injection_samples' in f.keys():
        # Load timeseries data
        samples = f['injection_samples']['l1_strain'][:i]
        # Load parameters for injections
        parameters = {'SNR': f['injection_parameters']['injection_snr'][:i]}
            
    elif 'Strain' in f.keys(): 
        samples = f['Strain'][:i,0:2048]
        parameters = {'SNR': f['SNR'][:i]}

    # Resample timeseries data with multiprocessing
    with Pool(processes=16) as pool:
        resample_data = np.array(pool.map(process_sample, samples), dtype=np.float32)

    # Prepare labels
    labels = np.full(len(resample_data), label)

    # Combine time series data and parameters
    data = {'data': resample_data, 'labels': labels}
    data.update(parameters)
    
    return data

def process_folder(folder_path, output_name):
    """
    Process all HDF5 files in a folder, combine the data, and save the final dataset.
    """
    combined_data = None
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.hdf') or file_name.endswith('.hdf5'):
            file_path = os.path.join(folder_path, file_name)
            label = extract_label_from_filename(file_name)
            file_data = process_file(file_path, label)
            
            if combined_data is None:
                combined_data = file_data
            else:
                # Concatenate each key's data
                for key in combined_data.keys():
                    combined_data[key] = np.concatenate([combined_data[key], file_data[key]], axis=0)
    
    # Verify that all keys have the same size
    sizes = set()
    for value in combined_data.values():
        sizes.add(len(value))

    if len(sizes) != 1:
        print("Error: Values have different sizes.")
        for key, value in combined_data.items():
            print(f"Size of {key}: {len(value)}")
        return
    
    print("All values have the same size.")
    ds = Dataset.from_dict(combined_data)
    ds.save_to_disk(output_name)
    print(f"Dataset saved to {output_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process HDF5 files and resample timeseries data.")
    parser.add_argument('--folder_path', type=str, default='/GW-Whisper/Glitch_classification/data/generic/test_data', help="Path to the folder containing HDF5 files.")
    parser.add_argument('--output_name', type=str, default='/GW-Whisper/Glitch_classification/data/generic/test_data/generic_combined_resampled_dataset_test', help="Path to save the combined dataset.")
    args = parser.parse_args()

    process_folder(args.folder_path, args.output_name)
