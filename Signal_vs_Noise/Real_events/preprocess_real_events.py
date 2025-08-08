import os
import numpy as np
import h5py
import datasets
from multiprocessing import Pool
from scipy.signal import resample
import time
import argparse
import logging
from transformers import WhisperFeatureExtractor

def extract_segments(data, window_size=2048, step_size=204):
    segments = []
    for start in range(0, len(data) - window_size + 1, step_size):
        segment = data[start:start + window_size]
        segments.append(segment)
    return segments

def resample_timeseries(data):
    original_sampling_rate = 2048
    target_sampling_rate = 16000
    target_length = len(data) * target_sampling_rate // original_sampling_rate
    return resample(data, target_length)

def process_sample(sample):
    return resample_timeseries(sample)

def load_data_from_path(path, keys, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Loading datasets.")

    with h5py.File(path, 'r') as f:
        h1_strain, l1_strain = [], []
        
        for key in keys:
            h1_strain.append(f[key]['h1_strain'][()])
            l1_strain.append(f[key]['l1_strain'][()])
                
        h1_strain, l1_strain = np.array(h1_strain), np.array(l1_strain)
        
    logging.info("Extracting segments from datasets")
    
    start_time = time.time()
    
    h1_segments, l1_segments = [], []
    for h1_sample, l1_sample in zip(h1_strain, l1_strain):
        h1_segments.extend(extract_segments(h1_sample))
        l1_segments.extend(extract_segments(l1_sample))

    # Resample segments
    with Pool(processes=4) as pool:
        resample_h1 = np.array(pool.map(process_sample, h1_segments), dtype=np.float32)
        resample_l1 = np.array(pool.map(process_sample, l1_segments), dtype=np.float32)

    # Save each individual segment as a separate dataset in respective folders
    for i, key in enumerate(keys):
        event_output_dir = os.path.join(output_dir, key)
        os.makedirs(event_output_dir, exist_ok=True)
        
        # Separate directories for h1_segments and l1_segments
        h1_output_dir = os.path.join(event_output_dir, "h1_segments")
        l1_output_dir = os.path.join(event_output_dir, "l1_segments")
        os.makedirs(h1_output_dir, exist_ok=True)
        os.makedirs(l1_output_dir, exist_ok=True)

        # Determine segment indices for this event
        h1_event_segments = resample_h1[i * len(h1_segments) // len(keys):(i + 1) * len(h1_segments) // len(keys)]
        l1_event_segments = resample_l1[i * len(l1_segments) // len(keys):(i + 1) * len(l1_segments) // len(keys)]

        # Save h1 and l1 segments separately
        for j, (h1_segment, l1_segment) in enumerate(zip(h1_event_segments, l1_event_segments)):
            # Save h1_segment in h1_segments directory
            h1_segment_data = {'h1_timeseries': [h1_segment]}
            h1_segment_ds = datasets.Dataset.from_dict(h1_segment_data)
            h1_segment_ds.save_to_disk(os.path.join(h1_output_dir, f"segment_{j}"))

            # Save l1_segment in l1_segments directory
            l1_segment_data = {'l1_timeseries': [l1_segment]}
            l1_segment_ds = datasets.Dataset.from_dict(l1_segment_data)
            l1_segment_ds.save_to_disk(os.path.join(l1_output_dir, f"segment_{j}"))

    end_time = time.time()
    print(f"Time to resample and save: {end_time - start_time} seconds")
    
    return

def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets.')
    parser.add_argument('--timeseries_data_test', type=str, required=True, help='Path to the HDF file containing testing timeseries data')
    parser.add_argument('--event_name', type=str, default=None, help='Name of the event')
    parser.add_argument('--encoder', type=str, default='tiny', help='Whisper encoder size (tiny, base, small, medium, large)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the resampled segments')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    print("Loading datasets...")
    logging.info("Loading datasets...")

    f1 = h5py.File(args.timeseries_data_test, 'r')
    
    if args.event_name is None:
        keys = [key for key in f1.keys()]
    else:
        keys = args.event_name

    load_data_from_path(args.timeseries_data_test, keys, args.output_dir)
    
    print("Preprocessing datasets complete.")
    logging.info("Preprocessing datasets complete.")

if __name__ == '__main__':
    main()
