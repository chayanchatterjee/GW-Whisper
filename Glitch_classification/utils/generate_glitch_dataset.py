import argparse
import os
import pandas as pd
import numpy as np
import h5py
import logging
from gwpy.timeseries import TimeSeries
from astropy.utils.data import clear_download_cache
import pycbc
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to download data for a given GPS time and interferometer
def download_gw_data(gps_time, ifo):
    try:
        start_time = gps_time - 8  # 0.5 seconds before the event
        end_time = gps_time + 8    # 0.5 seconds after the event
        logging.info(f"Fetching data for IFO: {ifo}, GPS time: {gps_time}")
        data = TimeSeries.fetch_open_data(ifo, start_time, end_time, cache=True)
        return data
    except ValueError as e:
        logging.error(f"Data not available for GPS time {gps_time} and interferometer {ifo}: {e}")
        clear_download_cache()
        return None

# Function to process a single row
def process_row(row):
    try:
        gps_time = row['GPStime']
        snr = row['snr']
        ifo = row['ifo']
        logging.info(f"Processing row - GPS: {gps_time}, SNR: {snr}, IFO: {ifo}")
        
        strain_data = download_gw_data(gps_time, ifo)
        if strain_data is None:
            return None
        
        # Downsample and whiten data
        data = pycbc.types.timeseries.TimeSeries(
            strain_data.value[::2], 
            delta_t=1.0 / 2048, 
            epoch=strain_data.times[0].value
        )
        data, _ = data.whiten(
            segment_duration=4, 
            max_filter_duration=4, 
            remove_corrupted=True, 
            return_psd=True
        )
        # Apply a high-pass filter
        data = data.highpass_fir(
            frequency=30, 
            remove_corrupted=True, 
            order=512
        )
        # Crop data to 1 second
        data = data.time_slice(gps_time - 0.80, gps_time + 0.20)
        
        data_array = data.numpy()
        if not np.isnan(data_array).any():
            logging.info(f"Valid data processed for GPS time: {gps_time}")
            return data_array, snr
        else:
            logging.warning(f"Data contains NaNs for GPS time: {gps_time}")
            return None
    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return None

# Function to process a single CSV file
def process_csv_file(args):
    input_file_path, output_folder = args
    logging.info(f"Processing file: {input_file_path}")
    
    # Read the first 7000 rows of the CSV file
    # For training data
#    glitches = pd.read_csv(input_file_path).head(7000)

    # For test data
    glitches = pd.read_csv(input_file_path).iloc[7000:8000]

    
    # Use multiprocessing to process rows in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_row, [row for _, row in glitches.iterrows()])
    
    # Filter valid results
    valid_results = [(strain, snr) for result in results if result is not None for strain, snr in [result]]
    strains = [strain for strain, _ in valid_results]  # Collect all strains
    snrs = [snr for _, snr in valid_results]           # Collect all SNRs
    
    if strains:
        max_length = max(len(arr) for arr in strains)
        padded_arrays = [
            np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) 
            for arr in strains
        ]
        combined_strains = np.array(padded_arrays)
        combined_snrs = np.array(snrs)
        
        output_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + '_processed.hdf5'
        output_file_path = os.path.join(output_folder, output_file_name)
        
        with h5py.File(output_file_path, 'w') as f:
            f.create_dataset('Strain', data=combined_strains)
            f.create_dataset('SNR', data=combined_snrs)
        
        logging.info(f"Processed data saved to {output_file_path}")
    else:
        logging.warning(f"No valid data found in {input_file_path}")

# Main function to process CSV files sequentially
def process_csv_files(input_folder, output_folder):
    logging.info(f"Processing CSV files from input folder: {input_folder}")
    if not os.path.exists(output_folder):
        logging.info(f"Output folder does not exist. Creating: {output_folder}")
        os.makedirs(output_folder)
    
    # Get list of CSV files in the input folder
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not csv_files:
        logging.warning("No CSV files found in the input folder.")
        return
    
    for csv_file in csv_files:
        process_csv_file((csv_file, output_folder))

# Parse arguments and call main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gravitational wave data from CSV files.")
    parser.add_argument("--input_folder", type=str, help="Path to the input folder containing CSV files.")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder for HDF5 files.")
    args = parser.parse_args()
    
    logging.info("Script started")
    try:
        process_csv_files(args.input_folder, args.output_folder)
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
    logging.info("Script finished")
