import argparse
import h5py
import numpy as np
import logging
import os
import concurrent.futures

from transformers import WhisperFeatureExtractor, AdamW

import whisper
from whisper.audio import pad_or_trim
from whisper.model import Whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

from network import BaselineModel, ligo_binary_classifier, one_channel_ligo_binary_classifier, LoRA_layer, LoRa_linear

from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperModel
from peft import LoraConfig, PeftModel, PeftConfig

import torch
nn = torch.nn

from scipy.signal import resample

device = 'cuda'
dtype = torch.float32

def load_model(lora_weights_path, dense_layers_path):
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper_model = whisper_model.encoder

    peft_model = PeftModel.from_pretrained(whisper_model, lora_weights_path)

    # Create the full model with loaded LoRA and dense layers
    model = one_channel_ligo_binary_classifier(encoder=peft_model)
    
    # Load the Dense layer weights
    model.classifier.load_state_dict(torch.load(dense_layers_path, map_location=device))

    return model

def resample_timeseries(data):
    original_sampling_rate = 2048
    target_sampling_rate = 16000
    target_length = len(data) * target_sampling_rate // original_sampling_rate
    return resample(data, target_length)

def extract_whisper_features_batch(batch_data, feature_extractor, device):
    if isinstance(batch_data[0], torch.Tensor):
        batch_data = [data.numpy() for data in batch_data]
    batch_data = [data.squeeze() for data in batch_data]
    inputs = feature_extractor(batch_data, sampling_rate=16000, return_tensors="pt").to(device)
    return inputs.input_features.cpu().numpy()

def get_already_processed_files(log_file_path):
    processed_files = []
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if "Successfully saved output to" in line:
                processed_files.append(line.split("Successfully saved output to")[-1].strip())
    return processed_files

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', required=True, type=str, help="The path to a directory from which the files will be read.")
    parser.add_argument('--output-dir', required=True, type=str, help="The path to a directory in which the output files will be stored.")
    parser.add_argument('--log-file', required=True, type=str, help="Path to the log file containing successfully processed files.")
    parser.add_argument('--verbose', action='store_true', help="Print status updates while loading.")
    parser.add_argument('--debug', action='store_true', help="Print debugging info")
    parser.add_argument('--batch-size', type=int, default=0, required=False, help="Batch size, 0 means entire data files. Default: 0")
    parser.add_argument('--device', type=str, default=device, required=False, help="Device to use for calculation.")
    parser.add_argument('--remove-softmax', action='store_true', help="Replace the final softmax layer by a 'mutual subtraction layer'.")

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARN
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    logging.info(f'Initializing network on device {args.device}')
    lora_weights_fname = '/workspace/LIGO/ml-training-strategies/Pytorch/state_dicts/best_lora_weights_run_0000.pt'
    dense_layers_fname = '/workspace/LIGO/ml-training-strategies/Pytorch/state_dicts/best_dense_layers_run_0000.pth'
        
    Network = load_model(lora_weights_fname, dense_layers_fname).to(device)
    Network.eval()

    if args.remove_softmax:
        logging.info('Removing the softmax layer')
        new_layer = torch.nn.Linear(2, 2, bias=False)
        new_layer._parameters['weight'] = torch.nn.Parameter(torch.Tensor([[1., -1.], [-1., 1.]]), requires_grad=False)
        new_layer.to(device=device)
        layers = list(Network.classifier.children())
        if isinstance(layers[-1], torch.nn.Softmax):
            layers[-1] = new_layer
            Network.classifier = torch.nn.Sequential(*layers)
        else:
            raise ValueError("The last layer of the classifier is not a Softmax layer.")

    logging.info(f'Network initialized, starting to load files')
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    file_list = os.listdir(args.input_dir)
    logging.info(f'Found {len(file_list)} files in the input directory')

    # Get the list of already processed files
    processed_files = get_already_processed_files(args.log_file)
    logging.info(f"Skipping {len(processed_files)} files that are already processed.")

    for fn in file_list:
        fout = os.path.join(args.output_dir, fn)
        if fout in processed_files:
            logging.info(f"Skipping already processed file: {fn}")
            continue

        fin = os.path.join(args.input_dir, fn)
        logging.debug(f'Trying to load file from {fin}')
        
        with h5py.File(fin, 'r') as fp:
            in_data = fp['H1/data'][:]

        logging.info('Starting resampling')
        mel_data_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            resampled_data_list = list(executor.map(resample_timeseries, in_data))

        logging.info('Starting Whisper feature extraction in parallel')
        batch_size = 64  # Reduce batch size to fit within memory constraints
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for i in range(0, len(resampled_data_list), batch_size):
                batch_data = resampled_data_list[i:i + batch_size]
                mel_data = extract_whisper_features_batch(batch_data, feature_extractor, 'cpu')  # Extract on CPU
                mel_data_list.append(mel_data)

        mel_data = np.vstack(mel_data_list)
        
        # Process mel_data in smaller chunks
        output_chunks = []
        chunk_size = 16  # Use a small chunk size to fit in GPU memory
        for i in range(0, mel_data.shape[0], chunk_size):
            mel_chunk = torch.from_numpy(mel_data[i:i + chunk_size]).to(device=args.device, dtype=dtype)
            with torch.no_grad():
                out_chunk = Network(mel_chunk).cpu().numpy()
            output_chunks.append(out_chunk)
            del mel_chunk  # Clear GPU memory immediately
            torch.cuda.empty_cache()

        out_data = np.concatenate(output_chunks, axis=0)

        with h5py.File(fout, 'w') as fp:
            fp.create_dataset('data', data=out_data)

        logging.info(f'Successfully saved output to {fout}')
        
        del mel_data, out_data, output_chunks, mel_data_list, resampled_data_list
        torch.cuda.empty_cache()  # Clear CUDA memory after each file

    logging.info('Finished processing all files')
    return

if __name__ == "__main__":
    main()
