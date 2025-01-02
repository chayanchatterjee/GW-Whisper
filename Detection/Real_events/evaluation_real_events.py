import os
import numpy as np
import torch
import h5py
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperModel
from peft import LoraConfig, PeftModel, PeftConfig
from transformers import WhisperFeatureExtractor
from tqdm import tqdm
import glob
import argparse
from src.model import two_channel_ligo_binary_classifier, TwoChannelLIGOBinaryClassifierCNN

def load_models(lora_weights_path, dense_layers_path, num_classes=1):
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper_model = whisper_model.encoder
    peft_model = PeftModel.from_pretrained(whisper_model, lora_weights_path)
    model = two_channel_ligo_binary_classifier(encoder=peft_model, num_classes=num_classes)
    # model = TwoChannelLIGOBinaryClassifierCNN(encoder=peft_model, num_classes=num_classes)
    model.classifier.load_state_dict(torch.load(dense_layers_path))
    return model

def save_output_to_hdf(output, event_names, output_path):
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("model_output", data=output)
        f.create_dataset("event_names", data=event_names)

def run_inference_on_segments(model, folder_H1, folder_L1, device, feature_extractor):
    segment_ds_H1 = load_from_disk(folder_H1)
    segment_ds_L1 = load_from_disk(folder_L1)
    segment_ds_H1.set_format(type='torch', columns=['h1_timeseries'])
    segment_ds_L1.set_format(type='torch', columns=['l1_timeseries'])
    
    data_loader_H1 = DataLoader(segment_ds_H1, batch_size=1, shuffle=False)
    data_loader_L1 = DataLoader(segment_ds_L1, batch_size=1, shuffle=False)
    
    model.eval()
    model = model.to(device)
    
    all_outputs = []
    with torch.no_grad():
        for batch_H1, batch_L1 in zip(data_loader_H1, data_loader_L1):
            # Convert l1_timeseries to numpy array and flatten if needed
            h1_timeseries = batch_H1['h1_timeseries'].squeeze().numpy()
            l1_timeseries = batch_L1['l1_timeseries'].squeeze().numpy()
            
            # Apply feature extraction on raw audio data
            h1_inputs = feature_extractor(h1_timeseries, sampling_rate=16000, return_tensors="pt")
            h1_input_features = h1_inputs.input_features.to(device)
            
            # Apply feature extraction on raw audio data
            l1_inputs = feature_extractor(l1_timeseries, sampling_rate=16000, return_tensors="pt")
            l1_input_features = l1_inputs.input_features.to(device)
            
            # Run inference on the model
            output = model(h1_input_features, l1_input_features)
            
            # Pass the output through a Sigmoid function
            output = torch.sigmoid(output)
            
            # Remove unnecessary dimensions (convert to shape (1,) or scalar)
            output = output.squeeze().cpu().numpy()  # Squeeze to remove dimensions like (1, 1)
            all_outputs.append(output)

    all_outputs = np.array(all_outputs)  # Final shape should be (31,) or (31, 1)
    return all_outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on LIGO segments")
    parser.add_argument('--results_path', type=str, default='/workspace/ligo_results/Mass-8to100/finetune_q_k_v_o/', help="Path to results directory")
    parser.add_argument('--ds_base_path', type=str, default='/workspace/ligo_data/real_events/GWTC-3/Real_events_detection/*', help="Base path for dataset")
    parser.add_argument('--output_path', type=str, default='/workspace/ligo_data/real_events/GWTC-3/results_1_sec/model_outputs_2_detectors_finetune_q_k_v_o_mass-8to100_best.hdf', help="Path to save HDF5 output")
    parser.add_argument('--encoder', type=str, default='tiny', help="Whisper model size")
    parser.add_argument('--lora_weights_file', type=str, default='best_lora_weights_8_32', help="LoRA weights file name")
    parser.add_argument('--dense_layers_file', type=str, default='best_dense_layers_8_32.pth', help="Dense layers file name")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_models(os.path.join(args.results_path, args.lora_weights_file), os.path.join(args.results_path, args.dense_layers_file))

    all_folders = sorted(glob.glob(args.ds_base_path))
    folder_names = [os.path.basename(folder).split('-')[0] for folder in all_folders]
    
    segment_folders_H1 = sorted(glob.glob(f'{args.ds_base_path}/h1_segments/segment_0'))
    segment_folders_L1 = sorted(glob.glob(f'{args.ds_base_path}/l1_segments/segment_0'))
    
    outputs = []
    
    # Initialize Whisper feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{args.encoder}")
    
    for folder_H1, folder_L1 in tqdm(zip(segment_folders_H1, segment_folders_L1), desc="Processing segments"):
        # Run inference on the segment
        outputs.append(run_inference_on_segments(model, folder_H1, folder_L1, device, feature_extractor))

    save_output_to_hdf(np.array(outputs), folder_names, args.output_path)
    print(f'Saved model output to {args.output_path}')
    print("Inference and saving complete for all segments.")
