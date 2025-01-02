import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor
import numpy as np


class two_channel_LigoBinaryData(Dataset):
    def __init__(self, ds, device, encoder):
        self.ds = ds
        self.device = device
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{encoder}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        h1_audio, l1_audio, label, snr = self.ds[idx]['h1_timeseries'], self.ds[idx]['l1_timeseries'], self.ds[idx]['labels'], self.ds[idx]['injection_snr']
        
        h1_inputs = self.feature_extractor(h1_audio, sampling_rate=16000, return_tensors="pt")
        l1_inputs = self.feature_extractor(l1_audio, sampling_rate=16000, return_tensors="pt")
        
        h1_input_features = h1_inputs.input_features.squeeze(0)
        l1_input_features = l1_inputs.input_features.squeeze(0)

        return h1_input_features, l1_input_features, label, snr
    
class one_channel_LigoBinaryData(Dataset):
    def __init__(self, ds, device, encoder):
        self.ds = ds
        self.device = device
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{encoder}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        l1_audio, label, snr = self.ds[idx]['l1_timeseries'], self.ds[idx]['labels'], self.ds[idx]['injection_snr']
        
        l1_inputs = self.feature_extractor(l1_audio, sampling_rate=16000, return_tensors="pt")
        label = torch.tensor(label).float()
#        label = torch.tensor([1, 0]) if label == 0 else torch.tensor([0, 1])
        
        l1_input_features = l1_inputs.input_features.squeeze(0)

        return l1_input_features, label, snr
    

