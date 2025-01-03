import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor
import numpy as np


class one_channel_LigoBinaryData(Dataset):
    def __init__(self, ds, device, encoder):
        """
        Args:
            ds: The dataset containing strain data, labels, and SNR.
            device: The device to use ('cpu' or 'cuda').
            encoder: The Whisper encoder size (e.g., 'small', 'medium').
        """
        self.ds = ds
        self.device = device
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{encoder}")
        
        # Encode string labels into numerical indices
        self.label_encoder = LabelEncoder()
        labels = self.ds['labels']  # Extract labels as a list
        encoded_labels = self.label_encoder.fit_transform(labels)  # Encode the labels
        
        # Add the encoded labels back to the dataset as a new column
        self.ds = self.ds.map(lambda x, i: {'encoded_labels': encoded_labels[i]}, with_indices=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Fetch a single data point from the dataset.

        Args:
            idx: Index of the data point to fetch.

        Returns:
            Tuple of (l1_input_features, label, snr).
        """
        l1_audio = self.ds[idx]['data']
        label = self.ds[idx]['encoded_labels']
        snr = self.ds[idx]['SNR']
        
        # Extract Whisper features
        l1_inputs = self.feature_extractor(l1_audio, sampling_rate=16000, return_tensors="pt")
        l1_input_features = l1_inputs.input_features.squeeze(0)

        return l1_input_features, label, snr

    def decode_label(self, label_idx):
        """
        Convert a numerical label back into its original string form.

        Args:
            label_idx: Numerical index of the label.

        Returns:
            Original string label.
        """
        return self.label_encoder.inverse_transform([label_idx])[0]
