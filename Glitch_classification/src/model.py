import torch
import torch.nn as nn

class one_channel_ligo_binary_classifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder  # Pretrained Whisper encoder
        
        # Classifier with Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # 0.3
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # 0.3
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # 0.3
            nn.Linear(128, num_classes)
        )

    def forward(self, mel_tensor):
        """
        Forward pass for the one-channel classifier.

        Args:
            mel_tensor (Tensor): Input mel spectrogram of shape (batch_size, num_features, time_steps).

        Returns:
            Tensor: Logits of shape (batch_size, num_classes).
        """
        # Encoder processes the mel spectrogram and produces the last hidden state
        encoder_outputs = self.encoder(mel_tensor)
        hidden_state = encoder_outputs.last_hidden_state[:, -1, :]  # Use the last time step
        
        # Pass through the classifier to produce logits
        logits = self.classifier(hidden_state)
        return logits
