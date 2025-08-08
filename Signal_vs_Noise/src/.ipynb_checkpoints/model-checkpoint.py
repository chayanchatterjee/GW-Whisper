import torch
import torch.nn as nn

class two_channel_ligo_binary_classifier(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
#        self.encoder = AdaLoraModel(encoder, AdaLoraConfig(r=8, alpha=32, target_modules=['q', 'v'], dropout=0.01))
        self.classifier = nn.Sequential(
#            nn.Linear(self.encoder.ln_post.normalized_shape[0] * 2, 1024),
            nn.Linear(self.encoder.config.d_model * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
#            nn.Linear(256, 128),
#            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel_tensor_0, mel_tensor_1):
#        output_h1 = self.encoder(mel_tensor_0)[:, -1, :]
#        output_l1 = self.encoder(mel_tensor_1)[:, -1, :]
        output_h1 = self.encoder(mel_tensor_0).last_hidden_state[:, -1, :]
        output_l1 = self.encoder(mel_tensor_1).last_hidden_state[:, -1, :]
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits
    
class one_channel_ligo_binary_classifier(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
#            nn.Linear(self.encoder.ln_post.normalized_shape[0] * 2, 1024),
            nn.Linear(self.encoder.config.d_model, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
#            nn.Sigmoid()
        )

    def forward(self, mel_tensor_0):
        output_h1 = self.encoder(mel_tensor_0).last_hidden_state[:, -1, :]
        logits = self.classifier(output_h1)
        return logits
    
import torch
import torch.nn as nn

class TwoChannelLIGOBinaryClassifierCNN(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
        
        # CNN-based decoder
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pool to a single value per feature map
            nn.Flatten(),  # Flatten the pooled output for classification
            nn.Linear(256, num_classes)  # Final classification layer
        )

    def forward(self, mel_tensor_0, mel_tensor_1):
        # Extract features from Whisper encoder
        output_h1 = self.encoder(mel_tensor_0).last_hidden_state[:, -1, :]  # Shape: (batch_size, d_model)
        output_l1 = self.encoder(mel_tensor_1).last_hidden_state[:, -1, :]  # Shape: (batch_size, d_model)
        
        # Concatenate the outputs along the feature dimension
        combined_features = torch.stack((output_h1, output_l1), dim=1)  # Shape: (batch_size, 2, d_model)
        
        # Pass through the CNN-based decoder
        logits = self.classifier(combined_features)  # Shape: (batch_size, num_classes)
        return logits
