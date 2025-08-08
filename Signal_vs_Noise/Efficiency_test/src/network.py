import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel_tensor_0, mel_tensor_1):
        flat_h1 = torch.flatten(mel_tensor_0, start_dim=1)
        flat_l1 = torch.flatten(mel_tensor_1, start_dim=1)

        flat_inputs = torch.cat((flat_h1, flat_l1), dim=1)
        logits = self.classifier(flat_inputs)
        return logits

class ligo_binary_classifier(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.ln_post.normalized_shape[0], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel_tensor):
        outputs = self.encoder(mel_tensor)
        logits = self.classifier(outputs)
        return logits

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
    def __init__(self, encoder, num_classes=2):
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
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, mel_tensor_0):
        output_h1 = self.encoder(mel_tensor_0).last_hidden_state[:, -1, :]
        logits = self.classifier(output_h1)
        return logits

class LoRA_layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        x = self.alpha / self.rank * (x @ self.A @ self.B)
        return x

class LoRa_linear(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRA_layer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)