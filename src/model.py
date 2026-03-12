import torch
from torch import nn

from src.melbanks import LogMelFilterBanks


class SpeechCommandCNN(nn.Module):
    def __init__(self, n_mels=80, groups=1):
        super().__init__()

        self.features = LogMelFilterBanks(n_mels=n_mels)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=n_mels, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2, groups=groups),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        # x: (batch, time)
        x = self.features(x)          # (batch, n_mels, frames)
        x = self.encoder(x)           # (batch, 128, 1)
        x = x.squeeze(-1)             # (batch, 128)
        x = self.classifier(x)        # (batch, 2)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)