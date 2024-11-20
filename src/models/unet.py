# your_project/models/unet.py

import torch.nn as nn
import torch

class UNetAudio(nn.Module):
    def __init__(self):
        super(UNetAudio, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(64, 16, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv1d(32, 2, kernel_size=15, stride=1, padding=7),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)  # [B, 16, T]
        e2 = self.enc2(e1)  # [B, 32, T/2]
        e3 = self.enc3(e2)  # [B, 64, T/4]
        e4 = self.enc4(e3)  # [B, 128, T/8]

        # Decoding path with skip connections
        d1 = self.dec1(e4)  # [B, 64, T/4]
        d1 = torch.cat([d1, e3], dim=1)  # [B, 128, T/4]
        d2 = self.dec2(d1)  # [B, 32, T/2]
        d2 = torch.cat([d2, e2], dim=1)  # [B, 64, T/2]
        d3 = self.dec3(d2)  # [B, 16, T]
        d3 = torch.cat([d3, e1], dim=1)  # [B, 32, T]
        out = self.dec4(d3)  # [B, 2, T]
        return out
