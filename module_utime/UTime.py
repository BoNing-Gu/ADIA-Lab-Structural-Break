import time
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from module_utime.layers.UTime_layers import Encoder, Decoder

class Model(nn.Module):
    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.verbose = config.verbose
        self.normalizer = nn.BatchNorm1d(config.in_channels)

        self.encoder = Encoder(
            filters=config.filters,
            in_channels=config.in_channels,
            maxpool_kernels=config.maxpool_kernels,
            kernel_size=config.kernel_size,
            dilation=config.dilation,
            dropout=config.dropout,
            verbose=config.verbose,
        )
        self.decoder = Decoder(
            filters=config.filters[::-1],
            upsample_kernels=config.maxpool_kernels[::-1],
            in_channels=config.filters[-1] * 2,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            verbose=config.verbose,
        )

        self.final_conv = nn.Conv1d(
            in_channels=config.filters[0],
            out_channels=2,
            kernel_size=1,
            bias=True,
        )
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Reshape
        if self.verbose:
            print(f"[Input] x shape: {x.shape}") 
        x = x.permute(0, 2, 1)          # [B, C, L]
        if self.verbose:
            print(f"[Permuted] x shape: {x.shape}")     # [B, C, L]
        
        # Normalize inputs
        z = self.normalizer(x)
        if self.verbose:
            print(f"[After BatchNorm] z shape: {z.shape}")

        # Run through encoder
        z, shortcuts = self.encoder(z)
        if self.verbose:
            print(f"[After Encoder] z shape: {z.shape}")

        # Run through decoder
        z = self.decoder(z, shortcuts)
        if self.verbose:
            print(f"[After Decoder] z shape: {z.shape}")

        # Run through 1x1 conv to collapse channels
        z = self.final_conv(z)
        if self.verbose:
            print(f"[After Final Conv] z shape: {z.shape}")

        # Run softmax
        z = self.softmax(z)
        if self.verbose:
            print(f"[After Softmax] z shape: {z.shape}")

        # Pad output to match input length
        if z.shape[-1] != x.shape[-1]:
            diff = x.shape[-1] - z.shape[-1]
            if diff > 0:
                z = F.pad(z, (0, diff))  # pad on the right
            elif diff < 0:
                z = z[..., :x.shape[-1]]  # trim on the right
        if self.verbose:
            print(f"[Adjusted] z padded/truncated to match input z shape: {z.shape}")

        return z

