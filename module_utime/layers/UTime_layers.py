import time
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.5, verbose=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.verbose = verbose
        self.padding = (
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1
        ) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)
    
    def forward(self, x):
        # x: [B, C, L]
        out = self.layers(x)
        if self.verbose:
            print(f"[ConvBlock] Input shape: {x.shape} -> Output shape: {out.shape}")
        return out   # out: [B, out_channels, L]


class Encoder(nn.Module):
    def __init__(
        self,
        filters=[16, 32, 64, 128],
        in_channels=5,
        maxpool_kernels=[10, 8, 6, 4],
        kernel_size=5,
        dilation=2,
        dropout=0.5,
        verbose=False
    ):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.verbose = verbose
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(                  # 扩展时序通道
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                        dropout=self.dropout,
                        verbose=self.verbose
                    ), 
                    ConvBlock(                  # 提取特征
                        in_channels=self.filters[k],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                        dropout=self.dropout,
                        verbose=self.verbose
                    ),
                )
                for k in range(self.depth)
            ]
        )

        self.maxpools = nn.ModuleList(          # 降采样，缩短时间长度
            [nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)]
        )

        self.bottom = nn.Sequential(            # 提取最终编码的高层语义特征
            ConvBlock(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                verbose=self.verbose,
            ),
            ConvBlock(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                verbose=self.verbose,
            ),
        )

    def forward(self, x):
        # x: [B, C, L]
        shortcuts = []
        for i, (layer, maxpool) in enumerate(zip(self.blocks, self.maxpools)):
            z = layer(x)             # ConvBlock x2 → shape: [B, filters[i], L]
            if self.verbose:
                print(f"[Encoder] After block {i}: z shape = {z.shape}")
            shortcuts.append(z)      # Save for skip connection
            x = maxpool(z)           # Downsample → shape: [B, filters[i], L']
            if self.verbose:
                print(f"[Encoder] After maxpool {i}: x shape = {x.shape}")

        # Bottom part
        encoded = self.bottom(x)   # shape: [B, filters[-1]*2, L"]
        if self.verbose:
            print(f"[Encoder] After bottom: encoded shape = {encoded.shape}")

        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(
        self,
        filters=[128, 64, 32, 16],
        upsample_kernels=[4, 6, 8, 10],
        in_channels=256,
        out_channels=5,
        kernel_size=5,
        dropout=0.5,
        verbose=False,
    ):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.dropout = dropout
        self.verbose = verbose
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=self.upsample_kernels[k]),
                    ConvBlock(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dropout=self.dropout,
                        verbose=self.verbose,
                    ),
                )
                for k in range(self.depth)
            ]
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dropout=self.dropout,
                        verbose=self.verbose,
                    ),
                    ConvBlock(
                        in_channels=self.filters[k],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dropout=self.dropout,
                        verbose=self.verbose,
                    ),
                )
                for k in range(self.depth)
            ]
        )

    def forward(self, z, shortcuts):
        for i, (upsample, block, shortcut) in enumerate(zip(
            self.upsamples, self.blocks, shortcuts[::-1]
        )):
            z = upsample(z)
            if self.verbose:
                print(f"[Decoder] After upsample {i}: z shape = {z.shape}, shortcut shape = {shortcut.shape}")

            padding = shortcut.shape[2] - z.shape[2]
            z = F.pad(z, (0, padding, 0, 0))

            z = torch.cat([shortcut, z], dim=1)
            if self.verbose:
                print(f"[Decoder] After concat {i}: z shape = {z.shape}")
            z = block(z)
            if self.verbose:
                print(f"[Decoder] After block {i}: z shape = {z.shape}")

        return z