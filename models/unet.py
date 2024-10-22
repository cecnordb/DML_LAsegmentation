import torch.nn as nn
import torch
from typing import List
from enum import Enum

class NormalizationType(Enum):
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"
    NONE = "none"

class MultiConv3d(nn.Module):
    def __init__(self, channels:List = None, normalization:NormalizationType = NormalizationType.BATCH_NORM) -> None:
        super(MultiConv3d, self).__init__()
        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))
            if normalization == NormalizationType.BATCH_NORM:
                layers.append(nn.BatchNorm3d(out_channels))
            elif normalization == NormalizationType.GROUP_NORM:
                num_groups = min(16, out_channels // 4)
                layers.append(nn.GroupNorm(num_groups, out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=None, normalization:NormalizationType = NormalizationType.BATCH_NORM) -> None:
        super(UNet3D, self).__init__()
        if features == None: 
            features = [32, 64, 128, 256]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(MultiConv3d([in_channels, feature, feature], normalization))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(MultiConv3d([feature*2, feature, feature], normalization))

        # Bottleneck
        self.bottleneck = MultiConv3d([features[-1], features[-1]*2, features[-1]*2], normalization)
        
        # Output
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []


        # Encoding
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoding
        skip_connections = reversed(skip_connections)
        for skip_connection, trans_conv, multiconv in zip(skip_connections, self.decoder[0::2], self.decoder[1::2]):
            x = trans_conv(x)
            if x.shape != skip_connection.shape:
                x = self._pad_if_needed(x, skip_connection)
            
            # Concatenate along channels
            x = torch.cat((skip_connection, x), dim=1)
            x = multiconv(x)

        return self.final_conv(x)
    
    def _pad_if_needed(self, x, skip_connection):
        """Handle padding issues in case feature maps have different shapes"""
        diff_depth = skip_connection.size(2) - x.size(2)
        diff_height = skip_connection.size(3) - x.size(3)
        diff_width = skip_connection.size(4) - x.size(4)
        x = nn.functional.pad(x, [diff_width // 2, diff_width - diff_width // 2,
                                  diff_height // 2, diff_height - diff_height // 2,
                                  diff_depth // 2, diff_depth - diff_depth // 2])
        return x
