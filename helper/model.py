import torch
from torch import nn


class FCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        # Add padding to maintain same size (like TensorFlow's padding='same')
        padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FCN(nn.Module):
    def __init__(self, in_channels, num_classes, layers=(128, 256, 128), kss=(7, 5, 3)):
        super().__init__()

        self.layer_1 = FCNBlock(in_channels, layers[0], kss[0])
        self.layer_2 = FCNBlock(layers[0], layers[1], kss[1])
        self.layer_3 = FCNBlock(layers[1], layers[2], kss[2])

        # Global Average Pooling - averages across the entire time dimension
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(layers[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        # Global Average Pooling reduces time dimension to 1
        x = self.gap(x)  # Shape: [batch_size, features, 1]
        x = self.flatten(x)  # Shape: [batch_size, features]
        x = self.fc(x)
        return x