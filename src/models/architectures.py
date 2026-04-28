# src/models/architectures.py

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Standard CNN baseline for EuroSAT classification.
    Used as the accuracy-focused reference model.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution:
    depthwise spatial filtering + pointwise channel mixing.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightweightCNN(nn.Module):
    """
    Lightweight CNN using depthwise separable convolutions.
    This is the deployment-efficiency reference model.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ECABlock(nn.Module):
    """
    Efficient Channel Attention.

    Unlike SE attention, ECA avoids dimensionality reduction.
    It uses a lightweight 1D convolution over channel descriptors.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)                  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)   # [B, 1, C]
        y = self.conv(y)                      # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1) # [B, C, 1, 1]
        y = self.sigmoid(y)

        return x * y


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention.

    Captures long-range positional information separately along
    height and width, making it useful for spatial structures such
    as roads, rivers, crop layouts, and residential patterns.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        reduced_channels = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.shared = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

        self.attn_h = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.attn_w = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        b, c, h, w = x.size()

        x_h = self.pool_h(x)                  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)      # [B, C, H+W, 1]
        y = self.shared(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        attn_h = self.sigmoid(self.attn_h(y_h))
        attn_w = self.sigmoid(self.attn_w(y_w))

        return identity * attn_h * attn_w


class EAGLEBlock(nn.Module):
    """
    EAGLE Block:
    Depthwise separable convolution + ECA + Coordinate Attention.

    This is the real attention-enhanced building block.
    """

    def __init__(self, in_channels, out_channels, use_coord=True):
        super().__init__()

        self.conv = DepthwiseSeparableConv(in_channels, out_channels)
        self.eca = ECABlock(out_channels)

        self.coord = (
            CoordinateAttention(out_channels)
            if use_coord
            else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.eca(x)
        x = self.coord(x)
        return x


class EAGLENet(nn.Module):
    """
    EAGLE-Net v2:
    Efficient Attention for Geo-spatial Land Estimation Network.

    Uses:
    - depthwise separable convolutions for efficiency
    - ECA for lightweight channel attention
    - Coordinate Attention for spatial/positional awareness
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            EAGLEBlock(32, 64, use_coord=False),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            EAGLEBlock(64, 128, use_coord=True),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            EAGLEBlock(128, 256, use_coord=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return self.classifier(x)


def create_model(model_name="eagle_net", num_classes=10):
    """
    Model factory.
    """

    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes)

    if model_name == "lightweight_cnn":
        return LightweightCNN(num_classes)

    if model_name in ["eagle_net", "eager_net"]:
        return EAGLENet(num_classes)

    raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)