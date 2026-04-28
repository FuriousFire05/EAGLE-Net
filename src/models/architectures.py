# src/models/architectures.py

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BlurPool(nn.Module):
    """
    Anti-aliased downsampling with a fixed binomial blur kernel.
    """

    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride

        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, :, :].repeat(channels, 1, 1, 1)

        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return F.conv2d(
            x,
            self.kernel,
            stride=self.stride,
            padding=1,
            groups=x.size(1),
        )


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        reduced_channels = max(8, channels // reduction)

        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class SpatialGate(nn.Module):
    """
    Lightweight spatial gate for late-stage feature refinement.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.gate(torch.cat([avg_map, max_map], dim=1))
        return x * gate


class DualKernelInvertedResidual(nn.Module):
    """
    Inverted residual block with parallel 3x3 and 5x5 depthwise branches.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=2,
        stride=1,
        use_se=False,
    ):
        super().__init__()

        hidden_channels = in_channels * expansion
        branch_channels = hidden_channels // 2
        other_channels = hidden_channels - branch_channels
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.blur = BlurPool(hidden_channels) if stride == 2 else nn.Identity()

        self.dw3 = nn.Sequential(
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.dw5 = nn.Sequential(
            nn.Conv2d(
                other_channels,
                other_channels,
                kernel_size=5,
                padding=2,
                groups=hidden_channels - branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels - branch_channels),
            nn.ReLU(inplace=True),
        )

        self.se = SEBlock(hidden_channels) if use_se else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x

        x = self.expand(x)
        x = self.blur(x)

        c = x.size(1)
        c_half = c // 2
        x3, x5 = x[:, :c_half, :, :], x[:, c_half:, :, :]

        x = torch.cat([self.dw3(x3), self.dw5(x5)], dim=1)
        x = self.se(x)
        x = self.project(x)

        if self.use_residual:
            x = x + identity

        return x


class EAGLENet(nn.Module):
    """
    EAGLE-Net v3:
    Efficient Attention for Geo-spatial Land Estimation Network.

    Uses:
    - dual-kernel inverted residual blocks
    - 3x3 and 5x5 depthwise branches
    - SE attention only in mid/late blocks
    - one late spatial gate
    - anti-aliased downsampling with BlurPool
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding = self.kernel.size(-1) // 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            DualKernelInvertedResidual(32, 48, expansion=2, stride=1, use_se=False),
            DualKernelInvertedResidual(48, 64, expansion=2, stride=2, use_se=False),
        )

        self.stage2 = nn.Sequential(
            DualKernelInvertedResidual(64, 96, expansion=3, stride=1, use_se=True),
            DualKernelInvertedResidual(96, 128, expansion=3, stride=2, use_se=True),
        )

        self.stage3 = nn.Sequential(
            DualKernelInvertedResidual(128, 160, expansion=3, stride=1, use_se=True),
            DualKernelInvertedResidual(160, 256, expansion=3, stride=2, use_se=True),
        )

        self.spatial_gate = SpatialGate()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.25)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.spatial_gate(x)

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

    if model_name == "eagle_net":
        return EAGLENet(num_classes)

    raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model):
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
