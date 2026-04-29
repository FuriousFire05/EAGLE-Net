# src/models/architectures.py
"""Neural network architectures for EuroSAT satellite image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Standard CNN baseline for EuroSAT classification.
    Used as the accuracy-focused reference model.

    Args:
        num_classes: Number of output classes for classification.
    """

    def __init__(self, num_classes=10):
        """Initialize convolutional feature extraction and classifier layers."""
        super().__init__()

        # Four convolutional stages increase channel capacity while reducing
        # spatial resolution through max pooling.
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
        """Run a forward pass and return class logits."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block for efficient feature extraction."""

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize a depthwise convolution followed by pointwise projection.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            stride: Spatial stride for the depthwise convolution.
        """
        super().__init__()

        # Depthwise convolution captures spatial patterns per channel; pointwise
        # convolution mixes information across channels.
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
        """Apply the separable convolution block to an input tensor."""
        return self.block(x)


class LightweightCNN(nn.Module):
    """Compact CNN using depthwise separable convolutions for lower latency."""

    def __init__(self, num_classes=10):
        """
        Initialize the lightweight classification network.

        Args:
            num_classes: Number of output classes for classification.
        """
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
        """Run a forward pass and return class logits."""
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class BlurPool(nn.Module):
    """Anti-aliasing downsampling layer implemented with a fixed blur kernel."""

    def __init__(self, channels, stride=2):
        """
        Initialize a channel-wise blur kernel.

        Args:
            channels: Number of channels the blur kernel should be repeated for.
            stride: Downsampling stride used during convolution.
        """
        super().__init__()
        self.stride = stride

        # Build a normalized 3x3 binomial filter and repeat it so grouped
        # convolution applies the same blur independently to each channel.
        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, :, :].repeat(channels, 1, 1, 1)

        self.register_buffer("kernel", kernel)

    def forward(self, x):
        """Blur and optionally downsample an input feature map."""
        padding = self.kernel.size(-1) // 2  # ✅ FIXED

        return F.conv2d(
            x,
            self.kernel,
            stride=self.stride,
            padding=padding,
            groups=x.size(1),
        )


class SEBlock(nn.Module):
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, channels, reduction=8):
        """
        Initialize channel attention layers.

        Args:
            channels: Number of input and output feature channels.
            reduction: Reduction ratio for the intermediate attention channels.
        """
        super().__init__()

        reduced_channels = max(8, channels // reduction)

        # Global pooling summarizes each channel; 1x1 convolutions produce
        # per-channel gates in the range [0, 1].
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply channel-wise attention to an input feature map."""
        return x * self.block(x)


class SpatialGate(nn.Module):
    """Spatial attention gate using average and max feature summaries."""

    def __init__(self, kernel_size=7):
        """
        Initialize the spatial gating convolution.

        Args:
            kernel_size: Kernel size used to compute the spatial attention map.
        """
        super().__init__()

        padding = kernel_size // 2
        self.gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply spatial attention and return gated feature maps."""
        # Average and max projections provide complementary spatial summaries.
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.gate(torch.cat([avg_map, max_map], dim=1))
        return x * gate


class DualKernelInvertedResidual(nn.Module):
    """EAGLE-Net block with expansion, dual-kernel depthwise branches, and projection."""

    def __init__(self, in_channels, out_channels, expansion=2, stride=1, use_se=False):
        """
        Initialize an inverted residual block with 3x3 and 5x5 depthwise paths.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of projected output channels.
            expansion: Multiplier used to expand hidden channel width.
            stride: Spatial stride; stride 2 uses BlurPool downsampling.
            use_se: Whether to apply squeeze-and-excitation channel attention.
        """
        super().__init__()

        hidden_channels = in_channels * expansion
        branch_channels = hidden_channels // 2
        other_channels = hidden_channels - branch_channels

        self.use_residual = stride == 1 and in_channels == out_channels

        # Expand channels before splitting into separate spatial-kernel branches.
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.blur = BlurPool(hidden_channels) if stride == 2 else nn.Identity()

        # Parallel depthwise branches capture local and wider spatial context
        # without the cost of full convolutions.
        self.dw3 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.dw5 = nn.Sequential(
            nn.Conv2d(other_channels, other_channels, kernel_size=5, padding=2, groups=other_channels, bias=False),  # ✅ FIXED
            nn.BatchNorm2d(other_channels),
            nn.ReLU(inplace=True),
        )

        self.se = SEBlock(hidden_channels) if use_se else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """Run the dual-kernel residual block and return projected features."""
        identity = x

        x = self.expand(x)
        x = self.blur(x)

        # Split expanded channels between the 3x3 and 5x5 depthwise branches.
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
    """Robust satellite image classifier built from dual-kernel residual blocks."""

    def __init__(self, num_classes=10):
        """
        Initialize the EAGLE-Net architecture.

        Args:
            num_classes: Number of output classes for classification.
        """
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),  # ✅ FIXED
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
        """Run a forward pass and return class logits."""
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
    Create a model instance by name.

    Args:
        model_name: Architecture identifier.
        num_classes: Number of output classes for classification.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If the requested model name is not supported.
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
    Count trainable parameters in a model.

    Args:
        model: PyTorch module to inspect.

    Returns:
        Number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
