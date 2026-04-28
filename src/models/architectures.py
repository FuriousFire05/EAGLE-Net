"""Model architectures for EAGLE-Net project."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline CNN for EuroSAT classification.
    Standard convolutional blocks with batch normalization.
    """
    
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        
        # Conv Block 1: 3 -> 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
        )
        
        # Conv Block 2: 32 -> 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
        )
        
        # Conv Block 3: 64 -> 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
        )
        
        # Conv Block 4: 128 -> 256 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.
    Reduces parameters: standard conv -> depthwise + pointwise.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise: apply conv separately to each input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels
        )
        
        # Pointwise: 1x1 conv to change channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class LightweightCNN(nn.Module):
    """
    Lightweight CNN using depthwise separable convolutions.
    Significantly fewer parameters than BaselineCNN.
    """
    
    def __init__(self, num_classes=10):
        super(LightweightCNN, self).__init__()
        
        # Initial conv (standard, small kernel)
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Block 1: 32 -> 64 (depthwise separable)
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
        )
        
        # Block 2: 64 -> 128
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
        )
        
        # Block 3: 128 -> 256
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention Block (SE-Net style).
    Learns to weight channels adaptively.
    """
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared FC layers
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        
        # Apply to input
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Block.
    Learns spatial distribution of important regions.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        out = self.sigmoid(out)
        
        # Apply to input
        return x * out


class EAGLENet(nn.Module):
    """
    EAGLE-Net: Efficient Attention for Geo-spatial Land Estimation Network.
    
    Combines:
    - LightweightCNN backbone (depthwise separable convolutions)
    - Channel Attention (SE-Net style)
    - Spatial Attention (adaptive spatial weighting)
    """
    
    def __init__(self, num_classes=10):
        super(EAGLENet, self).__init__()
        
        # Initial conv
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Block 1: 32 -> 64 with attention
        self.block1_conv = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.block1_channel_attn = ChannelAttention(64, reduction=4)
        self.block1_spatial_attn = SpatialAttention(kernel_size=7)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32
        
        # Block 2: 64 -> 128 with attention
        self.block2_conv = nn.Sequential(
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
        )
        self.block2_channel_attn = ChannelAttention(128, reduction=8)
        self.block2_spatial_attn = SpatialAttention(kernel_size=7)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 -> 16
        
        # Block 3: 128 -> 256 with attention
        self.block3_conv = nn.Sequential(
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.block3_channel_attn = ChannelAttention(256, reduction=16)
        self.block3_spatial_attn = SpatialAttention(kernel_size=7)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 16 -> 8
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv0(x)
        
        # Block 1
        x = self.block1_conv(x)
        x = self.block1_channel_attn(x)
        x = self.block1_spatial_attn(x)
        x = self.block1_pool(x)
        
        # Block 2
        x = self.block2_conv(x)
        x = self.block2_channel_attn(x)
        x = self.block2_spatial_attn(x)
        x = self.block2_pool(x)
        
        # Block 3
        x = self.block3_conv(x)
        x = self.block3_channel_attn(x)
        x = self.block3_spatial_attn(x)
        x = self.block3_pool(x)
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def create_model(model_name="eagle_net", num_classes=10):
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes)
    
    elif model_name == "lightweight_cnn":
        return LightweightCNN(num_classes)
    
    # support both temporarily (backward compatibility)
    elif model_name in ["eagle_net", "eager_net"]:
        return EAGLENet(num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
