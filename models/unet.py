"""
Standard U-Net architecture for image-to-image regression.
Adapted for SimRegMatch framework with feature extraction for similarity-based pseudo-labeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(Up, self).__init__()
        
        # Upsampling using transposed convolution
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch (if input size is not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for image-to-image regression.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for height map)
        base_channels: Number of channels in first layer (default: 64)
        dropout: Dropout rate (default: 0.1)
    
    Returns:
        predictions: (B, out_channels, H, W) - predicted height map
        encoding: (B, feature_dim) - feature vector for similarity computation
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, dropout=0.1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, dropout=dropout)
        self.down1 = Down(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=dropout)
        self.down4 = Down(base_channels * 8, base_channels * 16, dropout=dropout)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8, dropout=dropout)
        self.up2 = Up(base_channels * 8, base_channels * 4, dropout=dropout)
        self.up3 = Up(base_channels * 4, base_channels * 2, dropout=dropout)
        self.up4 = Up(base_channels * 2, base_channels, dropout=dropout)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Global average pooling for feature extraction (for similarity computation)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension for encoding
        self.feature_dim = base_channels * 16

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16) - bottleneck
        
        # Extract feature vector from bottleneck for similarity computation
        encoding = self.global_pool(x5)  # (B, 1024, 1, 1)
        encoding = encoding.view(encoding.size(0), -1)  # (B, 1024)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # Output layer
        predictions = self.outc(x)  # (B, 1, H, W)
        
        return predictions, encoding


class UNetSmall(nn.Module):
    """
    Smaller U-Net with 3 encoder/decoder levels for image-to-image regression.
    Adapted for SimRegMatch framework with feature extraction for similarity-based pseudo-labeling.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for height map)
        base_channels: Number of channels in first layer (default: 64)
        dropout: Dropout rate (default: 0.1)
    
    Returns:
        predictions: (B, out_channels, H, W) - predicted height map
        encoding: (B, feature_dim) - feature vector for similarity computation
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, dropout=0.1):
        super(UNetSmall, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder (3 levels)
        self.inc = DoubleConv(in_channels, base_channels, dropout=dropout)
        self.down1 = Down(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=dropout)
        
        # Decoder (3 levels)
        self.up1 = Up(base_channels * 8, base_channels * 4, dropout=dropout)
        self.up2 = Up(base_channels * 4, base_channels * 2, dropout=dropout)
        self.up3 = Up(base_channels * 2, base_channels, dropout=dropout)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Global average pooling for feature extraction (for similarity computation)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension for encoding (512 channels instead of 1024)
        self.feature_dim = base_channels * 8

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8) - bottleneck
        
        # Extract feature vector from bottleneck for similarity computation
        encoding = self.global_pool(x4)  # (B, 512, 1, 1)
        encoding = encoding.view(encoding.size(0), -1)  # (B, 512)
        
        # Decoder with skip connections
        x = self.up1(x4, x3)  # (B, 256, H/4, W/4)
        x = self.up2(x, x2)   # (B, 128, H/2, W/2)
        x = self.up3(x, x1)   # (B, 64, H, W)
        
        # Output layer
        predictions = self.outc(x)  # (B, 1, H, W)
        
        return predictions, encoding


def unet(in_channels=3, out_channels=1, base_channels=64, dropout=0.1):
    """
    Create a U-Net model.
    
    Args:
        in_channels: Number of input channels (default: 3)
        out_channels: Number of output channels (default: 1)
        base_channels: Base number of channels (default: 64)
        dropout: Dropout rate (default: 0.1)
    
    Returns:
        UNet model
    """
    return UNet(in_channels=in_channels, 
                out_channels=out_channels, 
                base_channels=base_channels, 
                dropout=dropout)


def unet_small(in_channels=3, out_channels=1, base_channels=64, dropout=0.1):
    """
    Create a smaller U-Net model with 3 encoder/decoder levels.
    
    Args:
        in_channels: Number of input channels (default: 3)
        out_channels: Number of output channels (default: 1)
        base_channels: Base number of channels (default: 64)
        dropout: Dropout rate (default: 0.1)
    
    Returns:
        UNetSmall model
    """
    return UNetSmall(in_channels=in_channels, 
                     out_channels=out_channels, 
                     base_channels=base_channels, 
                     dropout=dropout)


if __name__ == '__main__':
    # Test the model
    model = unet(in_channels=3, out_channels=1, dropout=0.1)
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    predictions, encoding = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Encoding shape: {encoding.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
