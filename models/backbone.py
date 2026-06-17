import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    # Pre-activation style residual block with InstanceNorm.
    # Architecture: Conv3x3 -> InstanceNorm2d -> ReLU -> Conv3x3 -> InstanceNorm2d + skip

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FeatureBackbone(nn.Module):
    """Multi-scale ResNet feature extractor.
    Produces feature maps at three spatial resolutions suitable for
    the coarse-to-fine NMRF processing pipeline.

    Architecture overview::

        Level 1 (H×W):      Conv(in_ch → c)  + IN + ReLU, 2× ResBlock(c)
        Level 2 (H/2×W/2):  Conv(c → 2c, s=2) + IN + ReLU, 2× ResBlock(2c)
        Level 3 (H/4×W/4):  Conv(2c → 4c, s=2) + IN + ReLU, 2× ResBlock(4c)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # Level 1: full resolution  (B, c, H, W)
        self.level1 = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            ResidualBlock(c),
            ResidualBlock(c),
        )

        # Level 2: half resolution  (B, 2c, H/2, W/2)
        self.level2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(2 * c),
            nn.ReLU(inplace=True),
            ResidualBlock(2 * c),
            ResidualBlock(2 * c),
        )

        # Level 3: quarter resolution  (B, 4c, H/4, W/4)
        self.level3 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(4 * c),
            nn.ReLU(inplace=True),
            ResidualBlock(4 * c),
            ResidualBlock(4 * c),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract multi-scale features from the input image.

        f1 = self.level1(x) # full-resolution features
        f2 = self.level2(f1) # half-resolution features
        f3 = self.level3(f2) # coarse features for NMRF
        return f1, f2, f3
