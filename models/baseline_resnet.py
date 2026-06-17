# Pure ResNet encoder-decoder baseline for image denoising.
# Contains NO MRF or message-passing components and it serves as the deep-learning-only comparison baseline

import torch
import torch.nn as nn

from .backbone import ResidualBlock


class ResNetDenoiser(nn.Module):
    """Encoder-decoder ResNet denoiser (no graphical model).

    Architecture::

        Encoder:
            Level 1: Conv(3 → c)    + IN + ReLU, 2× ResBlock(c)       → skip1
            Level 2: Conv(c → 2c, s=2)  + IN + ReLU, 2× ResBlock(2c)  → skip2
            Level 3: Conv(2c → 4c, s=2) + IN + ReLU, 2× ResBlock(4c)

        Bottleneck: 4× ResBlock(4c)

        Decoder:
            Stage 1: ConvT(4c → 2c, s=2) ─╮ concat skip2 ─► Conv(4c → 2c) + ResBlock
            Stage 2: ConvT(2c → c, s=2)  ─╮ concat skip1 ─► Conv(2c → c) + ResBlock

        Head: Conv(c → 3) → noise residual

    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # ---- Encoder --------------------------------------------------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            ResidualBlock(c),
            ResidualBlock(c),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(2 * c),
            nn.ReLU(inplace=True),
            ResidualBlock(2 * c),
            ResidualBlock(2 * c),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(4 * c),
            nn.ReLU(inplace=True),
            ResidualBlock(4 * c),
            ResidualBlock(4 * c),
        )

        # ---- Bottleneck -----------------------------------------------------
        self.bottleneck = nn.Sequential(
            ResidualBlock(4 * c),
            ResidualBlock(4 * c),
            ResidualBlock(4 * c),
            ResidualBlock(4 * c),
        )

        # ---- Decoder --------------------------------------------------------
        # Stage 1: upsample 4c → 2c, fuse with skip2
        self.up1 = nn.ConvTranspose2d(
            4 * c, 2 * c, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.up1_norm = nn.InstanceNorm2d(2 * c)
        self.up1_act = nn.ReLU(inplace=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(2 * c),
            nn.ReLU(inplace=True),
            ResidualBlock(2 * c),
        )

        # Stage 2: upsample 2c → c, fuse with skip1
        self.up2 = nn.ConvTranspose2d(
            2 * c, c, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.up2_norm = nn.InstanceNorm2d(c)
        self.up2_act = nn.ReLU(inplace=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            ResidualBlock(c),
        )

        # ---- Head -----------------------------------------------------------
        self.head = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, noisy_image: torch.Tensor) -> torch.Tensor:
        
        # Encode
        s1 = self.enc1(noisy_image)   # (B, c,  H,   W)
        s2 = self.enc2(s1)            # (B, 2c, H/2, W/2)
        x = self.enc3(s2)             # (B, 4c, H/4, W/4)

        # Bottleneck
        x = self.bottleneck(x)        # (B, 4c, H/4, W/4)

        # Decode stage 1
        x = self.up1_act(self.up1_norm(self.up1(x)))   # (B, 2c, H/2, W/2)
        x = torch.cat([x, s2], dim=1)                  # (B, 4c, H/2, W/2)
        x = self.dec1(x)                                # (B, 2c, H/2, W/2)

        # Decode stage 2
        x = self.up2_act(self.up2_norm(self.up2(x)))    # (B, c, H, W)
        x = torch.cat([x, s1], dim=1)                   # (B, 2c, H, W)
        x = self.dec2(x)                                 # (B, c, H, W)

        # Predict noise residual
        noise_residual = self.head(x)                    # (B, 3, H, W)
        denoised = torch.clamp(noisy_image - noise_residual, 0.0, 1.0)
        return denoised


# ------------------------------------------------------------------
#  Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = ResNetDenoiser()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("ResNetDenoiser")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    x = torch.randn(1, 3, 64, 64).clamp(0, 1)
    with torch.no_grad():
        y = model(x)
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")
    print(f"  Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print("  ✓ Forward pass OK")
