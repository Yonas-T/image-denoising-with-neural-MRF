
import torch
import torch.nn as nn

from .backbone import ResidualBlock


class ReconstructionHead(nn.Module):

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # --- Stage 1: 4c → 2c, upscale ×2 ---
        self.up1 = nn.ConvTranspose2d(
            4 * c, 2 * c, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.up1_norm = nn.InstanceNorm2d(2 * c)
        self.up1_act = nn.ReLU(inplace=True)

        # After concat with f2 skip: 2c + 2c = 4c → 2c
        self.fuse1 = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(2 * c),
            nn.ReLU(inplace=True),
            ResidualBlock(2 * c),
        )

        # --- Stage 2: 2c → c, upscale ×2 ---
        self.up2 = nn.ConvTranspose2d(
            2 * c, c, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.up2_norm = nn.InstanceNorm2d(c)
        self.up2_act = nn.ReLU(inplace=True)

        # After concat with f1 skip: c + c = 2c → c
        self.fuse2 = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            ResidualBlock(c),
        )

        # --- Final head: predict noise residual ---
        self.head = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(
        self,
        coarse_features: torch.Tensor,
        f2_skip: torch.Tensor,
        f1_skip: torch.Tensor,
    ) -> torch.Tensor:

        # Stage 1: upsample coarse 4c → 2c, fuse with f2
        x = self.up1_act(self.up1_norm(self.up1(coarse_features)))
        x = torch.cat([x, f2_skip], dim=1)  # (B, 4c, H/2, W/2)
        x = self.fuse1(x)  # (B, 2c, H/2, W/2)

        # Stage 2: upsample 2c → c, fuse with f1
        x = self.up2_act(self.up2_norm(self.up2(x)))
        x = torch.cat([x, f1_skip], dim=1)  # (B, 2c, H, W)
        x = self.fuse2(x)  # (B, c, H, W)

        # Predict noise residual
        noise_residual = self.head(x)  # (B, 3, H, W)
        return noise_residual
