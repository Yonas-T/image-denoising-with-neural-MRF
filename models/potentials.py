
import torch
import torch.nn as nn
import torch.nn.functional as F


class PottsAggregation(nn.Module):

    def __init__(self, dim: int, window_size: int = 5, sigma: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.sigma = sigma
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        M = self.window_size
        pad = M // 2

        # Reflect-pad to handle borders  -> (B, C, H+2*pad, W+2*pad)
        x_pad = F.pad(x, [pad, pad, pad, pad], mode="reflect")

        # Extract MxM neighborhoods via unfold -> (B, C, H, W, M, M)
        patches = x_pad.unfold(2, M, 1).unfold(3, M, 1)
        # -> (B, H, W, M*M, C)
        patches = (
            patches.contiguous()
            .reshape(B, C, H, W, M * M)
            .permute(0, 2, 3, 4, 1)
        )

        # Center pixel: (B, H, W, 1, C)
        center = x.permute(0, 2, 3, 1).unsqueeze(3)

        # L2 squared distance between center and each neighbor
        # (B, H, W, M*M)
        dist_sq = ((center - patches) ** 2).sum(dim=-1)

        # Gaussian weights: exp(-d^2 / (2 * sigma^2))
        weights = torch.exp(-dist_sq / (2.0 * self.sigma ** 2))

        # Normalize weights to sum to 1  (B, H, W, M*M)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted aggregation  (B, H, W, C)
        out = (weights.unsqueeze(-1) * patches).sum(dim=3)

        # Linear output projection
        out = self.out_proj(out)

        # -> (B, C, H, W)
        return out.permute(0, 3, 1, 2)
