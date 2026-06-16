"""Full NMRF denoiser combining backbone, neural MRF, and reconstruction.

Implements the complete Neural Markov Random Field pipeline for
image denoising as described in Guan et al. (CVPR 2024), adapted
from stereo matching to 2D image denoising:

1. **Backbone** — multi-scale feature extraction.
2. **Neural MRF** — iterative message passing on coarse features.
3. **Reconstruction** — U-Net-style decoder with skip connections.

The model follows residual learning: it predicts a noise residual
and subtracts it from the input to produce the denoised output.
"""

import torch
import torch.nn as nn

from .backbone import FeatureBackbone
from .message_passing import NeuralMRF
from .reconstruction import ReconstructionHead


class NMRFDenoiser(nn.Module):
    """Neural Markov Random Field image denoiser.

    End-to-end model that extracts multi-scale features, refines the
    coarsest level with iterative message passing, and decodes a
    noise residual via skip-connected upsampling.

    Args:
        in_channels: Number of input image channels (default: 3).
        base_channels: Base channel count ``c`` (default: 32).
        num_iterations: Number of message-passing iterations in the
            NMRF module.
        window_size: Spatial window ``M`` for neighbor aggregation.
        num_heads: Attention heads in neighbor aggregation.
        num_groups: Channel groups in self-edge aggregation.
        use_self_edges: Include self-edge aggregation layers.
        use_neural_potentials: If *True*, use learned attention
            potentials; if *False*, use fixed Potts/Gaussian.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_iterations: int = 4,
        window_size: int = 5,
        num_heads: int = 4,
        num_groups: int = 8,
        use_self_edges: bool = True,
        use_neural_potentials: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = FeatureBackbone(in_channels, base_channels)
        self.nmrf = NeuralMRF(
            dim=4 * base_channels,
            num_iterations=num_iterations,
            window_size=window_size,
            num_heads=num_heads,
            num_groups=num_groups,
            use_self_edges=use_self_edges,
            use_neural_potentials=use_neural_potentials,
        )
        self.reconstruction = ReconstructionHead(base_channels)

    def forward(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """Denoise an input image.

        Args:
            noisy_image: Noisy RGB image tensor of shape
                ``(B, 3, H, W)`` with pixel values in ``[0, 1]``.
                ``H`` and ``W`` must each be divisible by 4.

        Returns:
            Denoised image of shape ``(B, 3, H, W)`` clamped to
            ``[0, 1]``.
        """
        # 1. Multi-scale feature extraction
        f1, f2, f3 = self.backbone(noisy_image)

        # 2. Neural MRF message passing on coarse features
        refined = self.nmrf(f3)

        # 3. Decode noise residual
        noise_residual = self.reconstruction(refined, f2, f1)

        # 4. Residual learning: subtract predicted noise
        denoised = torch.clamp(noisy_image - noise_residual, 0.0, 1.0)
        return denoised


# ------------------------------------------------------------------
#  Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = NMRFDenoiser()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NMRFDenoiser")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    x = torch.randn(1, 3, 64, 64).clamp(0, 1)
    with torch.no_grad():
        y = model(x)
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")
    print(f"  Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print("  ✓ Forward pass OK")
