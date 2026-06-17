import torch
import torch.nn as nn

from .backbone import FeatureBackbone
from .message_passing import NeuralMRF
from .reconstruction import ReconstructionHead


class NMRFDenoiser(nn.Module):

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

        # 1. Multi-scale feature extraction
        f1, f2, f3 = self.backbone(noisy_image)

        # 2. Neural MRF message passing on coarse features
        refined = self.nmrf(f3)

        # 3. Decode noise residual
        noise_residual = self.reconstruction(refined, f2, f1)

        # 4. Residual learning: subtract predicted noise
        denoised = torch.clamp(noisy_image - noise_residual, 0.0, 1.0)
        return denoised


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
    print("  Forward pass OK")
