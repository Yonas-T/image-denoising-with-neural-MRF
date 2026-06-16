"""Noise generation utilities for image denoising.

Supports additive Gaussian noise and Poisson noise, both operating
on float tensors in the [0, 1] range.
"""

import torch


def add_gaussian_noise(clean_img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add additive white Gaussian noise.

    Args:
        clean_img: Clean image tensor of shape ``(C, H, W)`` or
            ``(B, C, H, W)`` with values in ``[0, 1]``.
        sigma: Noise level on the **[0, 255]** scale.  For example,
            ``sigma=25`` means the standard deviation of the noise
            is ``25/255 ≈ 0.098`` in the ``[0, 1]`` domain.

    Returns:
        Noisy image clamped to ``[0, 1]``, same shape as input.
    """
    noise = torch.randn_like(clean_img) * (sigma / 255.0)
    return torch.clamp(clean_img + noise, 0.0, 1.0)


def add_poisson_noise(clean_img: torch.Tensor, lam: float = 30.0) -> torch.Tensor:
    """Add Poisson noise.

    The noise model is: ``noisy = Poisson(clean × λ) / λ``.
    Higher ``lam`` produces *less* noise; lower ``lam`` produces *more*.

    Args:
        clean_img: Clean image tensor of shape ``(C, H, W)`` or
            ``(B, C, H, W)`` with values in ``[0, 1]``.
        lam: Poisson intensity parameter (default 30.0).

    Returns:
        Noisy image clamped to ``[0, 1]``, same shape as input.
    """
    lam = max(lam, 1.0)
    noisy = torch.poisson(clean_img * lam) / lam
    return torch.clamp(noisy, 0.0, 1.0)
