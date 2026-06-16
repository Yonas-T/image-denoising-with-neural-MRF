"""Neural MRF Image Denoising — Model package.

Exports the two primary model classes used for denoising:

* :class:`NMRFDenoiser` — Hybrid Neural Markov Random Field denoiser.
* :class:`ResNetDenoiser` — Pure deep-learning (ResNet) baseline.
"""

from .nmrf_denoiser import NMRFDenoiser
from .baseline_resnet import ResNetDenoiser

__all__ = ["NMRFDenoiser", "ResNetDenoiser"]
