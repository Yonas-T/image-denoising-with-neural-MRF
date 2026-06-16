"""Dataset classes for image denoising.

Provides:
* :class:`DenoisingDataset` — training dataset that extracts random
  patches from clean images and adds noise on-the-fly.
* :class:`DenoisingTestDataset` — evaluation dataset that loads full
  images and adds noise with a fixed seed for reproducibility.
"""

import os
import random
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .noise import add_gaussian_noise, add_poisson_noise

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(directory: str) -> list[str]:
    """Recursively list image files under *directory*."""
    paths = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths


class DenoisingDataset(Dataset):
    """Training dataset: random patches with on-the-fly noise injection.

    Each ``__getitem__`` call selects a random image, crops a random
    patch, optionally augments it, and adds synthetic noise.

    Args:
        image_dir: Path to directory of clean images.
        patch_size: Side length of square training patches.
        noise_type: ``'gaussian'`` or ``'poisson'``.
        noise_sigma: Standard deviation for Gaussian noise (0-255 scale).
        poisson_lambda: Intensity for Poisson noise.
        augment: Apply random flips and 90° rotations.
        patches_per_image: Virtual dataset multiplier — total length
            is ``len(images) × patches_per_image``.
    """

    def __init__(
        self,
        image_dir: str,
        patch_size: int = 64,
        noise_type: str = "gaussian",
        noise_sigma: float = 25.0,
        poisson_lambda: float = 30.0,
        augment: bool = True,
        patches_per_image: int = 8,
    ) -> None:
        self.image_paths = _list_images(image_dir)
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )
        self.patch_size = patch_size
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma
        self.poisson_lambda = poisson_lambda
        self.augment = augment
        self.patches_per_image = patches_per_image

    def __len__(self) -> int:
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(noisy_patch, clean_patch)`` each ``(3, P, P)``."""
        img_idx = idx % len(self.image_paths)
        img = Image.open(self.image_paths[img_idx]).convert("RGB")
        w, h = img.size
        ps = self.patch_size

        # Ensure image is large enough for a patch
        if w < ps or h < ps:
            img = img.resize((max(w, ps), max(h, ps)), Image.LANCZOS)
            w, h = img.size

        # Random crop
        x0 = random.randint(0, w - ps)
        y0 = random.randint(0, h - ps)
        patch = img.crop((x0, y0, x0 + ps, y0 + ps))

        # Convert to tensor [0, 1]
        clean = torch.from_numpy(
            np.asarray(patch).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, P, P)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                clean = torch.flip(clean, [2])
            # Random vertical flip
            if random.random() > 0.5:
                clean = torch.flip(clean, [1])
            # Random 90° rotation
            k = random.randint(0, 3)
            if k > 0:
                clean = torch.rot90(clean, k, [1, 2])

        # Add noise
        if self.noise_type == "poisson":
            noisy = add_poisson_noise(clean, lam=self.poisson_lambda)
        else:
            noisy = add_gaussian_noise(clean, sigma=self.noise_sigma)

        return noisy, clean


class DenoisingTestDataset(Dataset):
    """Evaluation dataset: full images with reproducible noise.

    Each image receives noise generated with a fixed per-index seed
    so results are reproducible across runs.

    Args:
        image_dir: Path to directory of clean test images.
        noise_type: ``'gaussian'`` or ``'poisson'``.
        noise_sigma: Gaussian noise level.
        poisson_lambda: Poisson noise intensity.
    """

    def __init__(
        self,
        image_dir: str,
        noise_type: str = "gaussian",
        noise_sigma: float = 25.0,
        poisson_lambda: float = 30.0,
    ) -> None:
        self.image_paths = _list_images(image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.noise_type = noise_type
        self.noise_sigma = noise_sigma
        self.poisson_lambda = poisson_lambda

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return ``(noisy_img, clean_img, filename)``."""
        path = self.image_paths[idx]
        filename = os.path.basename(path)

        img = Image.open(path).convert("RGB")
        clean = torch.from_numpy(
            np.asarray(img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        # Fixed seed per image index for reproducibility
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(idx + 42)

        if self.noise_type == "poisson":
            noisy = add_poisson_noise(clean, lam=self.poisson_lambda)
        else:
            noisy = add_gaussian_noise(clean, sigma=self.noise_sigma)

        torch.random.set_rng_state(rng_state)

        return noisy, clean, filename
