"""Visualization utilities for denoising results.

Provides helpers to create comparison plots, ablation charts,
and training curve figures using matplotlib.
"""

import os
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def plot_denoising_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised_nmrf: np.ndarray,
    denoised_baseline: np.ndarray,
    metrics_nmrf: Optional[dict] = None,
    metrics_baseline: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side comparison: Clean | Noisy | NMRF | Baseline.

    Args:
        clean: Clean image (H, W, 3) in [0, 1].
        noisy: Noisy image.
        denoised_nmrf: NMRF output.
        denoised_baseline: Baseline output.
        metrics_nmrf: Dict with 'psnr', 'ssim', 'mae' for NMRF.
        metrics_baseline: Same for baseline.
        save_path: If given, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Clean", "Noisy", "NMRF Denoised", "ResNet Baseline"]
    images = [clean, noisy, denoised_nmrf, denoised_baseline]
    metric_dicts = [None, None, metrics_nmrf, metrics_baseline]

    for ax, img, title, md in zip(axes, images, titles, metric_dicts):
        # Ensure (H, W, 3)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        ax.imshow(np.clip(img, 0, 1))
        label = title
        if md:
            label += f"\nPSNR={md.get('psnr', 0):.2f}  SSIM={md.get('ssim', 0):.4f}"
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_ablation_results(
    results: dict[str, dict[str, float]],
    title: str = "Ablation Study",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing ablation variants.

    Args:
        results: ``{variant_name: {'psnr': ..., 'ssim': ..., 'mae': ...}}``.
        title: Plot title.
        save_path: Optional save path.
    """
    variants = list(results.keys())
    psnrs = [results[v]["psnr"] for v in variants]
    ssims = [results[v]["ssim"] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(variants))
    bars1 = ax1.bar(x, psnrs, color="#6366f1", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title(f"{title} — PSNR")
    for bar, val in zip(bars1, psnrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", fontsize=8)

    bars2 = ax2.bar(x, ssims, color="#22d3ee", edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("SSIM")
    ax2.set_title(f"{title} — SSIM")
    for bar, val in zip(bars2, ssims):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(
    train_losses: list[float],
    val_psnrs: list[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot training loss and validation PSNR curves.

    Args:
        train_losses: Per-epoch training loss values.
        val_psnrs: Per-epoch validation PSNR values.
        save_path: Optional save path.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, color="#6366f1", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("L1 Loss")
    ax1.set_title("Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_psnrs, color="#10b981", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Validation PSNR")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
