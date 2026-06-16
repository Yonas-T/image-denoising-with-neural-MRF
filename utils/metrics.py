"""Image quality metrics for denoising evaluation.

All functions accept NumPy arrays **or** PyTorch tensors in ``[0, 1]``
and return a scalar ``float``.
"""

import numpy as np


def _to_numpy(x) -> np.ndarray:
    """Convert a tensor or ndarray to a float64 NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def compute_psnr(
    clean: np.ndarray, denoised: np.ndarray, data_range: float = 1.0
) -> float:
    """Peak Signal-to-Noise Ratio (higher is better).

    Args:
        clean: Ground-truth image.
        denoised: Restored image.
        data_range: Dynamic range of the images (1.0 for [0,1]).

    Returns:
        PSNR in dB.  Returns 50.0 if MSE ≈ 0.
    """
    clean, denoised = _to_numpy(clean), _to_numpy(denoised)
    mse = np.mean((clean - denoised) ** 2)
    if mse < 1e-10:
        return 50.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim(
    clean: np.ndarray,
    denoised: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Structural Similarity Index (higher is better).

    Uses ``skimage`` when available; otherwise falls back to a
    simplified single-scale implementation.

    Args:
        clean: Ground-truth image (H, W, C) or (C, H, W).
        denoised: Restored image, same shape.
        data_range: Dynamic range.

    Returns:
        SSIM value in ``[0, 1]``.
    """
    clean, denoised = _to_numpy(clean), _to_numpy(denoised)

    # 1. Squeeze batch dimension if present (e.g., (1, C, H, W) -> (C, H, W))
    if clean.ndim == 4 and clean.shape[0] == 1:
        clean = np.squeeze(clean, axis=0)
        denoised = np.squeeze(denoised, axis=0)

    # Ensure (H, W, C)
    if clean.ndim == 3 and clean.shape[0] in (1, 3):
        clean = np.transpose(clean, (1, 2, 0))
        denoised = np.transpose(denoised, (1, 2, 0))

    try:
        from skimage.metrics import structural_similarity
        
        # 2. Dynamically determine window size based on image size
        min_side = min(clean.shape[0], clean.shape[1])
        if min_side < 3:
            raise ValueError("Image dimensions too small for sliding window SSIM")
            
        win_size = min(7, min_side)
        if win_size % 2 == 0:
            win_size -= 1
            
        return float(
            structural_similarity(
                clean, denoised, data_range=data_range, 
                channel_axis=-1 if clean.ndim == 3 else None,
                win_size=win_size
            )
        )
    except (ImportError, ValueError):
        pass

    # Fallback: simplified global SSIM
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = clean.mean()
    mu_y = denoised.mean()
    sig_x_sq = clean.var()
    sig_y_sq = denoised.var()
    sig_xy = np.mean((clean - mu_x) * (denoised - mu_y))
    l = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
    cs = (2 * sig_xy + c2) / (sig_x_sq + sig_y_sq + c2)
    return float(l * cs)

def compute_mae(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Mean Absolute Error (lower is better).

    Args:
        clean: Ground-truth image.
        denoised: Restored image.

    Returns:
        MAE value.
    """
    clean, denoised = _to_numpy(clean), _to_numpy(denoised)
    return float(np.mean(np.abs(clean - denoised)))
