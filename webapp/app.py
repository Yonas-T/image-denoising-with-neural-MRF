"""
Neural MRF Image Denoiser — Flask Backend
==========================================
Serves the single-page web UI and exposes a /denoise endpoint that:
  1. Accepts an uploaded image + noise parameters.
  2. Adds Gaussian or Poisson noise.
  3. Runs both NMRF and ResNet-baseline denoisers.
  4. Returns base64-encoded images and quality metrics (PSNR, SSIM, MAE).
"""

import io
import sys
import os
import base64
import logging
import traceback

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Path setup – let imports find the project-root packages (models, utils, data)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lazy torch import (gives a clear error if missing)
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    raise SystemExit("PyTorch is required.  Install it:  pip install torch torchvision")

# ---------------------------------------------------------------------------
# Model & utility imports – gracefully degrade when code/checkpoints missing
# ---------------------------------------------------------------------------
MODELS_AVAILABLE = False
CHECKPOINTS_LOADED = False
_warnings: list[str] = []

nmrf_model = None
baseline_model = None
potts_model = None

try:
    from models import NMRFDenoiser, ResNetDenoiser
    MODELS_AVAILABLE = True
except Exception as exc:
    _warnings.append(f"Could not import model classes: {exc}")

# Noise helpers -----------------------------------------------------------------
try:
    from data.noise import add_gaussian_noise, add_poisson_noise
except Exception:
    # Fallback pure-torch implementations
    def add_gaussian_noise(img_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add Gaussian noise with *sigma* in [0, 255] scale (image is [0,1])."""
        return (img_tensor + torch.randn_like(img_tensor) * (sigma / 255.0)).clamp(0, 1)

    def add_poisson_noise(img_tensor: torch.Tensor, lam: float) -> torch.Tensor:
        """Poisson-noise approximation controlled by *lam* (higher = less noise)."""
        lam = max(lam, 1.0)
        noisy = torch.poisson(img_tensor * lam) / lam
        return noisy.clamp(0, 1)

# Metric helpers ----------------------------------------------------------------
try:
    from utils.metrics import compute_psnr, compute_ssim, compute_mae
except Exception:
    # Fallback metric implementations
    def compute_psnr(clean: torch.Tensor, restored: torch.Tensor) -> float:
        mse = torch.mean((clean - restored) ** 2).item()
        if mse < 1e-10:
            return 50.0
        return float(10 * np.log10(1.0 / mse))

    def compute_ssim(clean: torch.Tensor, restored: torch.Tensor) -> float:
        """Simplified SSIM on flattened tensors (structural placeholder)."""
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x = clean.mean()
        mu_y = restored.mean()
        sig_x = clean.var()
        sig_y = restored.var()
        sig_xy = ((clean - mu_x) * (restored - mu_y)).mean()
        l = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
        cs = (2 * sig_xy + c2) / (sig_x + sig_y + c2)
        return float(l * cs)

    def compute_mae(clean: torch.Tensor, restored: torch.Tensor) -> float:
        return float(torch.mean(torch.abs(clean - restored)).item())


def _load_models():
    """Instantiate both models and try to load checkpoint weights."""
    global nmrf_model, baseline_model, CHECKPOINTS_LOADED, _warnings

    if not MODELS_AVAILABLE:
        return

    device = torch.device("cpu")

    # --- Potts Model (Pure MRF) ---------------------------------------------
    try:
        potts_model = NMRFDenoiser(in_channels=3, base_channels=32, use_neural_potentials=False).to(device)
        ckpt_potts = os.path.join(PROJECT_ROOT, "checkpoints", "ablation", "ablation_potts_best.pth")
        if os.path.isfile(ckpt_potts):
            potts_model.load_state_dict(torch.load(ckpt_potts, map_location=device))
            logging.info("Loaded Potts model checkpoint: %s", ckpt_potts)
        else:
            _warnings.append("Potts model checkpoint not found – using untrained weights.")
        potts_model.eval()
    except Exception as exc:
        _warnings.append(f"Potts model init failed: {exc}")
        potts_model = None

    # --- NMRF ----------------------------------------------------------------
    try:
        nmrf_model = NMRFDenoiser(in_channels=3, base_channels=32).to(device)
        ckpt_nmrf = os.path.join(PROJECT_ROOT, "checkpoints", "nmrf_best.pth")
        if os.path.isfile(ckpt_nmrf):
            nmrf_model.load_state_dict(torch.load(ckpt_nmrf, map_location=device))
            logging.info("Loaded NMRF checkpoint: %s", ckpt_nmrf)
        else:
            _warnings.append("NMRF checkpoint not found – using untrained weights.")
        nmrf_model.eval()
    except Exception as exc:
        _warnings.append(f"NMRF model init failed: {exc}")
        nmrf_model = None

    # --- Baseline ------------------------------------------------------------
    try:
        baseline_model = ResNetDenoiser(in_channels=3, base_channels=32).to(device)
        ckpt_bl = os.path.join(PROJECT_ROOT, "checkpoints", "baseline_best.pth")
        if os.path.isfile(ckpt_bl):
            baseline_model.load_state_dict(torch.load(ckpt_bl, map_location=device))
            logging.info("Loaded Baseline checkpoint: %s", ckpt_bl)
        else:
            _warnings.append("Baseline checkpoint not found – using untrained weights.")
        baseline_model.eval()
    except Exception as exc:
        _warnings.append(f"Baseline model init failed: {exc}")
        baseline_model = None

    CHECKPOINTS_LOADED = nmrf_model is not None and baseline_model is not None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

logging.basicConfig(level=logging.INFO)

with app.app_context():
    _load_models()


# --------------- helpers ---------------------------------------------------

MAX_SIDE = 512


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image -> (1, 3, H, W) float32 tensor in [0, 1]."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor


def _tensor_to_base64(tensor: torch.Tensor, orig_h: int, orig_w: int) -> str:
    """(1,3,H,W) tensor -> base64 PNG string, cropped back to original size."""
    img = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    # Remove any padding
    img = img[:orig_h, :orig_w, :]
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_if_needed(img: Image.Image) -> Image.Image:
    """Resize so longest side <= MAX_SIDE, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= MAX_SIDE:
        return img
    scale = MAX_SIDE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _pad_to_divisible(tensor: torch.Tensor, divisor: int = 4) -> torch.Tensor:
    """Pad (right/bottom) so H and W are divisible by *divisor*."""
    _, _, h, w = tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor


# --------------- routes ----------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", warnings=_warnings)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_available": MODELS_AVAILABLE,
        "checkpoints_loaded": CHECKPOINTS_LOADED,
        "warnings": _warnings,
    })


@app.route("/denoise", methods=["POST"])
def denoise():
    try:
        # --- Validate inputs -------------------------------------------------
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        noise_type = request.form.get("noise_type", "gaussian").lower()
        noise_level = float(request.form.get("noise_level", 25))

        if noise_type not in ("gaussian", "poisson"):
            return jsonify({"error": "noise_type must be 'gaussian' or 'poisson'."}), 400

        # --- Load & pre-process image ----------------------------------------
        img = Image.open(file.stream).convert("RGB")
        img = _resize_if_needed(img)
        orig_w, orig_h = img.size  # before padding

        clean_tensor = _pil_to_tensor(img)  # (1,3,H,W) [0,1]
        padded = _pad_to_divisible(clean_tensor, 4)

        # --- Add noise -------------------------------------------------------
        with torch.no_grad():
            if noise_type == "gaussian":
                noisy = add_gaussian_noise(padded, sigma=noise_level)
            else:
                noisy = add_poisson_noise(padded, lam=noise_level)

            # --- Denoise with both models ------------------------------------
            if nmrf_model is not None:
                nmrf_out = nmrf_model(noisy)
            else:
                nmrf_out = noisy  # passthrough

            if baseline_model is not None:
                baseline_out = baseline_model(noisy)
            else:
                baseline_out = noisy

            if potts_model is not None:
                potts_out = potts_model(noisy)
            else:
                potts_out = noisy

        # --- Metrics (computed on the padded region for consistency) ----------
        metrics = {}
        for label, restored in [("nmrf", nmrf_out), ("potts", potts_out), ("baseline", baseline_out)]:
            metrics[label] = {
                "psnr": round(compute_psnr(padded, restored), 2),
                "ssim": round(compute_ssim(padded, restored), 4),
                "mae": round(compute_mae(padded, restored), 4),
            }

        # Also compute noisy metrics for reference
        metrics["noisy"] = {
            "psnr": round(compute_psnr(padded, noisy), 2),
            "ssim": round(compute_ssim(padded, noisy), 4),
            "mae": round(compute_mae(padded, noisy), 4),
        }

        # --- Encode images to base64 ----------------------------------------
        result = {
            "original": _tensor_to_base64(padded, orig_h, orig_w),
            "noisy": _tensor_to_base64(noisy, orig_h, orig_w),
            "nmrf_denoised": _tensor_to_base64(nmrf_out, orig_h, orig_w),
            "potts_denoised": _tensor_to_base64(potts_out, orig_h, orig_w),
            "baseline_denoised": _tensor_to_base64(baseline_out, orig_h, orig_w),
            "metrics": metrics,
            "warnings": _warnings,
        }
        return jsonify(result)

    except Exception as exc:
        logging.exception("Denoise endpoint error")
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
