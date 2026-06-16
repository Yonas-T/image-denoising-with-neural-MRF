"""Main evaluation entry point for trained denoising models.

Usage::

    python evaluate.py --model nmrf --checkpoint checkpoints/nmrf_best.pth --dataset bsds68
    python evaluate.py --model baseline --checkpoint checkpoints/baseline_best.pth --dataset kodak24
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from models import NMRFDenoiser, ResNetDenoiser
from data.dataset import DenoisingTestDataset
from training.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate a denoising model.")
    parser.add_argument(
        "--model", choices=["nmrf", "baseline"], default="nmrf",
        help="Model architecture.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth). If not given, uses default path.",
    )
    parser.add_argument(
        "--dataset", choices=["bsds68", "kodak24"], default="bsds68",
        help="Test dataset.",
    )
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--noise-type", default="gaussian", choices=["gaussian", "poisson"])
    parser.add_argument("--noise-sigma", type=float, default=25)
    parser.add_argument("--poisson-lambda", type=float, default=30.0)
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Resolve dataset path
    if args.dataset == "bsds68":
        test_dir = os.path.join(args.data_root, "BSDS500", "test")
    else:
        test_dir = os.path.join(args.data_root, "Kodak24")

    if not os.path.isdir(test_dir):
        print(f"[ERROR] Test data not found at {test_dir}")
        print("Run: python -c 'from data.download import setup_datasets; setup_datasets()'")
        sys.exit(1)

    # Create model
    if args.model == "nmrf":
        model = NMRFDenoiser(in_channels=3, base_channels=32)
        default_ckpt = "checkpoints/nmrf_best.pth"
    else:
        model = ResNetDenoiser(in_channels=3, base_channels=32)
        default_ckpt = "checkpoints/baseline_best.pth"

    # Load checkpoint
    ckpt_path = args.checkpoint or default_ckpt
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # Handle both raw state_dict and wrapped checkpoint
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint not found at {ckpt_path} — evaluating untrained model")

    # Create test dataset
    test_ds = DenoisingTestDataset(
        test_dir,
        noise_type=args.noise_type,
        noise_sigma=args.noise_sigma,
        poisson_lambda=args.poisson_lambda,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} images from {test_dir}")

    # Evaluate
    out_dir = os.path.join(args.output_dir, f"{args.model}_{args.dataset}")
    evaluator = Evaluator(model, test_loader, device=device)
    evaluator.evaluate_and_save(output_dir=out_dir)


if __name__ == "__main__":
    main()
