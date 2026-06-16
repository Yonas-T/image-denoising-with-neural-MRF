"""Main training entry point for image denoising models.

Usage::

    # Train NMRF denoiser
    python train.py --model nmrf --epochs 50 --noise-sigma 25

    # Train ResNet baseline
    python train.py --model baseline --epochs 50 --noise-sigma 25

    # Quick smoke test (5 epochs)
    python train.py --model nmrf --epochs 5 --quick-test
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from models import NMRFDenoiser, ResNetDenoiser
from data.dataset import DenoisingDataset, DenoisingTestDataset
from data.download import setup_datasets
from training.trainer import Trainer
from utils.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(
        description="Train an image denoising model."
    )
    parser.add_argument(
        "--model",
        choices=["nmrf", "baseline"],
        default="nmrf",
        help="Model to train: 'nmrf' (Neural MRF) or 'baseline' (ResNet).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--noise-type", default="gaussian", choices=["gaussian", "poisson"])
    parser.add_argument("--noise-sigma", type=float, default=25, help="Gaussian noise level (0-255)")
    parser.add_argument("--poisson-lambda", type=float, default=30.0, help="Poisson noise intensity")
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--quick-test", action="store_true", help="Quick 5-epoch test run")
    parser.add_argument("--download", action="store_true", help="Download datasets before training")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Download datasets if requested or if they don't exist
    train_dir = os.path.join(args.data_root, "BSDS500", "train")
    test_dir = os.path.join(args.data_root, "BSDS500", "test")

    if args.download or not os.path.isdir(train_dir):
        print("Setting up datasets...")
        setup_datasets(args.data_root)

    if not os.path.isdir(train_dir):
        print(f"[ERROR] Training data not found at {train_dir}")
        print("Run: python train.py --download")
        sys.exit(1)

    # Create datasets
    train_ds = DenoisingDataset(
        train_dir,
        patch_size=args.patch_size,
        noise_type=args.noise_type,
        noise_sigma=args.noise_sigma,
        poisson_lambda=args.poisson_lambda,
        patches_per_image=8,
    )
    test_ds = DenoisingTestDataset(
        test_dir,
        noise_type=args.noise_type,
        noise_sigma=args.noise_sigma,
        poisson_lambda=args.poisson_lambda,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0
    )

    print(f"Training set: {len(train_ds)} patches from {train_dir}")
    print(f"Test set: {len(test_ds)} images from {test_dir}")

    # Create model
    if args.model == "nmrf":
        model = NMRFDenoiser(
            in_channels=3, base_channels=32, num_iterations=4,
            window_size=5, num_heads=4, num_groups=8,
            use_self_edges=True, use_neural_potentials=True,
        )
        model_name = "nmrf"
    else:
        model = ResNetDenoiser(in_channels=3, base_channels=32)
        model_name = "baseline"

    if args.noise_type == "poisson":
        model_name = f"{model_name}_poisson"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({total_params:,} parameters)")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        model_name=model_name,
    )
    history = trainer.train()

    # Plot training curves
    os.makedirs("results", exist_ok=True)
    plot_training_curves(
        history["train_loss"],
        history["val_psnr"],
        save_path=f"results/{model_name}_training_curves.png",
    )
    print(f"Training curves saved to results/{model_name}_training_curves.png")


if __name__ == "__main__":
    main()
