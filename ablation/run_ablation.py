"""Ablation study runner for Neural MRF image denoising.

Runs three ablation studies:

1. **Neural vs Potts potentials** — learned attention vs fixed
   Gaussian-kernel pairwise potential.
2. **Self-edges** — with vs without intra-pixel competition.
3. **Message-passing iterations** — varying N_i ∈ {1, 2, 4, 6}.

Each study trains a model variant for a configurable number of
epochs, evaluates on a test set, and records PSNR / SSIM / MAE.

"""

import argparse
import os
import sys
import json

import torch
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import NMRFDenoiser
from data.dataset import DenoisingDataset, DenoisingTestDataset
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.visualization import plot_ablation_results


def _get_data_loaders(args):
    """Create train and test DataLoaders."""
    train_dir = os.path.join(args.data_root, "BSDS500", "train")
    test_dir = os.path.join(args.data_root, "BSDS500", "test")

    if not os.path.isdir(train_dir):
        print(f"[ERROR] Training data not found at {train_dir}")
        print("Run: python -c 'from data.download import setup_datasets; setup_datasets()'")
        sys.exit(1)

    train_ds = DenoisingDataset(
        train_dir,
        patch_size=64,
        noise_type=args.noise_type,
        noise_sigma=args.noise_sigma,
        patches_per_image=4,
    )
    test_ds = DenoisingTestDataset(
        test_dir,
        noise_type=args.noise_type,
        noise_sigma=args.noise_sigma,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    return train_loader, test_loader


def _train_and_eval(model, args, variant_name):
    """Train a model variant and return evaluation metrics."""
    train_loader, test_loader = _get_data_loaders(args)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=args.device,
        lr=5e-4,
        num_epochs=args.ablation_epochs,
        checkpoint_dir=os.path.join("checkpoints", "ablation"),
        model_name=variant_name,
    )
    trainer.train()

    evaluator = Evaluator(model, test_loader, device=args.device)
    results = evaluator.evaluate()
    return results["average"]


# ------------------------------------------------------------------
# Ablation 1: Neural potentials vs Potts model
# ------------------------------------------------------------------
def run_potentials_ablation(args) -> dict:
    """Compare learned neural potentials with hand-crafted Potts model."""
    print("\n" + "=" * 60)
    print("ABLATION 1: Neural Potentials vs Potts Model")
    print("=" * 60)

    results = {}

    # Neural (default)
    print("\n--- Training: Neural Potentials ---")
    model = NMRFDenoiser(use_neural_potentials=True, num_iterations=args.num_iters)
    results["Neural Potentials"] = _train_and_eval(model, args, "ablation_neural")

    # Potts
    print("\n--- Training: Potts Model ---")
    model = NMRFDenoiser(use_neural_potentials=False, num_iterations=args.num_iters)
    results["Potts Model"] = _train_and_eval(model, args, "ablation_potts")

    return results


# ------------------------------------------------------------------
# Ablation 2: Self-edges
# ------------------------------------------------------------------
def run_self_edges_ablation(args) -> dict:
    """Compare model with and without self-edge aggregation layers."""
    print("\n" + "=" * 60)
    print("ABLATION 2: Effect of Self-Edges")
    print("=" * 60)

    results = {}

    print("\n--- Training: With Self-Edges ---")
    model = NMRFDenoiser(use_self_edges=True, num_iterations=args.num_iters)
    results["With Self-Edges"] = _train_and_eval(model, args, "ablation_self_edges_on")

    print("\n--- Training: Without Self-Edges ---")
    model = NMRFDenoiser(use_self_edges=False, num_iterations=args.num_iters)
    results["No Self-Edges"] = _train_and_eval(model, args, "ablation_self_edges_off")

    return results


# ------------------------------------------------------------------
# Ablation 3: Number of message-passing iterations
# ------------------------------------------------------------------
def run_iterations_ablation(args) -> dict:
    """Test different numbers of message-passing iterations."""
    print("\n" + "=" * 60)
    print("ABLATION 3: Message-Passing Iterations")
    print("=" * 60)

    results = {}
    for n_iter in [1, 2, 4, 6]:
        print(f"\n--- Training: {n_iter} iterations ---")
        model = NMRFDenoiser(num_iterations=n_iter)
        results[f"N_i = {n_iter}"] = _train_and_eval(model, args, f"ablation_iter{n_iter}")

    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def _print_results_table(study_name: str, results: dict):
    """Pretty-print an ablation results table."""
    print(f"\n{'='*60}")
    print(f"Results: {study_name}")
    print(f"{'='*60}")
    header = f"{'Variant':<25} {'PSNR (dB)':>10} {'SSIM':>10} {'MAE':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<25} {m['psnr']:>10.2f} {m['ssim']:>10.4f} {m['mae']:>10.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--study",
        choices=["all", "potentials", "self_edges", "iterations"],
        default="all",
        help="Which ablation study to run.",
    )
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--ablation-epochs", type=int, default=20)
    parser.add_argument("--noise-sigma", type=float, default=25)
    parser.add_argument("--noise-type", default="gaussian")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-iters", type=int, default=4, help="Default NMRF iterations")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    results_dir = os.path.join("results", "ablation")
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    if args.study in ("all", "potentials"):
        r = run_potentials_ablation(args)
        _print_results_table("Neural vs Potts Potentials", r)
        all_results["potentials"] = r
        plot_ablation_results(
            r, title="Neural vs Potts Potentials",
            save_path=os.path.join(results_dir, "potentials.png"),
        )

    if args.study in ("all", "self_edges"):
        r = run_self_edges_ablation(args)
        _print_results_table("Self-Edges", r)
        all_results["self_edges"] = r
        plot_ablation_results(
            r, title="Self-Edges",
            save_path=os.path.join(results_dir, "self_edges.png"),
        )

    if args.study in ("all", "iterations"):
        r = run_iterations_ablation(args)
        _print_results_table("Message-Passing Iterations", r)
        all_results["iterations"] = r
        plot_ablation_results(
            r, title="Message-Passing Iterations",
            save_path=os.path.join(results_dir, "iterations.png"),
        )

    # Save combined results as JSON
    json_path = os.path.join(results_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"All results saved to {json_path}")


if __name__ == "__main__":
    main()
