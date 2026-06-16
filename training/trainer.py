"""Training loop for image denoising models.

Provides a :class:`Trainer` that handles the full training lifecycle:
optimizer, one-cycle LR scheduler, L1 loss, checkpointing, and
per-epoch validation with PSNR/SSIM/MAE.
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_psnr, compute_ssim, compute_mae


class Trainer:
    """Complete training loop for a denoising model.

    Args:
        model: The denoising model (NMRFDenoiser or ResNetDenoiser).
        train_loader: DataLoader yielding ``(noisy, clean)`` patch pairs.
        val_loader: DataLoader yielding ``(noisy, clean, filename)`` tuples.
        device: ``'cpu'`` or ``'cuda'``.
        lr: Peak learning rate for one-cycle scheduler.
        num_epochs: Total training epochs.
        checkpoint_dir: Where to save model checkpoints.
        model_name: Name prefix for checkpoint files.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        lr: float = 5e-4,
        num_epochs: int = 50,
        checkpoint_dir: str = "./checkpoints",
        model_name: str = "model",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )

        self.best_psnr = 0.0
        self.history = {"train_loss": [], "val_psnr": [], "val_ssim": [], "val_mae": []}

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs} [Train]",
            leave=False,
        )
        for noisy, clean in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            denoised = self.model(noisy)
            loss = self.criterion(denoised, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on the validation/test set.

        Returns:
            Dict with ``'psnr'``, ``'ssim'``, ``'mae'`` averages.
        """
        if self.val_loader is None:
            return {"psnr": 0.0, "ssim": 0.0, "mae": 0.0}

        self.model.eval()
        psnrs, ssims, maes = [], [], []

        for batch in self.val_loader:
            # Test dataset returns (noisy, clean, filename)
            if len(batch) == 3:
                noisy, clean, _ = batch
            else:
                noisy, clean = batch

            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # Pad to divisible by 4
            _, _, h, w = noisy.shape
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            if pad_h or pad_w:
                noisy = nn.functional.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")

            denoised = self.model(noisy)

            # Unpad
            if pad_h or pad_w:
                denoised = denoised[:, :, :h, :w]

            # Compute metrics per sample
            for i in range(clean.shape[0]):
                c = clean[i].cpu().numpy()
                d = denoised[i].cpu().numpy()
                psnrs.append(compute_psnr(c, d))
                ssims.append(compute_ssim(c, d))
                maes.append(compute_mae(c, d))

        return {
            "psnr": sum(psnrs) / max(len(psnrs), 1),
            "ssim": sum(ssims) / max(len(ssims), 1),
            "mae": sum(maes) / max(len(maes), 1),
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Training history dict.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            avg_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(avg_loss)

            # Validate
            metrics = self.validate()
            self.history["val_psnr"].append(metrics["psnr"])
            self.history["val_ssim"].append(metrics["ssim"])
            self.history["val_mae"].append(metrics["mae"])

            # Print progress
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{self.num_epochs} │ "
                f"Loss: {avg_loss:.5f} │ "
                f"PSNR: {metrics['psnr']:.2f} dB │ "
                f"SSIM: {metrics['ssim']:.4f} │ "
                f"MAE: {metrics['mae']:.4f} │ "
                f"Time: {elapsed:.0f}s"
            )

            # Save best model
            if metrics["psnr"] > self.best_psnr:
                self.best_psnr = metrics["psnr"]
                best_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_best.pth"
                )
                torch.save(self.model.state_dict(), best_path)
                print(f"  ► New best PSNR: {self.best_psnr:.2f} dB — saved to {best_path}")

            # Save periodic checkpoint
            if epoch % 10 == 0 or epoch == self.num_epochs:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.model_name}_epoch{epoch}.pth"
                )
                self.save_checkpoint(ckpt_path, epoch, metrics)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.0f}s.  Best PSNR: {self.best_psnr:.2f} dB")
        return self.history
