
import os
import csv
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from utils.metrics import compute_psnr, compute_ssim, compute_mae


class Evaluator:

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device

    @torch.no_grad()
    def evaluate(self) -> dict:

        per_image = []

        for noisy, clean, filenames in tqdm(self.test_loader, desc="Evaluating"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # Pad to divisible by 4
            _, _, h, w = noisy.shape
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            if pad_h or pad_w:
                noisy_pad = nn.functional.pad(noisy, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                noisy_pad = noisy

            denoised = self.model(noisy_pad)

            # Unpad
            if pad_h or pad_w:
                denoised = denoised[:, :, :h, :w]

            for i in range(clean.shape[0]):
                c = clean[i].cpu().numpy()
                d = denoised[i].cpu().numpy()
                fname = filenames[i] if isinstance(filenames, (list, tuple)) else filenames

                metrics = {
                    "filename": fname,
                    "psnr": compute_psnr(c, d),
                    "ssim": compute_ssim(c, d),
                    "mae": compute_mae(c, d),
                }
                per_image.append(metrics)

        # Average metrics
        avg = {
            "psnr": np.mean([m["psnr"] for m in per_image]),
            "ssim": np.mean([m["ssim"] for m in per_image]),
            "mae": np.mean([m["mae"] for m in per_image]),
        }

        return {"per_image": per_image, "average": avg}

    @torch.no_grad()
    def evaluate_and_save(self, output_dir: str = "./results") -> dict:

        os.makedirs(output_dir, exist_ok=True)
        results = self.evaluate()

        # Save CSV
        csv_path = os.path.join(output_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "mae"])
            writer.writeheader()
            for row in results["per_image"]:
                writer.writerow({
                    "filename": row["filename"],
                    "psnr": f"{row['psnr']:.4f}",
                    "ssim": f"{row['ssim']:.6f}",
                    "mae": f"{row['mae']:.6f}",
                })

        # Summary
        avg = results["average"]
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({len(results['per_image'])} images)")
        print(f"{'='*50}")
        print(f"  Average PSNR: {avg['psnr']:.2f} dB")
        print(f"  Average SSIM: {avg['ssim']:.4f}")
        print(f"  Average MAE:  {avg['mae']:.4f}")
        print(f"  Results saved to {csv_path}")

        return results
