# Neural Markov Random Field for Image Denoising

**Authors:** Yonas Tadesse and Nebiyu Daniel  
**Course:** Probabilistic Graphical Models — MSc Program

---

## Overview

This project implements and extends the **Neural Markov Random Field (NMRF)** framework from [Guan et al. (CVPR 2024)](https://github.com/aeolusguan/NMRF) for **image denoising**. The original framework was designed for stereo matching; we adapt it to denoising by:

- Treating each pixel as a node in an MRF graph
- Learning **unary potentials** (data likelihood) and **pairwise potentials** (spatial consistency) with neural networks
- Performing **neural message passing** via multi-head attention with relative positional encoding
- Using **residual learning** to predict and subtract noise

The project includes:
- **Hybrid NMRF Denoiser** — Neural MRF with learned potentials and message passing
- **Pure ResNet Baseline** — Standard encoder-decoder without graphical model components
- **Ablation Studies** — Neural vs Potts potentials, self-edges, message-passing iterations
- **Web Demo** — Flask app for interactive denoising comparison

---

## Architecture

```
Noisy Image → [Feature Backbone (ResNet)] → Multi-scale Features
                                               ↓
                                    [Neural MRF Message Passing]
                                    (Neighbor + Self-edge Attention)
                                               ↓
                                    [Reconstruction Head (U-Net Decoder)]
                                               ↓
                                    Noise Residual → Denoised Image
```
