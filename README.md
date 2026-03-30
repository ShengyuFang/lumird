# Introduction

This repository provides the PyTorch implementation of our low-light image enhancement framework **Luminance Residual Diffusion Network (LumiRD)**.

The proposed method follows a two-stage paradigm for low-light image enhancement: **degradation representation refinement** and **residual-based progressive restoration**. Specifically, LumiRD first employs an **Illumination Manifold Unfolding Module (IMUM)** to jointly model spatial, gradient, and frequency-domain information, so that suppressed responses in dark regions can be adaptively unfolded into more stable exposure-aligned features. Based on these refined features, a **Residual Diffusion Module (RDM)** further performs progressive restoration in the luminance residual space, which helps reduce pseudo-details, texture drift, and over-enhancement caused by direct full-image generation.

If you use this code or find it helpful in your research, please cite the corresponding paper.

**Paper title:**

> **Progressive Luminance Residual Diffusion with Manifold-Based Representation Refinement for Low-Light Image Enhancement**

---

# Dependencies

The implementation is based on the experimental setup described in the manuscript and the current released code version.

- Python 3.10.13
- PyTorch 2.1.0
- torchvision
- numpy
- Pillow
- tqdm

Recommended hardware environment:

- Windows 11 / Linux
- NVIDIA GPU with CUDA support
- 24 GB GPU memory is recommended for the default training setting

---

# Data Preparation

The method is evaluated on three public low-light image enhancement datasets:

- **LOL**
- **SICE-Part2**
- **LSRW**

Please download the corresponding datasets from their official or public sources and organize them into paired low/normal image folders.

The current dataloader searches paired images under the dataset root using the following candidate folder names:

- Low-light images: `low`, `input`, `lq`, `dark`
- Ground-truth / normal-light images: `high`, `gt`, `target`, `normal`

An example directory structure is shown below:

```text
./data/LOL/
├── train/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

You may also use other folder names, as long as they match the supported candidates defined in `config.py`.

---

# Environment Preparation

1. Create a Python environment.
2. Install PyTorch and torchvision according to your CUDA version.
3. Install the remaining dependencies:

```bash
pip install numpy pillow tqdm
```

If you want to reproduce the default setting in the manuscript, you may keep the default hyperparameters already provided in `config.py`, including:

- input size: `256 × 256`
- batch size: `8`
- training epochs: `300`
- optimizer: `AdamW`
- initial learning rate: `2e-4`
- weight decay: `1e-4`
- warm-up epochs: `20`
- diffusion steps: `8`

---

# Usage

The current codebase provides an integrated training and validation pipeline in `train.py`.

## 1. Start training

```bash
python train.py --train_root ./data/LOL/train --val_root ./data/LOL/test --save_dir ./runs/lumird_lol
```

## 2. Train with custom settings

```bash
python train.py \
  --train_root ./data/SICE/train \
  --val_root ./data/SICE/test \
  --save_dir ./runs/lumird_sice \
  --epochs 300 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda
```

During training, the framework will:

- train LumiRD end-to-end,
- perform validation after each epoch,
- save checkpoints to `save_dir/checkpoints`,
- optionally save visualization results to `save_dir/visuals`.

## 3. Main files

- `config.py`: training and model hyperparameters
- `data_loader.py`: paired low-light dataset loader
- `modules.py`: core building blocks used by IMUM and RDM
- `model.py`: complete LumiRD architecture
- `utils.py`: loss functions, metrics, optimizer, and helper functions
- `train.py`: training and validation script

---

# Method Overview

## Illumination Manifold Unfolding Module (IMUM)

IMUM is designed to refine degraded feature representations before enhancement. It projects backbone features into a latent space, integrates spatial responses, local gradients, and frequency-domain information, and then performs adaptive offset sampling and residual fusion. This stage helps improve structural expressiveness and feature discriminability in dark regions.

## Residual Diffusion Module (RDM)

RDM does not directly reconstruct the whole enhanced image through diffusion. Instead, it first predicts a coarse enhancement result and then progressively restores the missing components in the **residual space** under the guidance of high-frequency cues, gradient information, and structural projections. This design makes refinement more controllable and helps suppress pseudo-textures and detail drift.

---

# Notes

- The current implementation is written to match the method logic and experimental setting described in the manuscript as closely as possible.
- The validation process is integrated into `train.py`; there is no separate `test.py` file in this release.
- The deformable resampling component is implemented in a stable approximation form for easier reproduction across environments.

---

# Citation

If you use this repository in your research, please cite the paper as follows:

```text
[1] Progressive Luminance Residual Diffusion with Manifold-Based Representation Refinement for Low-Light Image Enhancement.
```

You may replace this placeholder with your final publication information after the paper is officially published.

---

# Acknowledgement

This codebase is developed based on the method design described in the manuscript and organized as a lightweight research implementation for reproduction and further extension.
