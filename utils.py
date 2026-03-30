import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def image_gradient(x: torch.Tensor) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(x.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(x.size(1), 1, 1, 1)
    gx = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    gy = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def laplacian_filter(x: torch.Tensor) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


def structure_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(image_gradient(pred), image_gradient(target)) + F.l1_loss(laplacian_filter(pred), laplacian_filter(target))


def compute_total_loss(outputs: Dict[str, torch.Tensor], gt: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    coarse_loss = F.l1_loss(outputs["coarse"], gt)
    recon_loss = charbonnier_loss(outputs["final"], gt, eps=cfg.charbonnier_eps)
    struct_loss = structure_consistency_loss(outputs["final"], gt)
    noise_loss = F.mse_loss(outputs["pred_noise"], outputs["noise"])
    total = cfg.lambda_coarse * coarse_loss + cfg.lambda_recon * recon_loss + cfg.lambda_structure * struct_loss + cfg.lambda_noise * noise_loss
    return {
        "loss": total,
        "coarse_loss": coarse_loss,
        "recon_loss": recon_loss,
        "struct_loss": struct_loss,
        "noise_loss": noise_loss,
    }


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse == 0:
        return 99.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def save_tensor_image(x: torch.Tensor, path: str):
    ensure_dir(os.path.dirname(path))
    save_image(x.clamp(0, 1), path)


def build_optimizer_scheduler(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    def lr_lambda(epoch: int):
        if epoch < cfg.warmup_epochs:
            return float(epoch + 1) / float(max(cfg.warmup_epochs, 1))
        progress = (epoch - cfg.warmup_epochs) / float(max(cfg.epochs - cfg.warmup_epochs, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = cfg.min_lr / cfg.lr
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
