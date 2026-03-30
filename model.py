from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from modules import ConvBlock, ResidualBlock, FrequencyEnhancer, MultiBranchOffsetSampler, ResidualFusion, image_gradient, laplacian_filter, GatedConditionFusion, TinyDenoiser


class ResNet34Encoder(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        backbone = resnet34(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.out_proj = nn.Sequential(
            nn.Conv2d(128, out_channels, 1, 1, 0),
            nn.GELU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_proj(x)


class IMUM(nn.Module):
    def __init__(self, in_channels: int = 64, feat_channels: int = 64, branches: int = 4, max_offset_pixels: float = 2.0):
        super().__init__()
        self.project = nn.Conv2d(in_channels, feat_channels, 1, 1, 0)
        self.freq = FrequencyEnhancer(feat_channels)
        self.merge = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels, 3, 1, 1),
            nn.GELU(),
            ResidualBlock(feat_channels),
        )
        self.sampler = MultiBranchOffsetSampler(feat_channels, branches=branches, max_offset_pixels=max_offset_pixels)
        self.fusion = ResidualFusion(feat_channels)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.project(feat)
        grad = image_gradient(x)
        freq = self.freq(x)
        guide = self.merge(torch.cat([x, grad, freq], dim=1))
        sampled = self.sampler(x, guide)
        return self.fusion(x, sampled - x)


class RDM(nn.Module):
    def __init__(self, feat_channels: int = 64, cond_dim: int = 64, diffusion_steps: int = 8, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.coarse_head = nn.Sequential(
            ConvBlock(feat_channels, feat_channels),
            nn.Conv2d(feat_channels, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.struct_proj = nn.Conv2d(feat_channels, cond_dim, 1, 1, 0)
        self.cond_fuse = nn.Sequential(
            nn.Conv2d(3 * 2 + cond_dim, cond_dim, 3, 1, 1),
            nn.GELU(),
            ResidualBlock(cond_dim),
        )
        self.gated_fusion = GatedConditionFusion(feat_ch=cond_dim, cond_ch=cond_dim)
        self.noise_proj = nn.Conv2d(3, cond_dim, 3, 1, 1)
        self.denoiser = TinyDenoiser(channels=cond_dim)
        self.output_mod = nn.Sequential(
            nn.Conv2d(3 + cond_dim, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def build_condition(self, low: torch.Tensor, exposure_feat: torch.Tensor) -> torch.Tensor:
        high_freq = laplacian_filter(low)
        grad = image_gradient(low)
        struct = self.struct_proj(exposure_feat)
        struct = F.interpolate(struct, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.cond_fuse(torch.cat([high_freq, grad, struct], dim=1))

    def q_sample(self, residual: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(a_bar) * residual + torch.sqrt(1.0 - a_bar) * noise

    def predict_noise(self, noisy_residual: torch.Tensor, cond: torch.Tensor, coarse: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noisy_feat = self.noise_proj(noisy_residual)
        fused = self.gated_fusion(noisy_feat, cond, coarse)
        return self.denoiser(noisy_residual, fused, t)

    def reconstruct_residual(self, noisy_residual: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return (noisy_residual - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar + 1e-8)

    def forward_train(self, low: torch.Tensor, exposure_feat: torch.Tensor, gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse = self.coarse_head(exposure_feat)
        cond = self.build_condition(low, exposure_feat)
        target_residual = gt - coarse
        noise = torch.randn_like(target_residual)
        t = torch.randint(0, self.diffusion_steps, (low.size(0),), device=low.device)
        noisy_residual = self.q_sample(target_residual, t, noise)
        pred_noise = self.predict_noise(noisy_residual, cond, coarse, t)
        pred_residual = self.reconstruct_residual(noisy_residual, pred_noise, t)
        final = torch.clamp(coarse + self.output_mod(torch.cat([pred_residual, cond], dim=1)), 0.0, 1.0)
        return {
            "coarse": coarse,
            "cond": cond,
            "target_residual": target_residual,
            "noise": noise,
            "t": t,
            "noisy_residual": noisy_residual,
            "pred_noise": pred_noise,
            "pred_residual": pred_residual,
            "final": final,
        }

    @torch.no_grad()
    def forward_infer(self, low: torch.Tensor, exposure_feat: torch.Tensor, fast: bool = True, sample_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        coarse = self.coarse_head(exposure_feat)
        cond = self.build_condition(low, exposure_feat)
        steps = sample_steps or self.diffusion_steps
        if fast:
            x = torch.zeros_like(coarse)
            t = torch.full((low.size(0),), self.diffusion_steps - 1, device=low.device, dtype=torch.long)
            pred_noise = self.predict_noise(x, cond, coarse, t)
            pred_residual = self.reconstruct_residual(x, pred_noise, t)
        else:
            x = torch.randn_like(coarse)
            for i in reversed(range(min(steps, self.diffusion_steps))):
                t = torch.full((low.size(0),), i, device=low.device, dtype=torch.long)
                pred_noise = self.predict_noise(x, cond, coarse, t)
                x = self.reconstruct_residual(x, pred_noise, t)
            pred_residual = x
        final = torch.clamp(coarse + self.output_mod(torch.cat([pred_residual, cond], dim=1)), 0.0, 1.0)
        return {"coarse": coarse, "cond": cond, "pred_residual": pred_residual, "final": final}


class LumiRD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ResNet34Encoder(out_channels=cfg.base_channels)
        self.imum = IMUM(
            in_channels=cfg.base_channels,
            feat_channels=cfg.imum_channels,
            branches=cfg.num_deform_branches,
            max_offset_pixels=cfg.max_offset_pixels,
        )
        self.rdm = RDM(
            feat_channels=cfg.imum_channels,
            cond_dim=cfg.cond_dim,
            diffusion_steps=cfg.diffusion_steps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )
        self.cfg = cfg

    def forward(self, low: torch.Tensor, gt: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat = self.encoder(low)
        exposure_feat = self.imum(feat)
        if self.training and gt is not None:
            out = self.rdm.forward_train(low, exposure_feat, gt)
        else:
            out = self.rdm.forward_infer(low, exposure_feat, fast=self.cfg.fast_infer, sample_steps=self.cfg.sample_steps)
        out["exposure_feat"] = exposure_feat
        return out
