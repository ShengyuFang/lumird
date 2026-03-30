import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(ch, ch, act=True),
            ConvBlock(ch, ch, act=False),
        )

    def forward(self, x):
        return x + self.net(x)


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


class FrequencyEnhancer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.fft2(x, norm="ortho")
        mag = torch.abs(freq)
        phase = torch.angle(freq)
        mag = mag * (1.0 + self.scale) + self.bias
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.fft.ifft2(torch.complex(real, imag), norm="ortho").real


class MultiBranchOffsetSampler(nn.Module):
    def __init__(self, channels: int, branches: int = 4, max_offset_pixels: float = 2.0):
        super().__init__()
        self.branches = branches
        self.max_offset_pixels = max_offset_pixels
        self.offset_head = nn.Conv2d(channels, branches * 2, 3, 1, 1)
        self.weight_head = nn.Conv2d(channels, branches, 3, 1, 1)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        offsets = torch.tanh(self.offset_head(guide))
        weights = F.softmax(self.weight_head(guide), dim=1)

        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

        out = 0.0
        px_to_norm_x = 2.0 / max(w - 1, 1)
        px_to_norm_y = 2.0 / max(h - 1, 1)

        for i in range(self.branches):
            dx = offsets[:, 2 * i:2 * i + 1] * self.max_offset_pixels * px_to_norm_x
            dy = offsets[:, 2 * i + 1:2 * i + 2] * self.max_offset_pixels * px_to_norm_y
            branch_grid = base_grid.clone()
            branch_grid[..., 0] = branch_grid[..., 0] + dx.squeeze(1)
            branch_grid[..., 1] = branch_grid[..., 1] + dy.squeeze(1)
            sampled = F.grid_sample(x, branch_grid, mode="bilinear", padding_mode="border", align_corners=True)
            out = out + sampled * weights[:, i:i + 1]
        return out


class ResidualFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gate = self.fuse(torch.cat([x, residual], dim=1))
        return x + gate * residual


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * (-scale))
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


class GatedConditionFusion(nn.Module):
    def __init__(self, feat_ch: int, cond_ch: int):
        super().__init__()
        self.cond_proj = nn.Conv2d(cond_ch, feat_ch, 1, 1, 0)
        self.coarse_proj = nn.Conv2d(3, feat_ch, 3, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch * 3, feat_ch, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, noisy_residual: torch.Tensor, cond: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(cond)
        coarse = self.coarse_proj(coarse)
        gate = self.gate(torch.cat([noisy_residual, cond, coarse], dim=1))
        return noisy_residual + gate * (cond + coarse)


class TinyDenoiser(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.in_proj = nn.Conv2d(3 + channels, channels, 3, 1, 1)
        self.time_mlp = SinusoidalTimeEmbedding(channels)
        self.body = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )
        self.out = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, noisy_residual: torch.Tensor, fused_cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_residual, fused_cond], dim=1)
        x = self.in_proj(x)
        x = x + self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = self.body(x)
        return self.out(x)
