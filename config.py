from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    train_root: str = "./data/LOL/train"
    val_root: str = "./data/LOL/test"
    low_dir_candidates: List[str] = field(default_factory=lambda: ["low", "input", "lq", "dark"])
    high_dir_candidates: List[str] = field(default_factory=lambda: ["high", "gt", "target", "normal"])
    image_size: Tuple[int, int] = (256, 256)

    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 300
    warmup_epochs: int = 20
    lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    seed: int = 42
    amp: bool = True

    in_channels: int = 3
    base_channels: int = 64
    imum_channels: int = 64
    num_deform_branches: int = 4
    max_offset_pixels: float = 2.0
    cond_dim: int = 64
    diffusion_steps: int = 8
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    lambda_coarse: float = 1.0
    lambda_recon: float = 1.0
    lambda_structure: float = 0.2
    lambda_noise: float = 1.0
    charbonnier_eps: float = 1e-3

    device: str = "cuda"
    save_dir: str = "./runs/lumird"
    save_every: int = 10
    log_interval: int = 20

    sample_steps: int = 8
    fast_infer: bool = True


def get_default_config() -> Config:
    return Config()
