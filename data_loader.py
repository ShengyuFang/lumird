from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _find_existing_dir(root: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    return None


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def scan_paired_images(root: str, low_candidates: List[str], high_candidates: List[str]) -> List[Tuple[str, str]]:
    root = Path(root)
    low_dir = _find_existing_dir(root, low_candidates)
    high_dir = _find_existing_dir(root, high_candidates)
    if low_dir is None or high_dir is None:
        raise FileNotFoundError(
            f"Could not find paired folders under {root}. Expected low in {low_candidates}, high in {high_candidates}."
        )

    low_imgs = _list_images(low_dir)
    high_imgs = _list_images(high_dir)
    high_map: Dict[str, Path] = {p.stem: p for p in high_imgs}

    pairs = []
    for lp in low_imgs:
        hp = high_map.get(lp.stem)
        if hp is None:
            for k, v in high_map.items():
                if k in lp.stem or lp.stem in k:
                    hp = v
                    break
        if hp is not None:
            pairs.append((str(lp), str(hp)))

    if not pairs:
        raise RuntimeError(f"No paired images found under {root}")
    return pairs


class PairedLowLightDataset(Dataset):
    def __init__(self, root: str, low_candidates: List[str], high_candidates: List[str], image_size=(256, 256), train: bool = True):
        self.pairs = scan_paired_images(root, low_candidates, high_candidates)
        self.train = train
        if train:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        low_path, high_path = self.pairs[index]
        low = Image.open(low_path).convert("RGB")
        high = Image.open(high_path).convert("RGB")
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        torch.manual_seed(seed)
        low = self.transform(low)
        torch.manual_seed(seed)
        high = self.transform(high)
        return {"low": low, "high": high, "low_path": low_path, "high_path": high_path}


def build_dataloaders(cfg):
    train_ds = PairedLowLightDataset(
        root=cfg.train_root,
        low_candidates=cfg.low_dir_candidates,
        high_candidates=cfg.high_dir_candidates,
        image_size=cfg.image_size,
        train=True,
    )
    val_ds = PairedLowLightDataset(
        root=cfg.val_root,
        low_candidates=cfg.low_dir_candidates,
        high_candidates=cfg.high_dir_candidates,
        image_size=cfg.image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
