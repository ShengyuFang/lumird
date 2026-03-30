import argparse
import os

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import get_default_config
from data_loader import build_dataloaders
from model import LumiRD
from utils import seed_everything, AverageMeter, compute_total_loss, psnr, ensure_dir, save_checkpoint, save_tensor_image, build_optimizer_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train LumiRD")
    parser.add_argument("--train_root", type=str, default=None)
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def override_cfg(cfg, args):
    for key in ["train_root", "val_root", "save_dir", "epochs", "batch_size", "num_workers", "device"]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def train_one_epoch(model, loader, optimizer, scaler, cfg, device, epoch):
    model.train()
    meter = AverageMeter()
    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.amp and device.type == "cuda"):
            outputs = model(low, high)
            losses = compute_total_loss(outputs, high, cfg)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), low.size(0))
        if step % cfg.log_interval == 0:
            pbar.set_postfix({"loss": f"{meter.avg:.4f}", "recon": f"{losses['recon_loss'].item():.4f}", "noise": f"{losses['noise_loss'].item():.4f}"})
    return meter.avg


@torch.no_grad()
def validate(model, loader, cfg, device, save_vis_dir=None):
    model.eval()
    meter = AverageMeter()
    for idx, batch in enumerate(tqdm(loader, desc="Validate", leave=False)):
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        outputs = model(low)
        pred = outputs["final"]
        meter.update(psnr(pred, high), 1)
        if save_vis_dir is not None and idx < 10:
            save_tensor_image(pred, os.path.join(save_vis_dir, f"{idx:03d}_pred.png"))
            save_tensor_image(low, os.path.join(save_vis_dir, f"{idx:03d}_low.png"))
            save_tensor_image(high, os.path.join(save_vis_dir, f"{idx:03d}_gt.png"))
    return meter.avg


def main():
    args = parse_args()
    cfg = override_cfg(get_default_config(), args)
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg.save_dir)
    ensure_dir(os.path.join(cfg.save_dir, "checkpoints"))
    ensure_dir(os.path.join(cfg.save_dir, "visuals"))

    train_loader, val_loader = build_dataloaders(cfg)
    model = LumiRD(cfg).to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")

    best_psnr = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch)
        val_psnr = validate(model, val_loader, cfg, device, save_vis_dir=os.path.join(cfg.save_dir, "visuals", f"epoch_{epoch:03d}") if epoch % cfg.save_every == 0 else None)
        scheduler.step()
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_psnr={val_psnr:.4f}")
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "cfg": vars(cfg),
        }
        if epoch % cfg.save_every == 0:
            save_checkpoint(ckpt, os.path.join(cfg.save_dir, "checkpoints", f"epoch_{epoch:03d}.pth"))
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt["best_psnr"] = best_psnr
            save_checkpoint(ckpt, os.path.join(cfg.save_dir, "checkpoints", "best.pth"))
    print(f"Training finished. Best PSNR: {best_psnr:.4f}")


if __name__ == "__main__":
    main()
