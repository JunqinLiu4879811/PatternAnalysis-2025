<<<<<<< HEAD
import argparse, os, math, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from recognition.unet_hipmri2d_48798112.dataset import HipMRI2DSlices
from recognition.unet_hipmri2d_48798112.modules import UNet


def dice_score_from_probs(probs, target, thr=0.5, eps=1e-6):
    """Compute mean Dice over a batch given sigmoid probabilities and binary target."""
    pred = (probs > thr).float()
    t = target.float()
    inter = (pred * t).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps
    return (2 * inter / denom).mean().item()


def save_mask_png(mask2d: np.ndarray, path: str):
    """Save a binary/float mask [H,W] as PNG using matplotlib (no extra deps)."""
    plt.figure(figsize=(4, 4))
    plt.imshow(mask2d, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay_png(image2d: np.ndarray, mask2d: np.ndarray, path: str, alpha: float = 0.35):
    """Gray image + semi-transparent mask overlay."""
    plt.figure(figsize=(4, 4))
    plt.imshow(image2d, cmap="gray")
    plt.imshow(mask2d, alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="UNet prediction on HipMRI 2D slices.")
    ap.add_argument("--data", required=True, help="HipMRI root, e.g., /home/groups/comp3710/HipMRI_Study_open")
    ap.add_argument("--weights", required=True, help="Path to checkpoint, e.g., outputs/best.ckpt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=256, help="resize to (size,size)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prostate-id", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="pred_out")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-mask-png", action="store_true", help="Save predicted mask PNGs")
    ap.add_argument("--save-overlay", action="store_true", help="Save gray+mask overlay PNGs")
    ap.add_argument("--save-n", type=int, default=24, help="Max number of samples to save as PNGs (evenly spaced)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.save_mask_png:
        os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataset / Loader
    ds = HipMRI2DSlices(
        root=args.data,
        split=args.split,
        target_size=(args.size, args.size),
        prostate_id=args.prostate_id,
    )

    # Wrap to keep index for file naming
    class WithIndex(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, msk = self.base[i]
            return img, msk, i

    loader = DataLoader(WithIndex(ds), batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    net = UNet(in_ch=1, num_classes=1).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state, strict=True)
    net.eval()

    # Choose indices to save as PNGs (evenly spaced)
    save_idx_set = set()
    if args.save_mask_png or args.save_overlay:
        if len(ds) <= args.save_n:
            save_idx_set = set(range(len(ds)))
        else:
            step = max(1, math.floor(len(ds) / args.save_n))
            save_idx_set = set(range(0, len(ds), step))

    # Inference loop (+ optional Dice)
    dice_sum, dice_cnt = 0.0, 0
    with torch.no_grad():
        for img, msk, idx in loader:
            img = img.to(device)
            msk = msk.to(device)

            logits = net(img)
            probs = torch.sigmoid(logits)

            # Metric (since ground truth exists)
            dice_sum += dice_score_from_probs(probs, msk, thr=args.threshold) * img.size(0)
            dice_cnt += img.size(0)

            # Optional saving
            if save_idx_set:
                probs_cpu = probs.cpu().numpy()
                img_cpu = img.cpu().numpy()
                for b in range(img.size(0)):
                    i_global = int(idx[b].item())
                    if i_global not in save_idx_set:
                        continue
                    p2d = probs_cpu[b, 0]  # [H,W]
                    x2d = img_cpu[b, 0]    # [H,W]
                    if args.save_mask_png:
                        save_mask_png((p2d > args.threshold).astype(np.float32),
                                      os.path.join(args.out, "masks", f"pred_{i_global:06d}.png"))
                    if args.save_overlay:
                        save_overlay_png(x2d, (p2d > args.threshold).astype(np.float32),
                                         os.path.join(args.out, "overlays", f"overlay_{i_global:06d}.png"))

    mean_dice = dice_sum / max(dice_cnt, 1)
    print(f"[predict] split={args.split}  mean Dice (thr={args.threshold:.2f}) = {mean_dice:.4f}")
    print(f"[predict] outputs saved under: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

=======
import argparse, os, math, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from recognition.unet_hipmri2d_48798112.dataset import HipMRI2DSlices
from recognition.unet_hipmri2d_48798112.modules import UNet


def dice_score_from_probs(probs, target, thr=0.5, eps=1e-6):
    """Compute mean Dice over a batch given sigmoid probabilities and binary target."""
    pred = (probs > thr).float()
    t = target.float()
    inter = (pred * t).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps
    return (2 * inter / denom).mean().item()


def save_mask_png(mask2d: np.ndarray, path: str):
    """Save a binary/float mask [H,W] as PNG using matplotlib (no extra deps)."""
    plt.figure(figsize=(4, 4))
    plt.imshow(mask2d, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay_png(image2d: np.ndarray, mask2d: np.ndarray, path: str, alpha: float = 0.35):
    """Gray image + semi-transparent mask overlay."""
    plt.figure(figsize=(4, 4))
    plt.imshow(image2d, cmap="gray")
    plt.imshow(mask2d, alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="UNet prediction on HipMRI 2D slices.")
    ap.add_argument("--data", required=True, help="HipMRI root, e.g., /home/groups/comp3710/HipMRI_Study_open")
    ap.add_argument("--weights", required=True, help="Path to checkpoint, e.g., outputs/best.ckpt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=256, help="resize to (size,size)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prostate-id", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="pred_out")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-mask-png", action="store_true", help="Save predicted mask PNGs")
    ap.add_argument("--save-overlay", action="store_true", help="Save gray+mask overlay PNGs")
    ap.add_argument("--save-n", type=int, default=24, help="Max number of samples to save as PNGs (evenly spaced)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.save_mask_png:
        os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataset / Loader
    ds = HipMRI2DSlices(
        root=args.data,
        split=args.split,
        target_size=(args.size, args.size),
        prostate_id=args.prostate_id,
    )

    # Wrap to keep index for file naming
    class WithIndex(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            img, msk = self.base[i]
            return img, msk, i

    loader = DataLoader(WithIndex(ds), batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    net = UNet(in_ch=1, num_classes=1).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state, strict=True)
    net.eval()

    # Choose indices to save as PNGs (evenly spaced)
    save_idx_set = set()
    if args.save_mask_png or args.save_overlay:
        if len(ds) <= args.save_n:
            save_idx_set = set(range(len(ds)))
        else:
            step = max(1, math.floor(len(ds) / args.save_n))
            save_idx_set = set(range(0, len(ds), step))

    # Inference loop (+ optional Dice)
    dice_sum, dice_cnt = 0.0, 0
    with torch.no_grad():
        for img, msk, idx in loader:
            img = img.to(device)
            msk = msk.to(device)

            logits = net(img)
            probs = torch.sigmoid(logits)

            # Metric (since ground truth exists)
            dice_sum += dice_score_from_probs(probs, msk, thr=args.threshold) * img.size(0)
            dice_cnt += img.size(0)

            # Optional saving
            if save_idx_set:
                probs_cpu = probs.cpu().numpy()
                img_cpu = img.cpu().numpy()
                for b in range(img.size(0)):
                    i_global = int(idx[b].item())
                    if i_global not in save_idx_set:
                        continue
                    p2d = probs_cpu[b, 0]  # [H,W]
                    x2d = img_cpu[b, 0]    # [H,W]
                    if args.save_mask_png:
                        save_mask_png((p2d > args.threshold).astype(np.float32),
                                      os.path.join(args.out, "masks", f"pred_{i_global:06d}.png"))
                    if args.save_overlay:
                        save_overlay_png(x2d, (p2d > args.threshold).astype(np.float32),
                                         os.path.join(args.out, "overlays", f"overlay_{i_global:06d}.png"))

    mean_dice = dice_sum / max(dice_cnt, 1)
    print(f"[predict] split={args.split}  mean Dice (thr={args.threshold:.2f}) = {mean_dice:.4f}")
    print(f"[predict] outputs saved under: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
>>>>>>> c7ba1cd (feat(unet_hipmri2d): local GPU run cmds & pairing fix; docs: data path)
