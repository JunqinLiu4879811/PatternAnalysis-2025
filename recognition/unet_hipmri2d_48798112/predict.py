# predict.py —— evaluate & save overlay images
import argparse, os, numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from recognition.unet_hipmri2d_48798112.dataset import HipMRI2DSlices
from recognition.unet_hipmri2d_48798112.modules import UNet

def dice_from_probs(probs, target, eps=1e-6):
    target = (target > 0.5).float()
    inter  = (probs * target).sum(dim=(1,2,3))
    denom  = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (2.0 * inter / denom)

def save_overlay(img_t, msk_t, pred_t, path):
    img = img_t.squeeze().cpu().numpy()
    msk = (msk_t.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    prd = (pred_t.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap="gray")
    plt.contour(msk, levels=[0.5], linewidths=1.0)
    plt.contour(prd, levels=[0.5], linewidths=1.0, linestyles="--")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prostate-id", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="runs/preds")
    ap.add_argument("--save-overlay", action="store_true")
    ap.add_argument("--save-n", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = HipMRI2DSlices(args.data, split=args.split,
                        target_size=(args.size, args.size),
                        prostate_id=args.prostate_id)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    net = UNet(in_ch=1, out_ch=1).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    dices = []
    save_dir = os.path.join(args.out, "overlays")
    if args.save_overlay:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        idx = 0
        for img, msk in ld:
            img = img.to(device); msk = msk.to(device)
            logits = net(img)
            probs = torch.sigmoid(logits)
            dices.append(dice_from_probs(probs, msk))
            if args.save_overlay:
                for b in range(img.size(0)):
                    if idx >= args.save_n: break
                    save_overlay(img[b], msk[b], probs[b] > args.threshold,
                                 os.path.join(save_dir, f"overlay_{idx:03d}.png"))
                    idx += 1

    dices = torch.cat(dices).mean().item()
    print(f"Test mean Dice = {dices:.4f}")

if __name__ == "__main__":
    main()
