# train.py —— minimal training loop (prints progress, saves best.ckpt)
import argparse, os
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

from recognition.unet_hipmri2d_48798112.dataset import HipMRI2DSlices
from recognition.unet_hipmri2d_48798112.modules import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau


def dice_loss_from_logits(logits, target, eps: float = 1e-6):
    import torch
    probs  = torch.sigmoid(logits)
    target = (target > 0.5).float()
    inter  = (probs * target).sum(dim=(1,2,3))
    denom  = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    dice   = (2.0 * inter / denom).mean()
    return 1.0 - dice

def dice_score(logits, target, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    target = (target > 0.5).float()
    inter  = (probs * target).sum(dim=(1, 2, 3))
    denom  = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (2.0 * inter / denom).mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="runs/unet256")
    ap.add_argument("--prostate-id", type=int, default=5)
    ap.add_argument("--max-train-steps", type=int, default=9999999)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-val-steps", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = HipMRI2DSlices(args.data, split="train",
                              target_size=(args.size, args.size),
                              prostate_id=args.prostate_id)
    val_ds   = HipMRI2DSlices(args.data, split="val",
                              target_size=(args.size, args.size),
                              prostate_id=args.prostate_id)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=0, pin_memory=True)

    net = UNet(in_ch=1, out_ch=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, min_lr=1e-5, verbose=True)


    best = -1.0
    print(f"device={device} train_slices={len(train_ds)} val_slices={len(val_ds)}", flush=True)

    for ep in range(1, args.epochs + 1):
        net.train()
        step = 0
        for img, msk in train_ld:
            img, msk = img.to(device), msk.to(device)
            logits = net(img)
            pos = msk.sum()
            neg = msk.numel() - pos
            pw  = torch.clamp(neg / (pos + 1e-6), min=1., max=50.).to(msk.device)  # pos_weight
            bce = F.binary_cross_entropy_with_logits(logits, msk, pos_weight=pw)
            dl  = dice_loss_from_logits(logits, msk)
            loss = 0.7 * bce + 0.3 * dl
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"[epoch {ep}] step {step} loss={loss.item():.4f}", flush=True)
            if step >= args.max_train_steps:
                break

        # validation (capped to speed up)
        net.eval(); s = 0.0; c = 0
        with torch.no_grad():
            for img, msk in val_ld:
                img, msk = img.to(device), msk.to(device)
                s += dice_score(net(img), msk); c += 1
                if c >= args.max_val_steps:
                    break
        vdice = s / max(1, c)
        print(f"[epoch {ep}] val dice={vdice:.4f}", flush=True)

        sched.step(vdice)
        print(f"lr now = {opt.param_groups[0]['lr']:.2e}", flush=True)

        if vdice > best:
            best = vdice
            torch.save({"state_dict": net.state_dict(), "dice": best}, os.path.join(args.out, "best.ckpt"))
            print(f"  saved best.ckpt (dice={best:.4f})", flush=True)

if __name__ == "__main__":
    main()
