import argparse, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from recognition.unet_hipmri2d_48798112.dataset import HipMRI2DSlices
from recognition.unet_hipmri2d_48798112.modules import UNet

def dice_score(logits, target, eps=1e-6):
    p = (logits.sigmoid() > 0.5).float()
    t = target.float()
    inter = (p*t).sum(dim=(1,2,3))
    denom = p.sum(dim=(1,2,3)) + t.sum(dim=(1,2,3)) + eps
    return (2*inter/denom).mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--prostate-id", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = HipMRI2DSlices(args.data, "train",
                              target_size=(args.size, args.size),
                              prostate_id=args.prostate_id)
    val_ds   = HipMRI2DSlices(args.data, "val",
                              target_size=(args.size, args.size),
                              prostate_id=args.prostate_id)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    net = UNet(1, 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best = -1.0
    for ep in range(1, args.epochs+1):
        net.train(); steps = 0
        for img, msk in train_ld:
            img, msk = img.to(device), msk.to(device)
            logits = net(img)
            loss = F.binary_cross_entropy_with_logits(logits, msk)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1
            if steps >= args.max_train_steps:
                break

        net.eval(); s=0.0; c=0
        with torch.no_grad():
            for img, msk in val_ld:
                img, msk = img.to(device), msk.to(device)
                s += dice_score(net(img), msk); c += 1
        vdice = s / max(c, 1)
        print(f"[epoch {ep}] val dice = {vdice:.4f}")
        if vdice > best:
            best = vdice
            torch.save({"epoch": ep, "state_dict": net.state_dict()},
                       os.path.join(args.out, "best.ckpt"))
            print(f"  saved best.ckpt (dice={best:.4f})")

if __name__ == "__main__":
    main()

