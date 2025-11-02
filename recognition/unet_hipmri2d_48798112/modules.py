# modules.py  ——  minimal, clean UNet (2D)
import torch
import torch.nn as nn

def _block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, widths=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = widths
        self.enc1 = _block(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _block(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _block(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _block(c3, c4)

        self.up3  = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _block(c3 + c3, c3)
        self.up2  = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _block(c2 + c2, c2)
        self.up1  = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _block(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b  = self.bottleneck(p3)
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)
