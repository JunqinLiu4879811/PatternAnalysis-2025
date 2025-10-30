# -*- coding: utf-8 -*-
"""
HipMRI 2D slice dataset for prostate segmentation.

- Expects NIfTI volumes under:
    ROOT/semantic_MRs/            (images, *.nii or *.nii.gz)
    ROOT/semantic_labels_only/    (labels, same filenames as images)

- Converts 3D volumes to 2D slices on the fly.
- Returns (img, mask) as torch.float32 tensors with shape (1, H, W).
- The mask is binarized for the prostate label.

Notes:
- Do NOT commit data/weights/logs to the repository.
- Use `--scan-labels` once to inspect label IDs if unsure about the prostate ID.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib


class HipMRI2DSlices(Dataset):
    """
    HipMRI 2D slice dataset.

    Args:
        root: Path to HipMRI root OR directly to `semantic_MRs`/`semantic_labels_only`.
        split: One of {"train", "val", "test"}. Volumes are split 70/15/15, then expanded to slices.
        axis: Slice axis {0, 1, 2}. Default 2 (axial).
        prostate_id: Integer label ID for the prostate. If None, the dataset infers it
                     per-slice as the largest non-zero value found (fallback heuristic).
        target_size: Optional (H, W) to resize both image and mask.
        seed: RNG seed for reproducible train/val/test volume split.
        transform: Optional callable taking (img, mask) -> (img, mask) after tensor creation.
        return_path: If True, also return the (image_path, slice_index) for debugging.

    Behavior:
        - Each dataset item is one 2D slice taken from a paired (image, label) volume.
        - Image is z-score normalized per slice.
        - Mask is binarized: (label == prostate_id) -> 1.0, else 0.0.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        axis: int = 2,
        prostate_id: int | None = None,
        target_size: Tuple[int, int] | None = None,
        seed: int = 48798112,
        transform=None,
        return_path: bool = False,
    ):
        self.root = Path(root)
        # Allow passing either ROOT or the direct subdirs
        self.img_dir = self.root if self.root.name == "semantic_MRs" else self.root / "semantic_MRs"
        self.msk_dir = (
            self.root if self.root.name == "semantic_labels_only" else self.root / "semantic_labels_only"
        )

        if not self.img_dir.exists() or not self.msk_dir.exists():
            raise FileNotFoundError(
                f"Expected directories:\n  {self.img_dir}\n  {self.msk_dir}\nCheck your DATA_ROOT."
            )

        self.axis = int(axis)
        assert self.axis in (0, 1, 2), "axis must be 0, 1, or 2"
        self.prostate_id = prostate_id
        self.target_size = target_size
        self.transform = transform
        self.return_path = return_path

        # Pair volumes by identical filenames
        img_vols = sorted(self.img_dir.glob("*.nii*"))
        pairs: List[tuple[Path, Path]] = []
        for ip in img_vols:
            mp = self.msk_dir / ip.name
            if mp.exists():
                pairs.append((ip, mp))

        if not pairs:
            raise RuntimeError("No paired NIfTI files found.")

        # Volume-level split, then expand to slices
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n = len(pairs)
        n_train, n_val = int(0.70 * n), int(0.15 * n)

        if split == "train":
            pairs = pairs[:n_train]
        elif split == "val":
            pairs = pairs[n_train : n_train + n_val]
        elif split == "test":
            pairs = pairs[n_train + n_val :]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'.")

        # Build slice index without loading full volumes into memory
        self.samples: List[tuple[Path, Path, int]] = []
        for ip, mp in pairs:
            depth = nib.load(str(ip)).shape[self.axis]  # fast: header info only
            for k in range(depth):
                self.samples.append((ip, mp, k))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        m, s = float(x.mean()), float(x.std())
        return (x - m) / (s + eps)

    @staticmethod
    def _slice2d(x3d: np.ndarray, k: int, axis: int) -> np.ndarray:
        if axis == 0:
            return x3d[k, :, :]
        elif axis == 1:
            return x3d[:, k, :]
        else:
            return x3d[:, :, k]

    def _binarize_prostate(self, lab2d: np.ndarray) -> np.ndarray:
        if self.prostate_id is not None:
            pid = int(self.prostate_id)
        else:
            uniq = np.unique(lab2d)
            nonzero = [u for u in uniq if u != 0]
            if not nonzero:
                # If the slice is empty, keep it empty
                return (lab2d * 0).astype(np.float32)
            # Heuristic: the largest non-zero label is often the organ of interest
            pid = int(max(nonzero))
        return (lab2d == pid).astype(np.float32)

    def __getitem__(self, idx: int):
        ip, mp, k = self.samples[idx]

        # Load volumes and extract the k-th slice
        img3d = nib.load(str(ip)).get_fdata().astype(np.float32)
        lab3d = nib.load(str(mp)).get_fdata().astype(np.float32)

        img2d = self._slice2d(img3d, k, self.axis)
        lab2d = self._slice2d(lab3d, k, self.axis)

        # Per-slice normalization and binarization
        img2d = self._zscore(img2d)
        msk2d = self._binarize_prostate(lab2d)

        # To tensors: [1, H, W]
        img = torch.from_numpy(img2d).unsqueeze(0)  # float32
        msk = torch.from_numpy(msk2d).unsqueeze(0)  # float32 in {0., 1.}

        # Optional resize
        if self.target_size is not None:
            H, W = self.target_size
            img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            msk = F.interpolate(msk.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)

        if self.transform is not None:
            img, msk = self.transform(img, msk)

        if self.return_path:
            return img, msk, (str(ip), k)
        return img, msk


# ---------- tiny utilities for quick inspection ----------
def scan_label_uniques(root: str | Path, max_vols: int = 3):
    """Print unique label values from a few volumes (helps decide prostate_id)."""
    root = Path(root)
    lab_dir = root if root.name == "semantic_labels_only" else root / "semantic_labels_only"
    vols = sorted(lab_dir.glob("*.nii*"))[:max_vols]
    print(f"[scan] reading {len(vols)} label volumes from: {lab_dir}")
    all_u = set()
    for p in vols:
        u = np.unique(nib.load(str(p)).get_fdata())
        print(f"  {p.name}: uniques={u}")
        all_u.update(u.tolist())
    print(f"[scan] union uniques: {sorted(all_u)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke tests for HipMRI2DSlices.")
    parser.add_argument("--data", required=True, help="HipMRI root, e.g., /home/groups/comp3710/HipMRI_Study_open")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--prostate-id", type=int, default=None)
    parser.add_argument("--scan-labels", action="store_true", help="List unique label values and exit")
    args = parser.parse_args()

    if args.scan_labels:
        scan_label_uniques(args.data)
        raise SystemExit(0)

    ds = HipMRI2DSlices(
        root=args.data,
        split=args.split,
        axis=args.axis,
        prostate_id=args.prostate_id,
    )
    print("length:", len(ds))
    x, y = ds[0]
    print("shapes:", tuple(x.shape), tuple(y.shape), "| dtype:", x.dtype, y.dtype)
