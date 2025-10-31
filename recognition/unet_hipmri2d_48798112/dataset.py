# -*- coding: utf-8 -*-
"""
HipMRI 2D slice dataset for prostate segmentation (robust pairing).

This version pairs image/label volumes using:
1) Exact same filename (preferred), else
2) A canonical "subject_week" prefix, e.g. "B006_Week0", derived from filenames
   like "B006_Week0_SEMANTIC.nii.gz" vs "B006_Week0_MR.nii.gz".

Directory layout:
    ROOT/
      semantic_MRs/            (*.nii or *.nii.gz)
      semantic_labels_only/    (*.nii or *.nii.gz)

Returns (img, mask) as float32 tensors with shape (1, H, W).
"""

from __future__ import annotations
from pathlib import Path
import argparse
import random
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib


def _stem_noext(p: Path) -> str:
    """Return filename without .nii/.nii.gz."""
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def _canonical_key(p: Path) -> str:
    """
    Build a robust pairing key from filename.
    Strategy:
      - If pattern like "<SUBJ>_Week<k>_..." exists, use "SUBJ_Week<k>".
      - Otherwise, drop common suffix tokens and keep the first 2 tokens.
    """
    s = _stem_noext(p)
    toks = s.split("_")
    # case 1: subject_week at the beginning
    if len(toks) >= 2 and toks[1].lower().startswith("week"):
        return f"{toks[0]}_{toks[1]}"
    # fallback: drop common suffix tokens and keep first two
    drop = {"semantic", "sem", "label", "labels", "mr", "mri", "img", "image",
            "t1", "t2", "flair", "hr", "lr", "seg", "segmentation"}
    core = [t for t in toks if t.lower() not in drop]
    if len(core) >= 2:
        return f"{core[0]}_{core[1]}"
    return core[0] if core else s


class HipMRI2DSlices(Dataset):
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
        debug_pairs: bool = False,
    ):
        self.root = Path(root)
        self.img_dir = self.root if self.root.name == "semantic_MRs" else self.root / "semantic_MRs"
        self.msk_dir = self.root if self.root.name == "semantic_labels_only" else self.root / "semantic_labels_only"
        if not self.img_dir.exists() or not self.msk_dir.exists():
            raise FileNotFoundError(f"Expected:\n  {self.img_dir}\n  {self.msk_dir}")

        self.axis = int(axis)
        assert self.axis in (0, 1, 2)
        self.prostate_id = prostate_id
        self.target_size = target_size
        self.transform = transform
        self.return_path = return_path

        # ---- build pairs (exact-match first, then prefix-match) ----
        imgs = sorted(self.img_dir.glob("*.nii*"))
        labs = sorted(self.msk_dir.glob("*.nii*"))
        if not imgs or not labs:
            raise RuntimeError("No NIfTI files found in images or labels directory.")

        # Map for exact names
        img_by_name: Dict[str, Path] = {_stem_noext(p): p for p in imgs}
        lab_by_name: Dict[str, Path] = {_stem_noext(p): p for p in labs}
        name_inter = sorted(set(img_by_name) & set(lab_by_name))
        pairs: List[tuple[Path, Path]] = [(img_by_name[k], lab_by_name[k]) for k in name_inter]

        if not pairs:
            # Fallback to canonical prefix key (e.g., "B006_Week0")
            img_by_key: Dict[str, Path] = {_canonical_key(p): p for p in imgs}
            lab_by_key: Dict[str, Path] = {_canonical_key(p): p for p in labs}
            key_inter = sorted(set(img_by_key) & set(lab_by_key))
            pairs = [(img_by_key[k], lab_by_key[k]) for k in key_inter]

        if debug_pairs:
            print(f"[debug] images: {len(imgs)}, labels: {len(labs)}")
            print(f"[debug] exact-name pairs: {len(name_inter)}")
            if not name_inter:
                print(f"[debug] prefix-key pairs: {len(pairs)} (using canonical subject_week keys)")
                for i, (ip, mp) in enumerate(pairs[:5]):
                    print(f"  pair[{i}]: {ip.name}  <->  {mp.name}")

        if not pairs:
            # Show a few filenames to help debugging
            some_imgs = ", ".join(p.name for p in imgs[:5])
            some_labs = ", ".join(p.name for p in labs[:5])
            raise RuntimeError(
                "No paired NIfTI files found.\n"
                f"Example image names: {some_imgs}\n"
                f"Example label names: {some_labs}\n"
                "If naming differs, try the prefix-based pairing logic."
            )

        # ---- split by volume then expand to slices ----
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n = len(pairs)
        n_train, n_val = int(0.70 * n), int(0.15 * n)
        if split == "train":
            pairs = pairs[:n_train]
        elif split == "val":
            pairs = pairs[n_train:n_train + n_val]
        elif split == "test":
            pairs = pairs[n_train + n_val:]
        else:
            raise ValueError("split must be 'train'|'val'|'test'.")

        self.samples: List[tuple[Path, Path, int]] = []
        for ip, mp in pairs:
            depth = nib.load(str(ip)).shape[self.axis]  # header read
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
        if axis == 0:   return x3d[k, :, :]
        if axis == 1:   return x3d[:, k, :]
        return x3d[:, :, k]

    def _binarize_prostate(self, lab2d: np.ndarray) -> np.ndarray:
        if self.prostate_id is not None:
            pid = int(self.prostate_id)
        else:
            uniq = np.unique(lab2d)
            nz = [u for u in uniq if u != 0]
            if not nz:
                return (lab2d * 0).astype(np.float32)
            pid = int(max(nz))  # heuristic
        return (lab2d == pid).astype(np.float32)

    def __getitem__(self, idx: int):
        ip, mp, k = self.samples[idx]
        img3d = nib.load(str(ip)).get_fdata().astype(np.float32)
        lab3d = nib.load(str(mp)).get_fdata().astype(np.float32)

        img2d = self._slice2d(img3d, k, axis=2) if self.axis == 2 else self._slice2d(img3d, k, self.axis)
        lab2d = self._slice2d(lab3d, k, axis=2) if self.axis == 2 else self._slice2d(lab3d, k, self.axis)

        img2d = self._zscore(img2d)
        msk2d = self._binarize_prostate(lab2d)

        img = torch.from_numpy(img2d).unsqueeze(0)
        msk = torch.from_numpy(msk2d).unsqueeze(0)

        if self.target_size is not None:
            H, W = self.target_size
            img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            msk = F.interpolate(msk.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)

        return img, msk


def scan_label_uniques(root: str | Path, max_vols: int = 3):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="HipMRI root, e.g., /home/groups/comp3710/HipMRI_Study_open")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--axis", type=int, default=2)
    ap.add_argument("--prostate-id", type=int, default=None)
    ap.add_argument("--scan-labels", action="store_true")
    ap.add_argument("--debug-pairs", action="store_true", help="Print first few paired filenames")
    args = ap.parse_args()

    if args.scan_labels:
        scan_label_uniques(args.data)
        raise SystemExit(0)

    ds = HipMRI2DSlices(
        root=args.data, split=args.split, axis=args.axis,
        prostate_id=args.prostate_id, debug_pairs=args.debug_pairs
    )
    print("length:", len(ds))
    x, y = ds[0]
    print("shapes:", tuple(x.shape), tuple(y.shape), "| dtype:", x.dtype, y.dtype)
