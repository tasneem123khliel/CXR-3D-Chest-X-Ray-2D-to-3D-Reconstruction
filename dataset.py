"""
dataset.py — Chest X-Ray Dataset Loader with direct download URLs

Supported datasets (all free, no login required):
  1. Montgomery County CXR  — 138 PA X-rays + lung masks (NLM / NIH)
  2. Shenzhen Hospital CXR  — 662 PA X-rays (NLM / NIH)

Both datasets are from:
  U.S. National Library of Medicine (NLM)
  https://lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayImageDataSets.html
"""

from __future__ import annotations

import os
import zipfile
import urllib.request
import urllib.error
import shutil
import glob
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
#  DATASET REGISTRY  — add more here as needed
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "montgomery": {
        "name":        "Montgomery County CXR Set",
        "description": "138 frontal chest X-rays (80 normal, 58 TB). Includes left/right lung masks.",
        "url":         "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
                       "Montgomery-County-CXR-Set/MontgomerySet.zip",
        "zip_name":    "MontgomerySet.zip",
        "images_glob": "**/CXR_png/*.png",
        "masks_glob":  "**/ManualMask/**/*.png",
        "size_mb":     27,
        "n_images":    138,
        "license":     "Public domain (U.S. Gov / NLM)",
        "citation":    "Jaeger et al., IEEE TBME 2014",
    },
    "shenzhen": {
        "name":        "Shenzhen Hospital CXR Set",
        "description": "662 frontal chest X-rays (326 normal, 336 TB).",
        "url":         "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
                       "Shenzhen-Hospital-CXR-Set/ShenzhenSet.zip",
        "zip_name":    "ShenzhenSet.zip",
        "images_glob": "**/*.png",
        "masks_glob":  None,
        "size_mb":     86,
        "n_images":    662,
        "license":     "Public domain (U.S. Gov / NLM)",
        "citation":    "Jaeger et al., IEEE TBME 2014",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class DownloadProgress:
    """Simple callback for tracking download progress."""
    def __init__(self, total_mb: float, callback=None):
        self.total   = total_mb * 1024 * 1024
        self.downloaded = 0
        self.callback = callback  # fn(pct: float, mb_done: float)

    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            self.total = total_size
        self.downloaded += block_size
        pct = min(100.0, 100.0 * self.downloaded / max(self.total, 1))
        mb  = self.downloaded / 1024 / 1024
        if self.callback:
            self.callback(pct, mb)


def download_dataset(
    dataset_key: str = "montgomery",
    data_dir: str = "./data",
    progress_callback=None,
) -> Path:
    """
    Download and extract a dataset into data_dir.

    Args:
        dataset_key:       "montgomery" or "shenzhen"
        data_dir:          local directory to store data
        progress_callback: fn(pct: float, mb: float) for UI updates

    Returns:
        Path to the extracted dataset folder.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASETS)}")

    info    = DATASETS[dataset_key]
    out_dir = Path(data_dir) / dataset_key
    zip_path = Path(data_dir) / info["zip_name"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip if already extracted ─────────────────────────────────────────────
    existing = list(out_dir.rglob("*.png"))
    if len(existing) >= 10:
        print(f"[dataset] {dataset_key} already extracted ({len(existing)} images found).")
        return out_dir

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"[dataset] Downloading {info['name']} (~{info['size_mb']} MB) …")
    print(f"[dataset] URL: {info['url']}")
    try:
        tracker = DownloadProgress(info["size_mb"], progress_callback)
        urllib.request.urlretrieve(info["url"], zip_path, reporthook=tracker)
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Download failed: {e}\n"
            f"Manual download URL: {info['url']}\n"
            f"Save as: {zip_path}"
        )

    # ── Extract ───────────────────────────────────────────────────────────────
    print(f"[dataset] Extracting {zip_path} → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()  # remove zip to save space

    imgs = list(out_dir.rglob("*.png"))
    print(f"[dataset] Done. {len(imgs)} PNG files extracted.")
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
#  DATASET CLASS
# ──────────────────────────────────────────────────────────────────────────────

class ChestXRayDataset:
    """
    Loads images (and optional masks) from a downloaded dataset directory.

    Usage:
        ds = ChestXRayDataset.from_download("montgomery", data_dir="./data")
        img, mask = ds[0]   # numpy float32 arrays, normalised [0, 1]
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths:  Optional[List[Optional[Path]]] = None,
        image_size:  int = 512,
    ):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths or [None] * len(image_paths)
        self.image_size  = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = self._load_image(self.image_paths[idx])
        msk = None
        if self.mask_paths[idx] is not None:
            msk = self._load_mask(self.mask_paths[idx])
        return img, msk

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        # CLAHE
        img_u8 = (img * 255).astype(np.uint8)
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img    = clahe.apply(img_u8).astype(np.float32) / 255.0
        return img

    def _load_mask(self, path: Path) -> np.ndarray:
        msk = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
        msk = cv2.resize(msk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return (msk > 127).astype(np.float32)

    def get_sample_images(self, n: int = 6) -> List[np.ndarray]:
        """Return n evenly-spaced images for preview."""
        indices = np.linspace(0, len(self) - 1, min(n, len(self)), dtype=int)
        return [self[i][0] for i in indices]

    @classmethod
    def from_download(
        cls,
        dataset_key: str = "montgomery",
        data_dir: str = "./data",
        image_size: int = 512,
        progress_callback=None,
    ) -> "ChestXRayDataset":
        """Download (if needed) and return a ready-to-use dataset."""
        out_dir = download_dataset(dataset_key, data_dir, progress_callback)
        info    = DATASETS[dataset_key]

        # ── Find images ───────────────────────────────────────────────────────
        image_paths = sorted(out_dir.rglob("*.png"))
        # Filter to only real X-ray images (exclude mask subdirectory)
        if dataset_key == "montgomery":
            image_paths = [p for p in image_paths if "CXR_png" in str(p)]
        if not image_paths:
            # fallback: any PNG
            image_paths = sorted(out_dir.rglob("*.png"))

        # ── Find masks (Montgomery only) ──────────────────────────────────────
        mask_paths = [None] * len(image_paths)
        if dataset_key == "montgomery":
            mask_root = out_dir
            for i, img_p in enumerate(image_paths):
                stem = img_p.stem  # e.g. MCUCXR_0001_0
                # Try left + right mask
                left  = list(mask_root.rglob(f"*{stem}*left*"))
                right = list(mask_root.rglob(f"*{stem}*right*"))
                all_m = left + right
                if all_m:
                    mask_paths[i] = all_m[0]   # use first found

        print(f"[dataset] Loaded {len(image_paths)} images, {sum(m is not None for m in mask_paths)} with masks.")
        return cls(image_paths, mask_paths, image_size)

    @classmethod
    def from_folder(
        cls,
        folder: str,
        image_size: int = 512,
    ) -> "ChestXRayDataset":
        """Load from a local folder of PNG/JPEG X-ray images."""
        folder = Path(folder)
        exts   = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        paths  = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        return cls(paths, None, image_size)


# ──────────────────────────────────────────────────────────────────────────────
#  PYTORCH DATASET WRAPPER  (optional — needs torch)
# ──────────────────────────────────────────────────────────────────────────────

def make_torch_dataset(chest_ds: ChestXRayDataset):
    """
    Wrap ChestXRayDataset in a torch Dataset for DataLoader use.

    Requires: torch, torchvision, albumentations

    Training example:
        ds  = ChestXRayDataset.from_download("montgomery")
        tds = make_torch_dataset(ds)
        loader = DataLoader(tds, batch_size=4, shuffle=True, num_workers=2)
        for imgs, masks in loader:
            seg_pred, depth_pred = model(imgs)
            loss = criterion(seg_pred, masks)
            ...
    """
    try:
        import torch
        from torch.utils.data import Dataset as TorchDataset
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install torch torchvision")

    class TorchWrapper(TorchDataset):
        def __init__(self, base_ds):
            self.ds = base_ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, mask = self.ds[idx]
            img_t = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
            if mask is not None:
                msk_t = torch.from_numpy(mask).unsqueeze(0)
            else:
                msk_t = torch.zeros(1, self.ds.image_size, self.ds.image_size)
            return img_t, msk_t

    return TorchWrapper(chest_ds)


# ──────────────────────────────────────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Dataset URLs:")
    for key, info in DATASETS.items():
        print(f"  [{key}]  {info['name']}")
        print(f"    URL  : {info['url']}")
        print(f"    Size : ~{info['size_mb']} MB  |  {info['n_images']} images")
        print(f"    Cite : {info['citation']}")
        print()
