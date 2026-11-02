"""Turn a FastSurfer-processed T1 MRI into a compact, normalized brain volume.

FastSurfer writes a conformed 256^3, 1 mm image (mri/orig_nu.mgz or mri/orig.mgz)
and a brain mask (mri/mask.mgz). We skull-strip, crop to the brain, pad to a cube,
resize to `size`^3 and z-score intensities inside the brain. Saved as float16 .npy.
"""

from pathlib import Path

import numpy as np

IMAGE_CANDIDATES = ["mri/orig_nu.mgz", "mri/orig.mgz"]
MASK_CANDIDATES = ["mri/mask.mgz", "mri/aparc.DKTatlas+aseg.deep.mgz"]


def _first(subject_dir: Path, names: list[str]) -> Path | None:
    return next((subject_dir / n for n in names if (subject_dir / n).exists()), None)


def load_subject(subject_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (image, brain mask) from a FastSurfer subject folder."""
    import nibabel as nib

    subject_dir = Path(subject_dir)
    img_p, mask_p = _first(subject_dir, IMAGE_CANDIDATES), _first(subject_dir, MASK_CANDIDATES)
    if img_p is None or mask_p is None:
        raise FileNotFoundError(f"FastSurfer image or mask missing in {subject_dir}")
    img = np.asarray(nib.load(str(img_p)).dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(mask_p)).dataobj) > 0
    return img, mask


def resize_cube(vol: np.ndarray, size: int) -> np.ndarray:
    """Trilinear resize of a cubic volume to size^3 (uses torch when available)."""
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(np.ascontiguousarray(vol))[None, None]
        return F.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)[0, 0].numpy()
    except ImportError:
        from scipy.ndimage import zoom

        return zoom(vol, size / vol.shape[0], order=1)


def preprocess(img: np.ndarray, mask: np.ndarray, size: int = 128, margin: int = 4) -> np.ndarray:
    """Skull-strip, crop to brain, pad to cube, resize, z-score inside the brain."""
    if not mask.any():
        raise ValueError("empty brain mask")
    img = np.where(mask, img, 0).astype(np.float32)
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, img.shape)
    crop, cmask = img[tuple(slice(a, b) for a, b in zip(lo, hi))], mask[
        tuple(slice(a, b) for a, b in zip(lo, hi))]
    side = max(crop.shape)
    pad = [((side - s) // 2, side - s - (side - s) // 2) for s in crop.shape]
    crop, cmask = np.pad(crop, pad), np.pad(cmask, pad)
    vals = crop[cmask]
    crop = np.where(cmask, (crop - vals.mean()) / (vals.std() + 1e-6), 0).astype(np.float32)
    return resize_cube(crop, size).astype(np.float16)
