"""Crop raw FLAIR volumes and restore aligned predictions.

Usage:
    from neuroseg_2d.processing import postprocess, preprocess
"""

__author__ = "Vu Hoi HUYNH"

import numpy as np
import torch
from torch import Tensor

from neuroseg_2d.constants import Z_END, Z_START

# ──────────────────────────────────────────────────────────────────────────────────────────────── #
# Preprocessing pipeline. #


def normalize_z_score(image: Tensor) -> Tensor:
    """Normalize nonzero intensities while keeping zero background unchanged."""
    # Select foreground before normalization can turn its mean intensity into zero.
    fg = image != 0
    vals = image[fg]

    # Apply volume-level population z-score normalization on foreground only.
    mean = vals.mean()
    std = vals.std(correction=0)
    norm = torch.zeros_like(image, dtype=torch.float32)
    norm[fg] = (vals - mean) / std.clamp_min(1e-8)

    return norm


def preprocess(
    img: Tensor,
    mask: Tensor | None,
    src_aff: np.ndarray,
) -> tuple[Tensor, Tensor | None, np.ndarray]:
    """Crop and normalize one aligned patient volume pair."""
    # Keep one fixed 144-slice axial range for every patient.
    dst_img = img[:, :, Z_START:Z_END]
    dst_mask = None if mask is None else mask[:, :, Z_START:Z_END]

    # Shift cropped voxel zero to source voxel `(0, 0, Z_START)` in world space.
    dst_aff = src_aff.copy()
    dst_aff[:3, 3] += src_aff[:3, 2] * Z_START

    # Normalize cropped FLAIR while keeping zero background unchanged.
    dst_img = normalize_z_score(dst_img)

    return dst_img, dst_mask, dst_aff


# ──────────────────────────────────────────────────────────────────────────────────────────────── #
# Postprocessing pipeline. #


def postprocess(
    probs: Tensor,
    src_shape: tuple[int, ...],
    threshold: float,
) -> np.ndarray:
    """Insert one thresholded crop into its original source shape."""
    # Convert cropped probabilities into one binary prediction.
    pred = (probs >= threshold).to(torch.uint8).numpy()

    # Restore exact source indices and leave cropped-out slices as background.
    restored = np.zeros(src_shape, dtype=np.uint8)
    restored[:, :, Z_START:Z_END] = pred

    return restored
