"""Infer one binary tumor mask from one raw FLAIR NIfTI volume.

Usage:
    python -m neuroseg_2d.infer [options]
"""

__author__ = "Vu Hoi HUYNH"

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from torch import Tensor, nn

from neuroseg_2d.constants import BATCH_SIZE, THRESHOLD
from neuroseg_2d.model import UNet
from neuroseg_2d.processing import postprocess, preprocess
from neuroseg_2d.visualize import load_ground_truth, show_gui


def parse_args() -> argparse.Namespace:
    """Parse and validate one inference request."""
    parser = argparse.ArgumentParser(description="Infer one tumor mask from one raw FLAIR volume.")
    parser.add_argument("--image", type=Path, required=True, help="Raw FLAIR NIfTI path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path.")
    parser.add_argument("--output", type=Path, required=True, help="Output binary NIfTI path.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--gui", action="store_true", help="Inspect source-grid axial predictions.")
    args = parser.parse_args()

    # Normalize and validate user input before loading model or image data.
    args.image = args.image.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    return args


def predict(model: nn.Module, image: Tensor, device: torch.device, batch_size: int) -> Tensor:
    """Predict all axial slices into one preallocated CPU probability volume."""
    slices = image.movedim(2, 0).unsqueeze(1)
    probabilities = torch.empty(tuple(image.shape), dtype=torch.float32)
    model.eval()

    # Predict ordered batches and release each GPU batch after its probabilities reach CPU.
    with torch.inference_mode():
        for start in range(0, len(slices), batch_size):
            end = min(start + batch_size, len(slices))
            batch = slices[start:end].to(device)
            batch_probabilities = torch.sigmoid(model(batch)).squeeze(1).cpu()
            probabilities[:, :, start:end] = batch_probabilities.movedim(0, 2)

    return probabilities


def main() -> None:
    """Load one raw FLAIR, infer its mask, and save it on the source grid."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preserve source geometry, then canonicalize the working image for training preprocessing.
    src_nii = nib.Nifti1Image.from_filename(args.image)
    src_aff = np.asarray(src_nii.affine)
    src_header = src_nii.header.copy()
    work_nii = nib.as_closest_canonical(src_nii)
    img_arr = work_nii.get_fdata(dtype=np.float32, caching="unchanged")
    if img_arr.ndim != 3 or not np.isfinite(img_arr).all() or not np.any(img_arr):
        raise ValueError("Input must be one finite, non-empty 3D FLAIR volume")
    src_img = torch.from_numpy(np.ascontiguousarray(img_arr))

    # Reuse training preprocessing and load the explicit checkpoint into the fixed U-Net.
    dst_img, _, _ = preprocess(
        src_img,
        None,
        np.asarray(work_nii.affine),
    )
    model = UNet(in_channels=1, out_channels=1).to(device)
    weights: dict[str, Tensor] = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights)

    # Restore canonical crop, then return its axes to source orientation before saving.
    probs = predict(model, dst_img, device, args.batch_size)
    mask = postprocess(probs, tuple(work_nii.shape), args.threshold)
    orientation = ornt_transform(io_orientation(work_nii.affine), io_orientation(src_aff))
    mask = apply_orientation(mask, orientation)
    src_header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask, src_aff, src_header), args.output)

    print(f"Input: {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}")
    print(f"Predicted tumor voxels: {int(mask.sum())}")

    # Open optional viewer only after durable output and normal summary are complete.
    if args.gui:
        src_img = src_nii.get_fdata(dtype=np.float32, caching="unchanged")
        target, target_message = load_ground_truth(args.image, tuple(src_nii.shape), src_aff)
        if target_message is not None:
            print(target_message)
        patient_name = (
            args.image.name.removesuffix(".nii.gz").removesuffix(".nii").removesuffix("_flair")
        )
        show_gui(
            src_img,
            mask.astype(bool),
            target,
            target_message,
            patient_name,
            src_aff,
            args.threshold,
        )


if __name__ == "__main__":
    main()
