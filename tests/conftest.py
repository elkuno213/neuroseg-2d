"""Build synthetic NIfTI fixtures for NeuroSeg 2D data tests.

Usage:
    uv run pytest tests/test_data.py
"""

__author__ = "Vu Hoi HUYNH"

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from neuroseg_2d.constants import SHAPE
from neuroseg_2d.data import Patient, discover


@pytest.fixture(scope="module")
def synthetic_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create deterministic full-size HGG and LGG NIfTI pairs."""
    data_dir = tmp_path_factory.mktemp("synthetic_brats")
    affine = np.eye(4, dtype=np.float64)

    # Build one bounded foreground with two balanced intensities around mean 2.
    raw_shape = (SHAPE[0], SHAPE[1], 155)
    image = np.zeros(raw_shape, dtype=np.float32)
    foreground = image[32:208, 32:208, 16:139]
    foreground[:] = 1.0
    foreground[::2] = 3.0

    patient_specs = [
        ("HGG", "Synthetic_HGG_001", 4, (slice(80, 120), slice(80, 120), slice(48, 80))),
        ("LGG", "Synthetic_LGG_001", 2, (slice(120, 152), slice(96, 128), slice(64, 88))),
    ]

    # Write canonical package filenames under grade and patient directories.
    for grade, patient_id, label, tumor in patient_specs:
        patient_dir = data_dir / grade / patient_id
        patient_dir.mkdir(parents=True)
        mask = np.zeros(raw_shape, dtype=np.uint8)
        mask[tumor] = label
        nib.save(nib.Nifti1Image(image, affine), patient_dir / f"{patient_id}_flair.nii")
        nib.save(nib.Nifti1Image(mask, affine), patient_dir / f"{patient_id}_seg.nii")

    return data_dir


@pytest.fixture(scope="module")
def raw_patients(synthetic_data_dir: Path) -> list[Patient]:
    """Discover synthetic raw patients through package API."""
    return discover(synthetic_data_dir)
