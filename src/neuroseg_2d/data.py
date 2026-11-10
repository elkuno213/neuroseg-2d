"""Extract BraTS data and load online-preprocessed axial samples.

Usage:
    python -m neuroseg_2d.data [options]
"""

__author__ = "Vu Hoi HUYNH"

import argparse
import math
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from neuroseg_2d.constants import (
    BATCH_SIZE,
    SEED,
    SHAPE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from neuroseg_2d.processing import preprocess


@dataclass(frozen=True, slots=True)
class Patient:
    """Store one patient's grade and required file paths."""

    grade: str  # HGG or LGG.
    id: str  # Directory name under HGG or LGG.
    flair_file: Path  # Absolute path to the FLAIR NIfTI file.
    seg_file: Path  # Absolute path to the segmentation NIfTI file.


def extract_dataset(archive_file: Path, data_dir: Path) -> Path:
    """Extract required BraTS files into grade and patient directories."""

    # Validate archive and prepare raw output directory.
    if not archive_file.is_file():
        raise FileNotFoundError(f"Archive file must exist at: {archive_file}")
    if data_dir.exists() and not data_dir.is_dir():
        raise NotADirectoryError(f"Data dir is currently not a directory: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Scan flat ZIP member list once and collect required files.
    with ZipFile(archive_file, mode="r") as archive:
        # Keep only FLAIR and segmentation files under HGG or LGG grade directories.
        members = archive.infolist()
        if not members:
            raise ValueError(f"Archive is empty: {archive_file}")
        required_members: list[tuple[ZipInfo, str, str, str]] = []
        for member in members:
            if member.is_dir():
                continue
            member_file = PurePosixPath(member.filename)
            member_filename = member_file.name
            if not member_filename.lower().endswith(("_flair.nii", "_seg.nii")):
                continue
            grade = member_file.parent.parent.name
            if grade not in {"HGG", "LGG"}:
                continue
            required_members.append((member, grade, member_file.parent.name, member_filename))

        # Fail fast if no required files are found in the archive.
        if not required_members:
            raise ValueError("No required file found in this archive")

        # Extract collected files under preserved grade/patient directories.
        for member, grade, patient_id, member_filename in tqdm(
            required_members,
            desc="Extracting",
            unit="file",
        ):
            # Skip complete files; replace missing or incomplete files.
            final_file = data_dir / grade / patient_id / member_filename
            final_file.parent.mkdir(parents=True, exist_ok=True)
            if not final_file.exists() or final_file.stat().st_size != member.file_size:
                with (
                    archive.open(member) as src,
                    final_file.open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)

    return data_dir


def discover(data_dir: Path) -> list[Patient]:
    """Build validated Patient objects from extracted files."""

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data dir is not found: {data_dir}")

    # Loop over both grades and all patient directories to find canonical FLAIR/segmentation pairs.
    patients: list[Patient] = []
    for grade in ("HGG", "LGG"):
        grade_dir = data_dir / grade
        if not grade_dir.is_dir():
            raise NotADirectoryError(f"Missing grade directory: {grade_dir}")
        for patient_dir in sorted(grade_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            # Require one canonical FLAIR/segmentation pair per patient.
            patient_id = patient_dir.name
            flair_file = patient_dir / f"{patient_id}_flair.nii"
            seg_file = patient_dir / f"{patient_id}_seg.nii"
            if not (flair_file.is_file() and seg_file.is_file()):
                raise ValueError(
                    f"Patient {patient_id} must have exactly one FLAIR file and one segmentation file"
                )

            patients.append(
                Patient(
                    grade=grade,
                    id=patient_id,
                    flair_file=flair_file,
                    seg_file=seg_file,
                )
            )

    if not patients:
        raise ValueError(f"No valid patients found in {data_dir}")
    return patients


def split(
    patients: list[Patient], seed: int = SEED
) -> tuple[list[Patient], list[Patient], list[Patient]]:
    """Split patients into train, validation, and test groups."""

    # Validate configuration and input before random splitting.
    if not math.isclose(sum([TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO]), 1.0):
        raise ValueError("Sum of all ratio must be 1")
    if not patients:
        raise ValueError("Patients must be not empty")

    grades = [patient.grade for patient in patients]
    train_patients, other_patients = train_test_split(
        patients,
        test_size=TEST_RATIO + VALIDATION_RATIO,
        random_state=seed,
        stratify=grades,
    )
    val_patients, test_patients = train_test_split(
        other_patients,
        test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO),
        random_state=seed,
        stratify=[patient.grade for patient in other_patients],
    )

    # Prove split is complete and leak-free before returning it.
    train_ids = {patient.id for patient in train_patients}
    val_ids = {patient.id for patient in val_patients}
    test_ids = {patient.id for patient in test_patients}
    if not (
        train_ids.isdisjoint(val_ids)
        and val_ids.isdisjoint(test_ids)
        and test_ids.isdisjoint(train_ids)
    ):
        raise ValueError("Train/Validation/Test datasets are not different to each other")

    ids = [patient.id for patient in patients]
    split_ids = [
        patient.id
        for patients in (train_patients, val_patients, test_patients)
        for patient in patients
    ]
    if len(ids) != len(split_ids) or set(ids) != set(split_ids):
        raise ValueError("Splits must contain every input patient exactly once")

    return train_patients, val_patients, test_patients


def read_patient(patient: Patient) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Read and validate one canonical pair, then convert both volumes to CPU tensors."""

    # Load metadata first, then make both voxel grids follow canonical RAS+ axes.
    flair_nifti = nib.as_closest_canonical(nib.Nifti1Image.from_filename(patient.flair_file))
    seg_nifti = nib.as_closest_canonical(nib.Nifti1Image.from_filename(patient.seg_file))

    # Check paired geometry after canonicalization.
    flair_shape = flair_nifti.shape
    seg_shape = seg_nifti.shape
    if len(flair_shape) != 3 or flair_shape != seg_shape:
        raise ValueError(
            f"Expected matching 3D images for {patient.id}, "
            f"got FLAIR {flair_shape} and segmentation {seg_shape}"
        )
    if not np.allclose(flair_nifti.affine, seg_nifti.affine, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Canonical affine mismatch for {patient.id}: "
            "FLAIR and segmentation volumes are not spatially aligned"
        )

    # Check voxel values after geometry validation
    flair_array = flair_nifti.get_fdata(dtype=np.float32, caching="unchanged")
    seg_array = np.asarray(seg_nifti.dataobj)
    if not np.isfinite(flair_array).all() or not np.isfinite(seg_array).all():
        raise ValueError(f"Non-finite voxel values for {patient.id}")
    if not np.any(flair_array):
        raise ValueError(f"Empty FLAIR foreground for {patient.id}")

    # Check segmentation labels after voxel value validation
    valid_labels = {0, 1, 2, 4}
    labels = set(np.unique(seg_array).tolist())
    if labels - valid_labels:
        raise ValueError(
            f"Unexpected segmentation labels for {patient.id}: "
            f"{sorted(labels)}; expected only {sorted(valid_labels)}"
        )

    # Convert both volumes to contiguous CPU tensors after all validations.
    image_3d = torch.from_numpy(np.ascontiguousarray(flair_array))
    mask_3d = torch.from_numpy(np.ascontiguousarray(np.asarray(seg_array > 0, dtype=np.uint8)))

    return image_3d, mask_3d, np.asarray(flair_nifti.affine)


def show_stats(name: str, patients: list[Patient], total_patients: int) -> None:
    """Print patient count, percentage, and grade balance."""

    num_hgg = sum(patient.grade == "HGG" for patient in patients)
    num_lgg = sum(patient.grade == "LGG" for patient in patients)
    percentage = len(patients) / total_patients * 100
    print(
        f"{name:<12} {len(patients):>3} patients "
        f"({percentage:>5.1f}%) — HGG {num_hgg:>3}, LGG {num_lgg:>3}"
    )


class BraTSSliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    def __init__(
        self,
        patients: list[Patient],
    ) -> None:
        # Cache indices of patient and axial slice without loading any volumes into memory yet.
        if not patients:
            raise ValueError("Patient list must not be empty")
        self.patients = patients
        self.samples: list[tuple[int, int]] = []
        for patient_idx, _ in enumerate(patients):
            for axial_idx in range(SHAPE[2]):
                self.samples.append((patient_idx, axial_idx))

        # Cache (patient index, image 3D, mask 3D) for consecutive slices.
        self.curr: tuple[int, torch.Tensor, torch.Tensor] | None = None

    def __len__(self) -> int:
        """Return number of indexed axial slices."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """Return one FLAIR slice, binary mask, grade, and patient ID."""

        patient_idx, axial_idx = self.samples[index]
        patient = self.patients[patient_idx]

        # Read and preprocess 3D images only when requested patient changes or cache is empty.
        if self.curr is None or self.curr[0] != patient_idx:
            src_image, src_mask, src_aff = read_patient(patient)
            image_3d, mask_3d, _ = preprocess(src_image, src_mask, src_aff)
            if mask_3d is None:
                raise ValueError(f"Missing preprocessed mask for {patient.id}")
            self.curr = (patient_idx, image_3d, mask_3d)

        # Select one axial slice from (H, W, D), then add channel dimension.
        _, image_3d, mask_3d = self.curr
        image = image_3d[:, :, axial_idx].unsqueeze(0).contiguous()
        mask = mask_3d[:, :, axial_idx].unsqueeze(0).contiguous()

        return image, mask, patient.grade, patient.id


class PatientSliceSampler(Sampler[int]):
    """Shuffle patients and their slices while keeping each patient's indices together."""

    def __init__(self, num_patients: int, seed: int = SEED) -> None:
        self.num_patients = num_patients
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        """Return the number of slices visited per epoch."""
        return self.num_patients * SHAPE[2]

    def set_epoch(self, epoch: int) -> None:
        """Select the absolute epoch, including after resuming training."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield shuffled dataset indices without changing the Dataset mapping."""
        # Reproduce this epoch's order without consuming global randomness.
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        # Visit each patient once, keeping its slices together for cache reuse.
        for patient_idx in torch.randperm(self.num_patients, generator=generator).tolist():
            offset = patient_idx * SHAPE[2]

            # Shuffle axial positions and map them to fixed dataset indices.
            for axial_idx in torch.randperm(SHAPE[2], generator=generator).tolist():
                yield offset + axial_idx


def parse_args() -> argparse.Namespace:
    """Parse and normalize dataset pipeline paths."""
    parser = argparse.ArgumentParser(
        description="Extract BraTS 2019 files and preview online-preprocessed slices."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="BraTS 2019 ZIP archive path.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory for extracted HGG and LGG patient files.",
    )
    args = parser.parse_args()

    # Normalize user paths once at the CLI boundary.
    args.archive = args.archive.expanduser().resolve()
    args.raw_dir = args.raw_dir.expanduser().resolve()

    return args


def main() -> None:
    """Extract raw data and inspect one online-preprocessed batch."""
    args = parse_args()

    try:
        args.raw_dir = extract_dataset(args.archive, args.raw_dir)
        patients = discover(args.raw_dir)
        train_patients, val_patients, test_patients = split(patients, seed=SEED)
        train_dataset = BraTSSliceDataset(train_patients)
        val_dataset = BraTSSliceDataset(val_patients)
        test_dataset = BraTSSliceDataset(test_patients)

        # Preview the same patient and slice order as the first training epoch.
        train_sampler = PatientSliceSampler(len(train_patients), seed=SEED)
        train_sampler.set_epoch(1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),  # Speed CPU-to-CUDA transfers.
            drop_last=False,  # Keep final partial batch during pipeline checks.
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        images, masks, grades, patient_ids = next(iter(train_loader))
    except BadZipFile as e:
        raise SystemExit(f"error: invalid ZIP archive: {e}") from e
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: {e}") from e

    # Report raw coverage.
    print("Dataset")
    show_stats("Raw", patients, len(patients))

    # Report patient splits.
    print("\nSplits")
    show_stats("Train", train_patients, len(patients))
    show_stats("Validation", val_patients, len(patients))
    show_stats("Test", test_patients, len(patients))

    # Report slice, batch, and tensor details.
    print("\nSamples")
    print(
        f"Slices: train={len(train_dataset)}, validation={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )
    print(
        f"Batches: train={len(train_loader)}, validation={len(val_loader)}, test={len(test_loader)}"
    )
    print("First train batch")
    print(f"  Images      shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"  Masks       shape={tuple(masks.shape)}, dtype={masks.dtype}")
    print(f"  Grades      {grades}")
    print(f"  Patient IDs {patient_ids}")


if __name__ == "__main__":
    main()
