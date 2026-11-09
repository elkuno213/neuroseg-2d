"""Verify NeuroSeg 2D data loading and patient-aware sampling.

Usage:
    uv run pytest tests/test_data.py
"""

__author__ = "Vu Hoi HUYNH"

from dataclasses import replace

import torch
from neuroseg_2d.constants import SHAPE
from neuroseg_2d.data import BraTSSliceDataset, Patient, PatientSliceSampler
from torch.utils.data import DataLoader


def test_dataset_and_dataloader(raw_patients: list[Patient]) -> None:
    """Return aligned slices, reuse patient cache, and stack CPU batches."""
    dataset = BraTSSliceDataset(raw_patients)

    # Read adjacent slices and confirm they reuse one cached patient volume.
    first_image, first_mask, first_grade, first_id = dataset[0]
    cached_patient = dataset.curr
    dataset[1]
    assert cached_patient is not None
    assert cached_patient[1].shape == cached_patient[2].shape == SHAPE
    assert dataset.curr is cached_patient

    # Move to next patient and confirm cache replacement without changing sample contract.
    dataset[SHAPE[2]]
    assert dataset.curr is not None
    assert dataset.curr is not cached_patient
    assert dataset.curr[0] == 1
    assert len(dataset) == SHAPE[2] * len(raw_patients)
    assert first_image.shape == first_mask.shape == (1, SHAPE[0], SHAPE[1])
    assert first_image.dtype == torch.float32
    assert first_mask.dtype == torch.uint8
    assert (first_grade, first_id) == (raw_patients[0].grade, raw_patients[0].id)

    # Stack aligned CPU tensors and retain grade and patient metadata.
    images, masks, grades, patient_ids = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
    )
    assert images.shape == masks.shape == (2, 1, SHAPE[0], SHAPE[1])
    assert images.device.type == masks.device.type == "cpu"
    assert grades == (raw_patients[0].grade,) * 2
    assert patient_ids == (raw_patients[0].id,) * 2


def test_patient_sampler(raw_patients: list[Patient]) -> None:
    """Reproduce three shuffled epochs through the Dataset and DataLoader."""
    patients = [
        replace(raw_patients[index % len(raw_patients)], id=f"Synthetic_{index}")
        for index in range(3)
    ]
    repeated_orders: list[list[list[int]]] = []

    # Recreate complete input pipeline twice to test independent reproducibility.
    for _ in range(2):
        dataset = BraTSSliceDataset(patients)
        sampler = PatientSliceSampler(len(patients), seed=42)
        loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=0)
        assert len(dataset) == len(sampler) == 432
        assert len(loader) == 27
        epoch_orders: list[list[int]] = []

        for epoch in range(1, 4):
            # Visit every slice once in shuffled, contiguous patient groups.
            sampler.set_epoch(epoch)
            rng_state = torch.get_rng_state()
            sample_indices = list(sampler)
            assert torch.equal(torch.get_rng_state(), rng_state)
            assert sorted(sample_indices) == list(range(len(dataset)))
            for group_start in range(0, len(sample_indices), SHAPE[2]):
                patient_indices = sample_indices[group_start : group_start + SHAPE[2]]
                assert len({index // SHAPE[2] for index in patient_indices}) == 1
                assert patient_indices != sorted(patient_indices)

            # Stream batches in sampler order and verify tensors against cached volumes.
            cached_patient = None
            num_loaded = 0
            for images, masks, grades, patient_ids in loader:
                batch_indices = sample_indices[num_loaded : num_loaded + len(images)]
                batch_samples = [dataset.samples[index] for index in batch_indices]
                patient_idx = batch_samples[0][0]
                axial_indices = [axial_idx for _, axial_idx in batch_samples]
                assert {index for index, _ in batch_samples} == {patient_idx}
                assert grades == (patients[patient_idx].grade,) * len(images)
                assert patient_ids == (patients[patient_idx].id,) * len(images)
                assert images.shape == masks.shape == (16, 1, SHAPE[0], SHAPE[1])
                assert dataset.curr is not None
                assert dataset.curr[0] == patient_idx

                # Reuse cache inside one patient group and replace it at next boundary.
                if cached_patient is not None and cached_patient[0] == patient_idx:
                    assert dataset.curr is cached_patient
                else:
                    assert dataset.curr is not cached_patient
                _, image_3d, mask_3d = dataset.curr
                torch.testing.assert_close(
                    images,
                    image_3d[:, :, axial_indices].permute(2, 0, 1).unsqueeze(1),
                )
                torch.testing.assert_close(
                    masks,
                    mask_3d[:, :, axial_indices].permute(2, 0, 1).unsqueeze(1),
                )
                cached_patient = dataset.curr
                num_loaded += len(images)

            assert num_loaded == len(dataset)
            epoch_orders.append(sample_indices)
        repeated_orders.append(epoch_orders)

    # Epochs differ, while rebuilding same pipeline reproduces every order.
    assert len({tuple(order) for order in repeated_orders[0]}) == 3
    patient_orders = [
        tuple(index // SHAPE[2] for index in order[:: SHAPE[2]]) for order in repeated_orders[0]
    ]
    assert len(set(patient_orders)) > 1
    assert repeated_orders[0] == repeated_orders[1]
