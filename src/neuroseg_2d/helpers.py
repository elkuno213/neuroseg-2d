"""Calculate metrics and persist training-run artifacts.

Usage:
    from neuroseg_2d.helpers import calculate_patient_metrics
"""

from __future__ import annotations

__author__ = "Vu Hoi HUYNH"

import json
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Literal, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from neuroseg_2d.constants import METRIC_FIELDS, REPORT_FIELDS

# ──────────────────────────────────────────────────────────────────────────────────────────────── #
# Segmentation metric calculation. #


def _calculate_metrics(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
) -> dict[str, float]:
    """Average defined segmentation metrics across aligned count tensors."""
    if not (
        tp.ndim == fp.ndim == fn.ndim == 1  # Flat count vectors.
        and tp.shape == fp.shape == fn.shape  # Same evaluation units.
        and tp.numel() > 0  # At least one slice or patient.
    ):
        raise ValueError("TP, FP, and FN must be non-empty aligned 1D tensors")

    # Convert integer counts once for metric division.
    tp = tp.to(dtype=torch.float64)
    fp = fp.to(dtype=torch.float64)
    fn = fn.to(dtype=torch.float64)

    formulas = {  # Each tuple stores (numerator, denominator).
        "dice": (2 * tp, 2 * tp + fp + fn),
        "iou": (tp, tp + fp + fn),
        "recall": (tp, tp + fn),
        "precision": (tp, tp + fp),
    }
    metrics: dict[str, float] = {}
    for name, (num, den) in formulas.items():
        # Treat zero denominators as undefined, then average only defined values.
        valid = den > 0
        if not valid.any():
            raise ValueError(f"{name} has no defined values")
        metrics[name] = float((num[valid] / den[valid]).mean().item())

    return metrics


def calculate_slice_metrics(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
) -> dict[str, float | int]:
    """Average segmentation metrics over axial slices."""
    return {"num_slices": tp.numel(), **_calculate_metrics(tp, fp, fn)}


def calculate_patient_metrics(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    patient_keys: list[str],
) -> dict[str, float | int]:
    """Sum slice counts per patient, then average patient metrics."""
    if len(patient_keys) != tp.numel():
        raise ValueError("Patient keys must align with slice counts")

    # Group flat slice positions by patient key for later count summation.
    patient_indices: dict[str, list[int]] = {}
    for index, patient_key in enumerate(patient_keys):
        patient_indices.setdefault(patient_key, []).append(index)

    # Sum slice counts into one TP, FP, and FN value per patient.
    patient_tp = torch.stack([tp[indices].sum() for indices in patient_indices.values()])
    patient_fp = torch.stack([fp[indices].sum() for indices in patient_indices.values()])
    patient_fn = torch.stack([fn[indices].sum() for indices in patient_indices.values()])

    return {
        "num_patients": patient_tp.numel(),
        **_calculate_metrics(patient_tp, patient_fp, patient_fn),
    }


# ──────────────────────────────────────────────────────────────────────────────────────────────── #
# Metric reporting and run persistence. #


def print_metrics(metrics: list[dict[str, object]], show_latest: bool = True) -> None:
    """Print test metrics or selected epoch history."""
    # Select latest epoch by default, or full history when requested.
    rows = metrics[-1:] if show_latest else metrics
    for row in rows:
        # Label training rows by epoch and final evaluation as Test.
        epoch = row.get("epoch")
        label = f"Epoch {epoch}" if isinstance(epoch, int) else "Test"

        # Print whichever training, validation, or test losses exist in this row.
        losses = [
            f"{name}={value:.6f}"
            for name in ("train_loss", "validation_loss", "loss")
            if isinstance((value := row.get(name)), float)
        ]
        print(f"{label}: {', '.join(losses)}")

        # Print slice-wise and patient-wise metrics on separate lines.
        for scope_name in ("slice", "patient"):
            scope = row.get(scope_name)
            if not isinstance(scope, dict):
                raise ValueError(f"{scope_name} metrics must be a dictionary")
            values = ", ".join(
                f"{name}={value:.6f}" if isinstance(value, float) else f"{name}={value}"
                for name, value in scope.items()
            )
            print(f"  {scope_name.title()}: {values}")


def create_run_dir(
    checkpoints_dir: Path,
    timestamp: datetime | None = None,
) -> Path:
    """Create one timestamped run directory without overwriting an existing run.

    Args:
        checkpoints_dir: Root directory for run artifacts and reference metadata.
        timestamp: Optional timestamp used by deterministic tests.

    Returns:
        Newly created run directory.
    """
    # Keep every run under one portable checkpoints root.
    runs_dir = checkpoints_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir()
    return run_dir


def save_metrics(
    run_dir: Path,
    metrics: list[dict[str, object]],
) -> None:
    """Validate and save ordered epoch metrics."""
    # Require non-empty history with unique ascending integer epochs.
    if not metrics:
        raise ValueError("Epoch metrics must not be empty")

    # Extract integer epoch numbers in one pass.
    epochs = [epoch for row in metrics if isinstance((epoch := row.get("epoch")), int)]
    if len(epochs) != len(metrics):
        raise ValueError("Epoch must be an integer")
    if epochs != sorted(set(epochs)):
        raise ValueError("Epoch metrics must have unique ascending epoch numbers")

    # Enforce top-level epoch fields.
    if any(set(row) != set(METRIC_FIELDS) for row in metrics):
        raise ValueError(f"Epoch metrics must contain fields {METRIC_FIELDS}")

    # Persist ordered history in the existing run directory.
    (run_dir / "metrics.json").write_text(
        json.dumps([dict(row) for row in metrics], indent=2) + "\n"
    )


def save_report(
    run_dir: Path,
    report: dict[str, object],
) -> None:
    """Write run configuration and final results."""
    if set(report) != set(REPORT_FIELDS):
        raise ValueError(f"Report must contain fields {REPORT_FIELDS}")

    (run_dir / "run.json").write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")


def update_run_reference(
    checkpoints_dir: Path,
    reference_name: Literal["best", "latest"],
    run_dir: Path,
    best_validation_loss: float,
) -> bool:
    """Atomically update latest or improved-best run metadata.

    Args:
        checkpoints_dir: Root directory containing run and reference directories.
        reference_name: Reference name, either ``best`` or ``latest``.
        run_dir: Completed timestamped run directory.
        best_validation_loss: Best validation loss reached by that run.

    Returns:
        ``True`` when reference changed, otherwise ``False``.

    Raises:
        ValueError: If run directory is outside checkpoints root or reference is malformed.
    """
    # Relative paths keep references valid after moving or mounting checkpoints.
    try:
        relative_run = run_dir.relative_to(checkpoints_dir).as_posix()
    except ValueError as error:
        raise ValueError(f"Run directory must be inside {checkpoints_dir}: {run_dir}") from error

    reference_dir = checkpoints_dir / reference_name
    reference_file = reference_dir / "target.json"

    # Keep the current best reference unless validation loss improves.
    if reference_name == "best" and reference_file.is_file():
        current = json.loads(reference_file.read_text())
        current_loss = current.get("best_validation_loss")
        if not isinstance(current_loss, int | float):
            raise ValueError(f"Invalid best reference: {reference_file}")
        if best_validation_loss >= float(current_loss):
            return False

    # Replace reference atomically so interrupted writes cannot corrupt it.
    reference_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = reference_dir / ".target.json.tmp"
    tmp_file.write_text(
        json.dumps(
            {
                "best_validation_loss": best_validation_loss,
                "run": relative_run,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    tmp_file.replace(reference_file)

    return True


# ──────────────────────────────────────────────────────────────────────────────────────────────── #
# Deterministic comparison images. #


class _SliceDataset(Protocol):
    """Provide indexed slice samples for comparison selection."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str, str]: ...


def select_comparison_indices(
    dataset: _SliceDataset,
    split_name: str,
) -> list[int]:
    """Select four deterministic tumor-positive slices across one split.

    Args:
        dataset: Validation or test slice dataset.
        split_name: Split label used in error messages.

    Returns:
        Dataset indices at the 20th, 40th, 60th, and 80th percentiles.

    Raises:
        ValueError: If the split has fewer than four tumor-positive slices.
    """
    positive_indices = [index for index in range(len(dataset)) if dataset[index][1].any().item()]
    if len(positive_indices) < 4:
        raise ValueError(f"{split_name} split must contain at least four tumor-positive slices")

    return [
        positive_indices[ceil(percentile * len(positive_indices)) - 1]
        for percentile in (0.2, 0.4, 0.6, 0.8)
    ]


def save_comparison(
    output_file: Path,
    images: Tensor,
    targets: Tensor,
    predictions: Tensor,
) -> None:
    """Write one four-sample segmentation comparison grid.

    Args:
        output_file: Destination PNG path.
        images: Four channel-first FLAIR samples.
        targets: Four matching binary target masks.
        predictions: Four matching binary predicted masks.

    Raises:
        ValueError: If tensors do not share shape ``(4, 1, H, W)``.
    """
    # Move report tensors to CPU arrays without changing training tensors.
    image_array = images.detach().cpu().numpy()
    target_array = targets.detach().cpu().bool().numpy()
    prediction_array = predictions.detach().cpu().bool().numpy()
    if (
        image_array.shape != target_array.shape
        or image_array.shape != prediction_array.shape
        or image_array.ndim != 4
        or image_array.shape[:2] != (4, 1)
    ):
        raise ValueError(
            "Images, targets, and predictions must share shape (4, 1, H, W), "
            f"got {image_array.shape}, {target_array.shape}, "
            f"and {prediction_array.shape}"
        )

    # Show four samples as FLAIR, target, prediction, and overlay columns.
    figure, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    for row in range(4):
        image = image_array[row, 0]
        target = target_array[row, 0]
        prediction = prediction_array[row, 0]
        panels = (
            (image, "FLAIR", "gray"),
            (target, "Target", "gray"),
            (prediction, "Prediction", "gray"),
        )
        for axis, (values, title, color_map) in zip(axes[row, :3], panels, strict=True):
            axis.imshow(values, cmap=color_map)
            if row == 0:
                axis.set_title(title)
            axis.axis("off")
        axes[row, 3].imshow(image, cmap="gray")
        axes[row, 3].imshow(
            np.ma.masked_where(~prediction, prediction),
            cmap="Reds",
            alpha=0.45,
        )
        if row == 0:
            axes[row, 3].set_title("Overlay")
        axes[row, 3].axis("off")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150)
    plt.close(figure)
