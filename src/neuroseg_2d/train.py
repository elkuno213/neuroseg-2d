"""Train NeuroSeg 2D on online-preprocessed BraTS FLAIR slices.

Usage:
    python -m neuroseg_2d.train [options]
"""

__author__ = "Vu Hoi HUYNH"

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from neuroseg_2d.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    SEED,
    THRESHOLD,
    WEIGHT_DECAY,
)
from neuroseg_2d.data import BraTSSliceDataset, PatientSliceSampler, discover, split
from neuroseg_2d.helpers import (
    calculate_patient_metrics,
    calculate_slice_metrics,
    create_run_dir,
    print_metrics,
    save_comparison,
    save_metrics,
    save_report,
    select_comparison_indices,
    update_run_reference,
)
from neuroseg_2d.model import UNet
from neuroseg_2d.visualize import save_metrics_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed training random generators for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True, slots=True)
class Result:
    """Summarize completed training epochs and best validation result."""

    epoch: int
    best_epoch: int
    best_val_loss: float
    epoch_metrics: list[dict[str, object]]


def resume_run(
    checkpoint_file: Path,
    run_dir: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Result:
    """Copy one source run and restore its latest training state.

    Args:
        checkpoint_file: Source run's ``latest.pt`` file.
        run_dir: New run directory receiving copied artifacts.
        model: Model receiving saved weights.
        optimizer: Optimizer receiving saved state.
        scaler: AMP scaler receiving saved state.
        device: Device used to map checkpoint tensors.

    Returns:
        Restored epoch, best-result tracking, and metric history.
    """
    # Copy source artifacts before changing continuation results.
    shutil.copytree(checkpoint_file.parent, run_dir, dirs_exist_ok=True)

    # Restore latest model, optimizer, scaler, and epoch tracking.
    checkpoint: dict[str, object] = torch.load(
        run_dir / "latest.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(cast(dict[str, Tensor], checkpoint["model"]))
    optimizer.load_state_dict(cast(dict[str, object], checkpoint["optimizer"]))
    scaler.load_state_dict(cast(dict[str, object], checkpoint["scaler"]))

    # Restore metric history already stored in current schema.
    epoch_metrics = cast(
        list[dict[str, object]],
        json.loads((run_dir / "metrics.json").read_text()),
    )

    return Result(
        epoch=cast(int, checkpoint["epoch"]),
        best_epoch=cast(int, checkpoint["best_epoch"]),
        best_val_loss=cast(float, checkpoint["best_validation_loss"]),
        epoch_metrics=epoch_metrics,
    )


class DiceBCELoss(nn.Module):
    """Combine all-pixel BCE and target-present-batch soft Dice equally."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return hybrid loss; a target-empty batch contributes BCE only."""
        # Pool float32 overlap across the batch; skip Dice when the complete target is empty.
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        cardinality = (probs + targets).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss *= targets.any()

        # Apply BCE to every pixel, including empty slices, and combine both losses equally.
        bce_loss = self.bce(logits, targets)
        return 0.5 * (bce_loss + dice_loss)


def train(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor, str, str]],
    train_sampler: PatientSliceSampler,
    val_loader: DataLoader[tuple[Tensor, Tensor, str, str]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epochs: int,
    threshold: float,
    run_dir: Path,
    result: Result,
) -> Result:
    """Train all requested epochs and return one result summary."""

    # Continue saved training state or start from fresh defaults.
    best_val_loss = result.best_val_loss
    best_epoch = result.best_epoch
    epoch_metrics = list(result.epoch_metrics)
    first_epoch = result.epoch + 1
    last_epoch = result.epoch + epochs

    non_blocking = device.type == "cuda"  # Use pinned memory for faster GPU transfers.
    amp_enabled = scaler.is_enabled()

    # Loop over epochs, then batches, and update model weights.
    for epoch in range(first_epoch, last_epoch + 1):
        # Shuffle patients and their slices using the absolute epoch, including on resume.
        train_sampler.set_epoch(epoch)
        model.train()

        # Initialize epoch totals and progress bar.
        train_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{last_epoch}",
            unit="batch",
        )

        for images, masks, _, _ in progress:
            # Move data to the training device and convert masks to float32 for BCEWithLogitsLoss.
            images = images.to(device, non_blocking=non_blocking)
            masks = masks.to(device, dtype=torch.float32, non_blocking=non_blocking)

            # Clear gradients left from the previous batch.
            optimizer.zero_grad()

            # Forward pass: use mixed precision on CUDA for prediction and loss.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = loss_fn(logits, masks)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss is not finite")

            # Backward pass: scale gradients, update model weights, then adjust the scaler.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate one scalar loss per batch.
            train_loss += loss.item()

            # Update progress bar with current loss for this batch.
            progress.set_postfix(loss=loss.item())

        # Compute epoch metrics and evaluate the full validation set.
        train_loss /= len(train_loader)
        val_metrics = validate(model, val_loader, loss_fn, device, threshold)
        val_loss = cast(float, val_metrics["loss"])
        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "slice": val_metrics["slice"],
                "patient": val_metrics["patient"],
            }
        )
        print_metrics(epoch_metrics)

        # Update best tracking and keep best weights dedicated to inference.
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best.pt")

        # Preserve full latest state and readable history for future continuation.
        latest: dict[str, object] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
        }
        torch.save(latest, run_dir / "latest.pt")
        save_metrics(run_dir, epoch_metrics)
        save_metrics_plot(run_dir, epoch_metrics)

    return Result(
        epoch=last_epoch,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        epoch_metrics=epoch_metrics,
    )


def validate(
    model: nn.Module,
    data_loader: DataLoader[tuple[Tensor, Tensor, str, str]],
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, object]:
    """Return loss plus slice-wise and patient-wise segmentation metrics."""

    # Set up pinned memory for faster GPU transfers if available.
    non_blocking = device.type == "cuda"

    # Set up read-only evaluation and full-loader totals.
    model.eval()

    # Initialize stats.
    evaluation_loss = 0.0
    tp_list: list[Tensor] = []
    fp_list: list[Tensor] = []
    fn_list: list[Tensor] = []
    patient_keys: list[str] = []

    with torch.inference_mode():
        for images, masks, grades, patient_ids in data_loader:
            # Move one batch to the evaluation device.
            images = images.to(device, non_blocking=non_blocking)
            masks = masks.to(device, dtype=torch.float32, non_blocking=non_blocking)

            # Predict raw logits and measure batch loss.
            logits = model(images)
            loss = loss_fn(logits, masks)
            if not torch.isfinite(loss):
                raise ValueError("Validation loss is not finite")

            # Accumulate one scalar loss per batch.
            evaluation_loss += loss.item()

            # Convert logits to binary predictions.
            predictions = torch.sigmoid(logits) >= threshold
            targets = masks >= 0.5

            # Flatten (B, C, H, W) to (B, C*H*W), then sum into one count per slice: (B,).
            tp_list.append((predictions & targets).flatten(1).sum(dim=1).cpu())
            fp_list.append((predictions & ~targets).flatten(1).sum(dim=1).cpu())
            fn_list.append((~predictions & targets).flatten(1).sum(dim=1).cpu())

            patient_keys.extend(
                f"{grade}/{patient_id}"
                for grade, patient_id in zip(grades, patient_ids, strict=True)
            )

    # Join batch count tensors into one slice-aligned tensor (N,) for metric calculation.
    tp = torch.cat(tp_list)
    fp = torch.cat(fp_list)
    fn = torch.cat(fn_list)

    return {
        "loss": evaluation_loss / len(data_loader),
        "slice": calculate_slice_metrics(tp, fp, fn),
        "patient": calculate_patient_metrics(
            tp,
            fp,
            fn,
            patient_keys,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse, normalize, and validate training arguments."""
    parser = argparse.ArgumentParser(description="Train NeuroSeg 2D on extracted BraTS data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Extracted raw directory containing HGG and LGG patient folders.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Root directory for timestamped runs and best/latest references.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Optional latest.pt checkpoint to continue from.",
    )
    args = parser.parse_args()

    # Normalize user paths once at the CLI boundary.
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoints_dir = args.checkpoints_dir.expanduser().resolve()
    args.resume_checkpoint = (
        args.resume_checkpoint.expanduser().resolve()
        if args.resume_checkpoint is not None
        else None
    )

    # Validate paths and numeric training options.
    if not args.data_dir.is_dir():
        parser.error(f"data directory does not exist: {args.data_dir}")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.resume_checkpoint}")
    if args.resume_checkpoint is not None and args.resume_checkpoint.name != "latest.pt":
        parser.error("--resume-checkpoint must point to latest.pt")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    return args


def main() -> None:
    """Build the data, model, and optimizer, then start training."""
    args = parse_args()

    # Seed every random source before constructing data and training state.
    set_seed(SEED)

    # Read and split data for training, validation, and testing.
    patients = discover(args.data_dir)
    train_patients, val_patients, test_patients = split(patients, seed=SEED)
    train_dataset = BraTSSliceDataset(train_patients)
    val_dataset = BraTSSliceDataset(val_patients)
    test_dataset = BraTSSliceDataset(test_patients)
    if any(len(dataset) == 0 for dataset in (train_dataset, val_dataset, test_dataset)):
        raise ValueError("Training, validation, and test datasets must not be empty")

    train_sampler = PatientSliceSampler(len(train_patients), seed=SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Build model, loss, optimizer and scaler.
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=DEVICE.type == "cuda")

    # Create a new run with fresh state, then restore latest state when resuming.
    run_dir = create_run_dir(args.checkpoints_dir)
    result = Result(
        epoch=0,
        best_epoch=0,
        best_val_loss=float("inf"),
        epoch_metrics=[],
    )
    if args.resume_checkpoint is not None:
        result = resume_run(args.resume_checkpoint, run_dir, model, optimizer, scaler, DEVICE)

    # Train requested additional epochs and persist best/latest state.
    result = train(
        model=model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaler=scaler,
        device=DEVICE,
        epochs=args.epochs,
        threshold=args.threshold,
        run_dir=run_dir,
        result=result,
    )

    # Load the best weights and evaluate on the test set.
    best_weights: dict[str, Tensor] = torch.load(
        run_dir / "best.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(best_weights)
    test_metrics = validate(model, test_loader, loss_fn, DEVICE, args.threshold)
    print_metrics([test_metrics])

    # Save run report, metrics, and run references.
    report: dict[str, object] = {
        "batch_size": args.batch_size,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_val_loss,
        "data_dir": str(args.data_dir),
        "device": str(DEVICE),
        "epochs": args.epochs,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seed": SEED,
        "threshold": args.threshold,
        "test_metrics": test_metrics,
    }
    save_report(run_dir, report=report)
    save_metrics(run_dir, result.epoch_metrics)
    update_run_reference(
        args.checkpoints_dir,
        "best",
        run_dir,
        best_validation_loss=result.best_val_loss,
    )
    update_run_reference(
        args.checkpoints_dir,
        "latest",
        run_dir,
        best_validation_loss=result.best_val_loss,
    )
    print(f"Run artifacts: {run_dir}")

    # Save deterministic comparison grids after durable run artifacts.
    for split_name, dataset in (("validation", val_dataset), ("test", test_dataset)):
        samples = [dataset[index] for index in select_comparison_indices(dataset, split_name)]
        images = torch.stack([image for image, _, _, _ in samples])
        targets = torch.stack([target for _, target, _, _ in samples])
        with torch.inference_mode():
            logits = model(images.to(DEVICE))
            predictions = torch.sigmoid(logits) >= args.threshold
        save_comparison(
            run_dir / f"{split_name}_comparison.png",
            images,
            targets,
            predictions,
        )


if __name__ == "__main__":
    main()
