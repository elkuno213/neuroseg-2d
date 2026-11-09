"""Visualize training metrics and source-grid inference results.

Usage:
    from neuroseg_2d.visualize import save_metrics_plot, show_gui
"""

__author__ = "Vu Hoi HUYNH"

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.widgets import Slider

FONT_FAMILY = "Noto Sans"


def save_metrics_plot(run_dir: Path, metrics: list[dict[str, object]]) -> None:
    """Save training losses and slice-wise and patient-wise validation metrics."""
    # Extract ordered history once for all panels.
    epochs = [cast(int, row["epoch"]) for row in metrics]
    train_loss = [cast(float, row["train_loss"]) for row in metrics]
    validation_loss = [cast(float, row["validation_loss"]) for row in metrics]
    slices = [cast(dict[str, float], row["slice"]) for row in metrics]
    patient = [cast(dict[str, float], row["patient"]) for row in metrics]

    # Draw loss, then compare slice-wise and patient-wise segmentation metrics.
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    axes[0, 0].plot(epochs, train_loss, marker="o", markersize=3, label="Train")
    axes[0, 0].plot(epochs, validation_loss, marker="o", markersize=3, label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend(frameon=False)

    panels = (("Dice", "dice", None), ("Recall", "recall", 0.95), ("Precision", "precision", 0.70))
    for axis, (title, field, target) in zip(axes.flat[1:], panels, strict=True):
        axis.plot(
            epochs,
            [row[field] for row in slices],
            linestyle="--",
            marker="o",
            markersize=3,
            label="Slice",
        )
        axis.plot(
            epochs,
            [row[field] for row in patient],
            marker="o",
            markersize=3,
            label="Patient",
        )
        if target is not None:
            axis.axhline(target, color="#dc2626", linestyle=":", label=f"Target {target:.2f}")
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.legend(frameon=False)

    # Apply shared presentation and replace one compact run artifact.
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    figure.savefig(run_dir / "metrics.png", dpi=150)
    plt.close(figure)


def load_ground_truth(
    image_file: Path,
    src_shape: tuple[int, ...],
    src_affine: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Load a source-aligned sibling segmentation when available."""
    name = image_file.name
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    stem = name.removesuffix(suffix)
    if not stem.endswith("_flair"):
        return None, "Ground truth unavailable: input name does not end with _flair."

    seg_file = image_file.with_name(f"{stem.removesuffix('_flair')}_seg{suffix}")
    if not seg_file.is_file():
        return None, f"Ground truth unavailable: {seg_file.name} not found."

    seg_nifti = nib.Nifti1Image.from_filename(seg_file)
    if seg_nifti.shape != src_shape or not np.allclose(
        seg_nifti.affine,
        src_affine,
        rtol=0.0,
        atol=1e-6,
    ):
        return None, f"Ground truth unavailable: {seg_file.name} geometry mismatch."

    return np.asarray(seg_nifti.dataobj) > 0, None


def _format_statistics(
    prediction: np.ndarray,
    target: np.ndarray | None,
    scale: float,
    unit: str,
) -> tuple[str, str, str, str, str]:
    """Format physical tumor size and available overlap metrics."""
    prediction_size = f"{np.count_nonzero(prediction) * scale:,.1f} {unit}"
    target_size = f"{np.count_nonzero(target) * scale:,.1f} {unit}" if target is not None else "N/A"
    if target is None:
        return prediction_size, target_size, "N/A", "N/A", "N/A"

    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    return (
        prediction_size,
        target_size,
        f"{2 * tp / (2 * tp + fp + fn):.3f}" if 2 * tp + fp + fn else "N/A",
        f"{tp / (tp + fn):.3f}" if tp + fn else "N/A",
        f"{tp / (tp + fp):.3f}" if tp + fp else "N/A",
    )


def _create_stats_panel(axis: Axes) -> tuple[Text, list[Text], list[Text]]:
    """Create aligned slice and volume statistics in one rectangular panel."""
    axis.set_facecolor("#ffffff")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    labels = ("Prediction", "Ground Truth", "Dice", "Recall", "Precision")
    sections = []
    for title, top in (("", 0.95), ("WHOLE VOLUME", 0.47)):
        title_artist = axis.text(
            0.07,
            top,
            title,
            fontsize=9,
            fontweight="bold",
            fontfamily=FONT_FAMILY,
            color="#64748b",
        )
        values = []
        for label, y in zip(labels, np.linspace(top - 0.12, top - 0.4, len(labels)), strict=True):
            axis.text(
                0.07,
                y,
                label,
                fontsize=9,
                fontfamily=FONT_FAMILY,
                color="#64748b",
                va="center",
            )
            values.append(
                axis.text(
                    0.93,
                    y,
                    "",
                    fontsize=10,
                    fontweight="bold",
                    fontfamily=FONT_FAMILY,
                    color="#0f172a",
                    ha="right",
                    va="center",
                )
            )
        sections.append((title_artist, values))

    axis.axhline(0.5, color="#cbd5e1", linewidth=0.8)
    return sections[0][0], sections[0][1], sections[1][1]


def show_gui(
    image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray | None,
    target_message: str | None,
    patient_name: str,
    src_affine: np.ndarray,
    threshold: float,
) -> None:
    """Show synchronized source-grid overlays and physical tumor statistics."""
    # Fix display values and physical scales once for the whole source volume.
    vmin, vmax = np.percentile(image[image != 0], (1, 99))
    initial = image.shape[2] // 2
    spacing = np.linalg.norm(src_affine[:3, :3], axis=0)
    pixel_area = np.linalg.norm(np.cross(src_affine[:3, 0], src_affine[:3, 1]))
    voxel_volume = abs(np.linalg.det(src_affine[:3, :3])) / 1000

    # Create source-grid panels under a clear patient and metadata header.
    figure, all_axes = plt.subplots(
        1,
        4,
        figsize=(16, 6.5),
        layout="compressed",
        gridspec_kw={"width_ratios": (1, 1, 1, 0.75)},
    )
    axes = all_axes[:3]
    stats_axis = all_axes[3]
    figure.patch.set_facecolor("#f8fafc")
    figure.text(
        0.03,
        0.96,
        patient_name,
        fontsize=16,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
        color="#0f172a",
        va="top",
    )
    figure.text(
        0.03,
        0.91,
        f"{' × '.join(map(str, image.shape))} voxels   •   "
        f"{' × '.join(f'{value:.2f}' for value in spacing)} mm spacing   •   "
        f"threshold {threshold:.2f}",
        fontsize=10,
        fontfamily=FONT_FAMILY,
        color="#64748b",
        va="top",
    )
    image_artists = [
        axes[index].imshow(image[:, :, initial], cmap="gray", vmin=vmin, vmax=vmax)
        for index in range(2 if target is None else 3)
    ]
    prediction_artist = axes[1].imshow(
        np.ma.masked_where(~prediction[:, :, initial].astype(bool), prediction[:, :, initial]),
        cmap="Reds",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    for axis, title in zip(axes, ("Original FLAIR", "Prediction", "Ground truth"), strict=True):
        axis.set_title(title, color="#334155", fontweight="bold", fontfamily=FONT_FAMILY)
        axis.axis("off")

    target_artist = None
    if target is None:
        axes[2].text(
            0.5,
            0.5,
            target_message,
            ha="center",
            va="center",
            wrap=True,
            fontfamily=FONT_FAMILY,
        )
    else:
        target_artist = axes[2].imshow(
            np.ma.masked_where(~target[:, :, initial], target[:, :, initial]),
            cmap="Greens",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

    # Keep both statistic scopes in one layout-managed sidebar.
    slice_title, slice_values, volume_values = _create_stats_panel(stats_axis)
    for artist, statistic in zip(
        volume_values,
        _format_statistics(prediction, target, voxel_volume, "mL"),
        strict=True,
    ):
        artist.set_text(statistic)

    slider_axis = figure.add_axes((0.12, 0.05, 0.55, 0.035))
    slider = Slider(slider_axis, "Axial slice", 0, image.shape[2] - 1, valinit=initial, valstep=1)
    slider.label.set_fontfamily(FONT_FAMILY)
    slider.valtext.set_fontfamily(FONT_FAMILY)

    # Update existing artists in place so all panels remain synchronized.
    def update(value: float) -> None:
        index = int(value)
        for artist in image_artists:
            artist.set_data(image[:, :, index])
        prediction_slice = prediction[:, :, index]
        prediction_artist.set_data(
            np.ma.masked_where(~prediction_slice.astype(bool), prediction_slice)
        )
        if target_artist is not None and target is not None:
            target_slice = target[:, :, index]
            target_artist.set_data(np.ma.masked_where(~target_slice, target_slice))
        target_slice = target[:, :, index] if target is not None else None
        slice_title.set_text(f"SLICE {index} / {image.shape[2] - 1}")
        for artist, statistic in zip(
            slice_values,
            _format_statistics(prediction_slice, target_slice, pixel_area, "mm²"),
            strict=True,
        ):
            artist.set_text(statistic)
        figure.canvas.draw_idle()

    update(initial)
    slider.on_changed(update)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(figure)
