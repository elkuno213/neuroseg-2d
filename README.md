# NeuroSeg 2D

NeuroSeg 2D trains a 2D U-Net to segment brain tumors in BraTS FLAIR MRI scans. The
corresponding segmentation masks provide the training labels.

The project targets **recall > 95%** and **precision > 70%** on the validation set, with results
reported per slice and per patient. See [published results](results/README.md) for training outcomes.

## Installation

Use Python 3.10 or newer. Run these commands from the project directory:

```bash
uv sync
uv build --wheel
```

The built wheel is saved in `dist/`. For remote GPU training, upload it with
[train.ipynb](notebooks/train.ipynb). Select a GPU-enabled PyTorch kernel and run the cells in order.

Fill in the dataset and checkpoint paths first.

## Data and model files

Download BraTS 2019 to train the model. Download the checkpoints to run predictions or resume training.

Public Google Drive links will be added here.

## Data preparation

```bash
uv run python -m neuroseg_2d.data \
    --archive "<dataset-archive.zip>" \
    --raw-dir "<raw-directory>"
```

The data command extracts FLAIR images and segmentation masks, grouped by grade and patient. When a
loader first requests a patient, it prepares the full pair in memory and reuses it for every slice:

1. Keep axial slices **5 through 148** from each **240 × 240 × 155** FLAIR-mask pair. This fixed
   half-open crop `[5, 149)` produces the required **240 × 240 × 144** shape without interpolation
   or patient-specific positioning.
2. Shift the cropped affine origin to original slice 5 while preserving orientation and **1 mm
   spacing on all axes**.
3. Normalize nonzero FLAIR values using their mean and population standard deviation. Clamp the
   divisor to at least `1e-8` and keep the background at zero. Values are not limited to `[-1, 1]`.

Original files remain unchanged, and no second NIfTI dataset is written. Prepared FLAIR tensors use
float32 values; retained mask voxels stay exact and binary.

## Model and training

The model processes one axial FLAIR slice at a time and produces a tumor score for each pixel. Its
encoder uses 64, 128, 256, and 512 channels, followed by a 1024-channel bottleneck and skip decoder.

Training weights Dice and BCE losses equally. It uses AdamW and CUDA mixed precision when available.

For logits $z_i$, probabilities $p_i = \sigma(z_i)$, binary targets $y_i$, and batch pixel count $N$:

$$
L_{BCE} = -\frac{1}{N}\sum_i\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

Dice pools overlap across every pixel in one batch, with $\epsilon = 10^{-6}$:

$$
L_{Dice} = 1 - \frac{2\sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}
$$

When the complete target batch is empty, $L_{Dice}=0$. BCE still penalizes false-positive pixels. The
training objective is $L = 0.5\,(L_{BCE}+L_{Dice})$.

Correctly predicted empty batches therefore have near-zero loss without thresholding probabilities.

Patients are divided into 75% training, 12.5% validation, and 12.5% test with seed 42. Grade
proportions are preserved, and every slice from one patient stays in the same split.

Each epoch shuffles patients, then slices within each patient, while visiting every slice once. Grouped
slices reuse cached volumes.

With 144 slices and batch size 16, each patient supplies 9 batches. Other batch sizes may cross a
patient boundary.

Validation and test order remains fixed. Seed and absolute epoch reproduce sampling order after resume
with the same ordered dataset and software environment.

Reproducible sampling does not guarantee identical model results or improved metrics.

Start a fresh run:

```bash
uv run python -m neuroseg_2d.train \
    --data-dir "<raw-directory>" \
    --checkpoints-dir "<checkpoints-directory>" \
    --epochs 10 --learning-rate 1e-4 --batch-size 16 --threshold 0.5
```

Resume for 5 additional epochs:

```bash
uv run python -m neuroseg_2d.train \
    --data-dir "<raw-directory>" \
    --checkpoints-dir "<checkpoints-directory>" \
    --resume-checkpoint "<previous-run-directory>/latest.pt" \
    --epochs 5 --batch-size 16 --threshold 0.5
```

Runs are saved under `<checkpoints-directory>/runs/`. To resume, keep the previous run intact and pass
its `latest.pt` path.

Training copies the run into a new folder. It restores the model, optimizer, mixed-precision scaler,
completed epoch, and metric history.

The optimizer keeps its saved learning rate. Resume with the same dataset, preprocessing, and seed.

Losses from an older loss definition are not comparable with the current objective. Lower loss after
changing that objective does not prove better recall or precision.

- `best.pt`: model weights from the epoch with the lowest validation loss, used for inference.
- `latest.pt`: training state from the last completed epoch; only this file can resume training.
- `run.json`: configuration and final test metrics.
- `metrics.json`: epoch history with validation losses and metrics.
- `metrics.png`: training and validation curves updated after each epoch.
- `validation_comparison.png` and `test_comparison.png`: predictions for 4 fixed samples from each
  split.

The `best/` and `latest/` folders point to completed runs. Download runs before remote sessions end.

Do not resume checkpoints trained with a different preprocessing grid.

## Validation and results

After each epoch, validation records Dice, IoU, recall, and precision. Lower validation loss updates
`best.pt`.

After training, the best model is evaluated once on the test set.

- **Slice-wise:** calculate each metric per axial slice, then average defined values.
- **Patient-wise:** sum TP, FP, and FN across one patient's slices, calculate each metric, then average
  defined patient values.

Recall is `TP / (TP + FN)`. Precision is `TP / (TP + FP)`.

| Target | Prediction | Dice / IoU | Recall | Precision |
| --- | --- | ---: | ---: | ---: |
| Non-empty | Non-empty | Calculated | Calculated | Calculated |
| Non-empty | Empty | 0 | 0 | Undefined |
| Empty | Non-empty | 0 | Undefined | 0 |
| Empty | Empty | Undefined | Undefined | Undefined |

Metrics with a zero denominator are undefined and excluded from their average. `num_slices` and
`num_patients` report scope sizes; per-metric valid-value counters are omitted.

Two empty masks correctly indicate no tumor but do not define tumor overlap.

Evaluation stops if any metric has no defined values. Recall and precision must exceed their targets;
equality does not count as a pass.

See [results/](results/README.md) for training configuration and validation/test metrics.

## Inference and viewer

```bash
uv run python -m neuroseg_2d.infer \
    --image "<patient-flair.nii>" \
    --checkpoint "<completed-run-directory>/best.pt" \
    --output "<prediction.nii.gz>" \
    --batch-size 16 --threshold 0.5 --gui
```

Choose an existing output directory. Remove `--gui` to save without opening the viewer.

Inference crops and normalizes FLAIR, predicts retained slices, and applies the threshold. It restores
the binary crop to original slices 5 through 148; other slices become background.

The saved NIfTI mask keeps source shape and spatial alignment.

The viewer shows original FLAIR, prediction, and ground-truth overlays with an axial slider and
slice/volume statistics. Ground truth comes from matching `_seg.nii` or `_seg.nii.gz`; otherwise,
the third panel explains its absence.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build --wheel
```

Tests use synthetic data. The model combines all tumor labels into one class and processes slices
independently, without neighboring-slice context.

Restoring predictions to source grid preserves alignment but cannot recover preprocessing detail.

Results remain provisional. This project supports learning and is not intended for clinical use.
