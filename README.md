# NeuroSeg 2D

NeuroSeg 2D trains a 2D U-Net to segment brain tumors in BraTS FLAIR MRI scans. BraTS
segmentation masks provide the training labels.

The project targets **recall > 95%** and **precision > 70%** on the validation set, with results
reported per slice and per patient. See [published results](results/README.md) for training outcomes.

## Installation

Use Python 3.10 or newer. From the project directory, run:

```bash
uv sync
uv build --wheel
```

The wheel is saved in `dist/`. For remote GPU training, open [train.ipynb](notebooks/train.ipynb),
fill in the dataset and checkpoint paths, then run the cells in order with a GPU-enabled kernel.

## Data and model files

Download BraTS 2019 for training. Download `best.pt`
[here](https://drive.google.com/drive/folders/1wTXW9qtk85E2wrD0iCeCGspBsnMO_aqx?usp=sharing)
to run inference.

## Data preparation

```bash
uv run python -m neuroseg_2d.data \
    --archive "<dataset-archive.zip>" \
    --raw-dir "<raw-directory>"
```

This command extracts FLAIR images and masks by grade and patient. On first access, the dataset
prepares one patient in memory and reuses that volume for every slice:

1. Keep axial slices **5 through 148**. This fixed crop `[5, 149)` converts each
   **240 × 240 × 155** pair to **240 × 240 × 144** without interpolation.
2. Move the affine origin to original slice 5 while preserving orientation and **1 mm spacing**.
3. Apply Z-score normalization to nonzero FLAIR voxels and keep background voxels at zero.

Raw files remain unchanged. Prepared FLAIR tensors use `float32`; masks remain binary.

## Model and training

The model processes one axial FLAIR slice at a time and predicts a tumor score for each pixel.
Training uses AdamW, mixed precision on CUDA, and equal Tversky and BCE loss weights.

For logits $z_i$, probabilities $p_i = \sigma(z_i)$, binary targets $y_i$, and batch pixel count $N$:

$$
L_{BCE} = -\frac{1}{N}\sum_i\left[w y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
$$

The positive weight $w$ defaults to 2. Tversky pools soft counts across one batch,
with $TP=\sum p_i y_i$, $FP=\sum p_i(1-y_i)$, $FN=\sum(1-p_i)y_i$ and $\epsilon=10^{-6}$:

$$
L_{Tversky} = 1 - \frac{TP + \epsilon}{TP + 0.3FP + 0.7FN + \epsilon}
$$

For an empty target batch, $L_{Tversky}=0$ and BCE still penalizes false positives. The full objective is
$L = 0.5\,(L_{BCE}+L_{Tversky})$. Dice remains an evaluation metric.

Patients are split into 80% training, 10% validation, and 10% test with seed 42. Grade proportions
are preserved, and all slices from one patient stay together. Each training epoch shuffles patients,
then their slices, while validation and test order remain fixed.

Start a fresh run:

```bash
uv run python -m neuroseg_2d.train \
    --data-dir "<raw-directory>" \
    --checkpoints-dir "<checkpoints-directory>" \
    --epochs 10 --learning-rate 1e-4 --batch-size 16 --threshold 0.5
```

The scheduler halves the learning rate after two consecutive non-improving validation checks,
down to `1e-6`. `--learning-rate` sets the starting rate for fresh training; resuming restores it.

Resume for 5 additional epochs:

```bash
uv run python -m neuroseg_2d.train \
    --data-dir "<raw-directory>" \
    --checkpoints-dir "<checkpoints-directory>" \
    --resume-checkpoint "<previous-run-directory>/latest.pt" \
    --epochs 5 --batch-size 16 --threshold 0.5
```

Runs are saved under `<checkpoints-directory>/runs/`. Resuming creates a new run and restores model,
optimizer, mixed-precision scaler, scheduler, completed epoch, and metric history from `latest.pt`.
Use matching loss settings, batch size, patient assignments, preprocessing, and seed.

- `best.pt`: weights from the lowest validation loss, used for inference.
- `latest.pt`: complete state from the last epoch, used to resume training.
- `run.json`: training settings and patient assignments, followed by best-epoch and test results.
- `metrics.json` and `metrics.png`: per-epoch losses, learning rates, metrics, and training curves.
- `validation_comparison.png` and `test_comparison.png`: four fixed examples from each split.

The `best/` and `latest/` folders point to completed runs. Never resume a checkpoint trained with a
different preprocessing grid.

## Validation and results

Each epoch records validation loss, Dice, IoU, recall, and precision. Lower validation loss updates
`best.pt`. After training, the best model is evaluated once on the test set.

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

Metrics with a zero denominator are undefined and excluded from the average. Two empty masks correctly
indicate no tumor but do not define tumor overlap. Evaluation stops when a metric has no defined value.
Recall and precision must strictly exceed their targets.

See [results/](results/README.md) for training configuration and validation/test metrics.

## Inference and viewer

```bash
uv run python -m neuroseg_2d.infer \
    --image "<patient-flair.nii>" \
    --checkpoint "<completed-run-directory>/best.pt" \
    --output "<prediction.nii.gz>" \
    --batch-size 16 --threshold 0.5 \
    --gui  # Launch the viewer after inference
```

## Verification

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build --wheel
```
