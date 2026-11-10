# Published results

This snapshot contains the best validation checkpoint after 12 completed epochs.
Training continues; these are interim results.

## Run configuration

| Setting | Value |
| --- | --- |
| Dataset | 335 BraTS patients; 268 training, 33 validation, 34 test |
| Preprocessed shape | 240 × 240 × 144 |
| Voxel spacing | 1 × 1 × 1 mm |
| Model | Binary 2D U-Net; 31,036,481 parameters |
| Optimizer | AdamW; initial learning rate 0.00003; weight decay 0.00001 |
| Scheduler | ReduceLROnPlateau; factor 0.5, patience 1, minimum rate 0.000001 |
| Loss | Equal Tversky + BCE; FP weight 0.3, FN weight 0.7, positive BCE weight 2 |
| Batch size / seed | 16 / 42 |
| Epochs | 12 cumulative; current continuation completed 2 after epoch 10 |
| Selected checkpoint | `best.pt`, epoch 12, lowest validation loss |
| Resume checkpoint | `latest.pt`, completed epoch 12 |
| Learning rate at selected epoch | 0.000015 |
| Validation loss | 0.088641 |
| Prediction threshold | 0.50 |

![Training and validation metrics](metrics.png)

## Validation and test metrics

Validation values come from epoch 12 at threshold 0.5. Test evaluation for this checkpoint is pending;
previous checkpoint test scores and comparison images are not included.

Each row keeps its own averaging scope; slice and patient scores are not interchangeable. Saved JSON
keeps total slice and patient counts but omits per-metric valid-value counters.

| Split | Scope | Dice | IoU | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Slice | 0.641761 | 0.570804 | 0.803531 | 0.682759 |
| Validation | Patient | 0.836463 | 0.733892 | 0.892981 | 0.804487 |
| Test | Slice | Pending | Pending | Pending | Pending |
| Test | Patient | Pending | Pending | Pending | Pending |

Validation requirements use strict comparisons on unrounded values:

| Scope | Recall > 0.95 | Precision > 0.70 |
| --- | --- | --- |
| Slice | FAIL | FAIL |
| Patient | FAIL | PASS |
