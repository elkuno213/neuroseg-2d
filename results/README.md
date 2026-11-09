# Published results

These tracked metrics come from the earlier 1.07 mm resampling pipeline. They remain a historical
baseline until a fixed-crop run replaces this snapshot.

## Run configuration

| Setting | Value |
| --- | --- |
| Dataset | 335 BraTS patients; 251 training, 42 validation, 42 test |
| Preprocessed shape | 240 × 240 × 144 |
| Voxel spacing | 1.07 × 1.07 × 1.07 mm |
| Model | Binary 2D U-Net; 31,036,481 parameters |
| Optimizer | AdamW; learning rate 0.0001; weight decay 0.00001 |
| Loss | Equally weighted Dice + binary cross-entropy |
| Batch size / seed | 16 / 42 |
| Epochs | 10 cumulative; this command added 5 after epoch 5 |
| Selected checkpoint | `best.pt`, epoch 10, lowest validation loss |
| Validation loss | 0.329956 |
| Prediction threshold | 0.50 |

![Training and validation metrics](metrics.png)

## Validation and test metrics

Validation values come from the selected best epoch. Test values were measured after loading its
weights.

Each row keeps its own averaging scope; slice and patient scores are not interchangeable. Saved JSON
keeps total slice and patient counts but omits per-metric valid-value counters.

| Split | Scope | Dice | IoU | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Slice | 0.558332 | 0.486072 | 0.748356 | 0.618215 |
| Validation | Patient | 0.746433 | 0.632849 | 0.830041 | 0.741846 |
| Test | Slice | 0.530661 | 0.464424 | 0.767902 | 0.570643 |
| Test | Patient | 0.730370 | 0.612990 | 0.853750 | 0.696795 |

Validation requirements use strict comparisons on unrounded values:

| Scope | Recall > 0.95 | Precision > 0.70 | Both requirements |
| --- | --- | --- | --- |
| Slice | FAIL | FAIL | FAIL |
| Patient | FAIL | PASS | FAIL |
