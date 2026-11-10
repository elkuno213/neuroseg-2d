# Published results

These metrics come from the fixed axial crop pipeline after 30 cumulative training epochs.

## Run configuration

| Setting | Value |
| --- | --- |
| Dataset | 335 BraTS patients; 251 training, 42 validation, 42 test |
| Preprocessed shape | 240 × 240 × 144 |
| Voxel spacing | 1 × 1 × 1 mm |
| Model | Binary 2D U-Net; 31,036,481 parameters |
| Optimizer | AdamW; learning rate 0.0001; weight decay 0.00001 |
| Loss | Equally weighted Dice + binary cross-entropy |
| Batch size / seed | 16 / 42 |
| Epochs | 30 cumulative; final command added 10 after epoch 20 |
| Selected checkpoint | `best.pt`, epoch 28, lowest validation loss |
| Validation loss | 0.095443 |
| Prediction threshold | 0.50 |

![Training and validation metrics](metrics.png)

## Validation and test metrics

Validation values come from the selected best epoch. Test values were measured after loading its
weights.

Each row keeps its own averaging scope; slice and patient scores are not interchangeable. Saved JSON
keeps total slice and patient counts but omits per-metric valid-value counters.

| Split | Scope | Dice | IoU | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Slice | 0.694182 | 0.613940 | 0.742256 | 0.818998 |
| Validation | Patient | 0.835105 | 0.729996 | 0.850668 | 0.850336 |
| Test | Slice | 0.685368 | 0.610085 | 0.760072 | 0.775840 |
| Test | Patient | 0.840792 | 0.739439 | 0.873869 | 0.832796 |

Validation requirements use strict comparisons on unrounded values:

| Scope | Recall > 0.95 | Precision > 0.70 |
| --- | --- | --- |
| Slice | FAIL | PASS |
| Patient | FAIL | PASS |
