# CPU Test-Split Inference

## Configuration

- Checkpoint: `results/best.pt` from epoch 64
- Device: local CPU
- Threshold: `0.5`
- Split: 34 saved test patients
- Runtime: 16 minutes 50 seconds

## Source-grid correction

Raw BraTS volumes use LPS axes, while training converts them to canonical RAS axes. Previous
inference restored cropped depth but saved canonical voxel order with original LPS affine, producing
an X/Y-flipped mask.

Inference now restores depth on canonical grid, transforms mask axes back to source orientation, then
saves mask with original affine and header.

## Aggregate patient-volume metrics

| Metric | Value |
|---|---:|
| Mean Dice | 0.8569 |
| Mean recall | 0.9366 |
| Mean precision | 0.8021 |
| Patients with recall > 0.95 | 19 / 34 |

## Patients with recall above 0.95

| Patient | Grade | Recall | Dice | Precision |
|---|---|---:|---:|---:|
| BraTS19_CBICA_ASN_1 | HGG | 0.9510 | 0.9165 | 0.8843 |
| BraTS19_TCIA03_265_1 | HGG | 0.9507 | 0.9374 | 0.9245 |
| BraTS19_TCIA13_653_1 | LGG | 0.9999 | 0.8389 | 0.7226 |
| BraTS19_CBICA_AXM_1 | HGG | 0.9522 | 0.8914 | 0.8379 |
| BraTS19_TCIA04_111_1 | HGG | 0.9843 | 0.7636 | 0.6237 |
| BraTS19_2013_27_1 | HGG | 0.9623 | 0.8116 | 0.7018 |
| BraTS19_TCIA08_319_1 | HGG | 0.9736 | 0.9229 | 0.8773 |
| BraTS19_TCIA03_419_1 | HGG | 0.9910 | 0.9353 | 0.8855 |
| BraTS19_TCIA02_607_1 | HGG | 0.9870 | 0.8187 | 0.6995 |
| BraTS19_CBICA_AYU_1 | HGG | 0.9726 | 0.8387 | 0.7373 |
| BraTS19_TCIA06_247_1 | HGG | 0.9952 | 0.8013 | 0.6706 |
| BraTS19_CBICA_BFP_1 | HGG | 0.9874 | 0.8670 | 0.7728 |
| BraTS19_TMC_12866_1 | HGG | 1.0000 | 0.8000 | 0.6667 |
| BraTS19_TCIA03_138_1 | HGG | 0.9636 | 0.9173 | 0.8754 |
| BraTS19_TCIA01_335_1 | HGG | 0.9844 | 0.9460 | 0.9104 |
| BraTS19_TCIA08_218_1 | HGG | 0.9693 | 0.9409 | 0.9140 |
| BraTS19_TCIA01_221_1 | HGG | 0.9752 | 0.9014 | 0.8379 |
| BraTS19_CBICA_APZ_1 | HGG | 0.9993 | 0.6084 | 0.4373 |
| BraTS19_CBICA_AQP_1 | HGG | 0.9826 | 0.8986 | 0.8279 |

## Verification

- All 34 predictions completed without inference errors.
- Every output preserved original `240 × 240 × 155` shape and affine.
- Every output used binary `uint8` values `{0, 1}`.
- Ruff, formatting, type checks, and tests passed.

Mean recall remains below project target despite 19 individual patients exceeding 0.95.
