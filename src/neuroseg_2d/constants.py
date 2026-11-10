"""Centralize NeuroSeg 2D settings.

Usage:
    from neuroseg_2d.constants import SHAPE
"""

__author__ = "Vu Hoi HUYNH"

from pathlib import Path

# ─── Preprocessing ────────────────────────────────────────────────────────────────────────────── #
SHAPE = (240, 240, 144)
Z_START = 5
Z_END = 149

# ─── Dataset and model ─────────────────────────────────────────────────────────────────────────── #
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
FEATURES: tuple[int, ...] = (64, 128, 256, 512)

# ─── Data loading, training, and inference ─────────────────────────────────────────────────────── #
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
POS_WEIGHT = 2.0
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
LR_FACTOR = 0.5
LR_PATIENCE = 1
MIN_LR = 1e-6
BATCH_SIZE = 16
EPOCHS = 10
NUM_WORKERS = 0
THRESHOLD = 0.5
CHECKPOINTS_DIR = Path("checkpoints")

# ─── Reporting ─────────────────────────────────────────────────────────────────────────────────── #
METRIC_FIELDS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "learning_rate",
    "slice",
    "patient",
)
REPORT_FIELDS = (
    "batch_size",
    "best_epoch",
    "best_validation_loss",
    "data_dir",
    "device",
    "epochs",
    "learning_rate",
    "loss",
    "scheduler",
    "patients",
    "resume_checkpoint",
    "num_parameters",
    "seed",
    "threshold",
    "test_metrics",
)
