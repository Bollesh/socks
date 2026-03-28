"""Paths and defaults for the offline Loki → LSTM pipeline."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
CHECKPOINTS = ROOT / "checkpoints"

for d in (DATA, PLOTS, CHECKPOINTS):
    d.mkdir(parents=True, exist_ok=True)

# Match drain3 defaults
WINDOW = int(os.environ.get("OFFLINE_WINDOW", "24"))
FEAT_DIM = int(os.environ.get("OFFLINE_FEAT_DIM", "8"))

LOKI_URL = os.environ.get("LOKI_URL", "http://localhost:3100").rstrip("/")
DEFAULT_QUERY = os.environ.get("LOKI_QUERY", '{service_name="locust"}')

TRAIN_FRAC = float(os.environ.get("OFFLINE_TRAIN_FRAC", "0.70"))
VAL_FRAC = float(os.environ.get("OFFLINE_VAL_FRAC", "0.15"))
# test = remainder

RAW_CSV = DATA / "raw_logs.csv"
FEATURES_CSV = DATA / "features.csv"
SPLITS_JSON = DATA / "splits.json"
WINDOWS_NPZ = DATA / "windows.npz"

CHECKPOINT_BEST = CHECKPOINTS / "lstm_best.pt"

TRAIN_EPOCHS = int(os.environ.get("OFFLINE_EPOCHS", "40"))
BATCH_SIZE = int(os.environ.get("OFFLINE_BATCH", "64"))
LR = float(os.environ.get("OFFLINE_LR", "1e-3"))
