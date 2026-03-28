"""
From data/raw_logs.csv → features.csv + numpy windows + chronological train/val/test split.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import (
    FEATURES_CSV,
    FEAT_DIM,
    RAW_CSV,
    SPLITS_JSON,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW,
    WINDOWS_NPZ,
)
from features import FEAT_NAMES, featurize, row_to_entry


def build_windows(feats: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """feats: (n, F). Returns X (N, window-1, F), y (N, F)."""
    n = feats.shape[0]
    if n < window:
        raise ValueError(f"Need at least window={window} rows, got {n}")
    xs, ys = [], []
    for k in range(0, n - window + 1):
        block = feats[k : k + window]
        xs.append(block[:-1])
        ys.append(block[-1])
    return np.stack(xs, dtype=np.float32), np.stack(ys, dtype=np.float32)


def split_indices(n_windows: int, train_frac: float, val_frac: float) -> tuple[slice, slice, slice]:
    n_train = int(n_windows * train_frac)
    n_val = int(n_windows * val_frac)
    n_test = n_windows - n_train - n_val
    if n_test <= 0:
        raise ValueError("Adjust TRAIN_FRAC + VAL_FRAC so test set is non-empty")
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(RAW_CSV))
    ap.add_argument("--features-out", default=str(FEATURES_CSV))
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    args = ap.parse_args()
    win = args.window

    df = pd.read_csv(args.input)
    if "timestamp_ns" not in df.columns:
        raise ValueError("CSV needs timestamp_ns column (run fetch_loki.py first)")
    df = df.sort_values("timestamp_ns").reset_index(drop=True)

    feat_rows = []
    for _, r in df.iterrows():
        e = row_to_entry(r.to_dict())
        v = featurize(e)
        feat_rows.append(v)
    F = np.stack(feat_rows, axis=0)
    feat_df = pd.DataFrame(F, columns=list(FEAT_NAMES))
    feat_df.insert(0, "timestamp_ns", df["timestamp_ns"].values)
    feat_df["level"] = df["level"].values if "level" in df.columns else ""
    feat_df["status"] = df["status"].values if "status" in df.columns else ""
    feat_df.to_csv(args.features_out, index=False)
    print(f"Wrote features {feat_df.shape} -> {args.features_out}")

    X_all, y_all = build_windows(F, win)
    n_w = X_all.shape[0]
    sl_tr, sl_va, sl_te = split_indices(n_w, args.train_frac, args.val_frac)

    meta = {
        "n_raw_rows": len(df),
        "n_windows": n_w,
        "window": win,
        "feat_dim": FEAT_DIM,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "train_slice": [sl_tr.start, sl_tr.stop],
        "val_slice": [sl_va.start, sl_va.stop],
        "test_slice": [sl_te.start, n_w],
    }
    SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote splits -> {SPLITS_JSON}")

    np.savez(
        WINDOWS_NPZ,
        train_X=X_all[sl_tr],
        train_y=y_all[sl_tr],
        val_X=X_all[sl_va],
        val_y=y_all[sl_va],
        test_X=X_all[sl_te],
        test_y=y_all[sl_te],
    )
    print(
        f"Wrote {WINDOWS_NPZ}: train {X_all[sl_tr].shape[0]}, "
        f"val {X_all[sl_va].shape[0]}, test {X_all[sl_te].shape[0]}"
    )


if __name__ == "__main__":
    main()
