"""
Normalise metric_raw.csv and build windowed numpy arrays → metric_windows.npz
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from config import DATA, WINDOWS_NPZ, SPLITS_JSON, WINDOW, TRAIN_FRAC, VAL_FRAC
from metric_features import METRIC_QUERIES, normalise

METRIC_WINDOWS_NPZ = DATA / "metric_windows.npz"
METRIC_SPLITS_JSON = DATA / "metric_splits.json"

def build_windows(feats, window):
    xs, ys = [], []
    for k in range(len(feats) - window + 1):
        block = feats[k:k+window]
        xs.append(block[:-1])
        ys.append(block[-1])
    return np.stack(xs, dtype=np.float32), np.stack(ys, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(DATA / "metric_raw.csv"))
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()

    df = pd.read_csv(args.input).sort_values("timestamp")
    df[METRIC_QUERIES] = df[METRIC_QUERIES].fillna(0.0)
    feats = np.stack([normalise(list(row[1:])) for row in df[METRIC_QUERIES].itertuples()], axis=0)

    X, y = build_windows(feats, args.window)
    n = len(X)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)

    np.savez(METRIC_WINDOWS_NPZ,
        train_X=X[:n_tr], train_y=y[:n_tr],
        val_X=X[n_tr:n_tr+n_va], val_y=y[n_tr:n_tr+n_va],
        test_X=X[n_tr+n_va:], test_y=y[n_tr+n_va:],
    )
    meta = {"window": args.window, "feat_dim": len(METRIC_QUERIES),
            "n_windows": n, "train": n_tr, "val": n_va, "test": n - n_tr - n_va}
    METRIC_SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRIC_SPLITS_JSON.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {METRIC_WINDOWS_NPZ}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
