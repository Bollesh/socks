"""
Evaluate saved checkpoint on the validation split; plots for errors and one feature dim.
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CHECKPOINT_BEST, FEAT_DIM, PLOTS, WINDOWS_NPZ
from features import FEAT_NAMES
from model import LSTMPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT_BEST))
    ap.add_argument("--data", default=str(WINDOWS_NPZ))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    z = np.load(args.data)
    X, y = z["val_X"], z["val_y"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=args.batch,
        shuffle=False,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LSTMPredictor(ckpt.get("feat_dim", FEAT_DIM)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pb = model(xb).cpu()
            preds.append(pb)
            trues.append(yb)
    pred = torch.cat(preds, dim=0).numpy()
    true = torch.cat(trues, dim=0).numpy()
    err = pred - true
    mse = np.mean(err**2)
    mae = np.mean(np.abs(err))
    per_dim_mse = np.mean(err**2, axis=0)

    report = {
        "split": "val",
        "n_samples": int(pred.shape[0]),
        "mse": float(mse),
        "mae": float(mae),
        "per_dim_mse": {FEAT_NAMES[i]: float(per_dim_mse[i]) for i in range(len(FEAT_NAMES))},
    }
    out_json = PLOTS / "validate_report.json"
    PLOTS.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")

    dim = 0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true[:, dim], pred[:, dim], s=8, alpha=0.35, c="navy")
    lim = max(true[:, dim].max(), pred[:, dim].max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel(f"true {FEAT_NAMES[dim]}")
    ax.set_ylabel(f"pred {FEAT_NAMES[dim]}")
    ax.set_title("Validation: pred vs true (dim 0)")
    fig.savefig(PLOTS / "validate_scatter_dim0.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.sqrt(np.mean(err**2, axis=1)), bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("per-sample RMSE")
    ax.set_title("Validation error distribution")
    fig.savefig(PLOTS / "validate_error_hist.png", dpi=150)
    plt.close(fig)
    print(f"Plots in {PLOTS}")


if __name__ == "__main__":
    main()
