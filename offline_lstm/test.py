"""
Final evaluation on held-out test split from windows.npz.
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    X, y = z["test_X"], z["test_y"]
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
            preds.append(model(xb).cpu())
            trues.append(yb)
    pred = torch.cat(preds, dim=0).numpy()
    true = torch.cat(trues, dim=0).numpy()
    err = pred - true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    per_dim_mse = np.mean(err**2, axis=0)

    report = {
        "split": "test",
        "n_samples": int(pred.shape[0]),
        "mse": mse,
        "mae": mae,
        "per_dim_mse": {FEAT_NAMES[i]: float(per_dim_mse[i]) for i in range(len(FEAT_NAMES))},
    }
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / "test_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(FEAT_NAMES)), per_dim_mse, color="teal", edgecolor="white")
    ax.set_xticks(np.arange(len(FEAT_NAMES)))
    ax.set_xticklabels(FEAT_NAMES, rotation=25, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("Test set MSE per feature dimension")
    fig.tight_layout()
    fig.savefig(PLOTS / "test_per_dim_mse.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
