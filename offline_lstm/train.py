"""
Train LSTM on data/windows.npz (train split), validate on val split, save checkpoints/lstm_best.pt
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    CHECKPOINT_BEST,
    CHECKPOINTS,
    FEAT_DIM,
    LR,
    PLOTS,
    SPLITS_JSON,
    TRAIN_EPOCHS,
    WINDOW,
    WINDOWS_NPZ,
)
from model import LSTMPredictor


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total += float(nn.functional.mse_loss(pred, yb).item()) * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default=str(WINDOWS_NPZ))
    args = ap.parse_args()

    device = torch.device(args.device)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    z = np.load(args.data)
    Xtr, ytr = z["train_X"], z["train_y"]
    Xva, yva = z["val_X"], z["val_y"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=args.batch,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        batch_size=args.batch,
        shuffle=False,
    )

    model = LSTMPredictor(FEAT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist_train, hist_val = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        nt = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * xb.size(0)
            nt += xb.size(0)
        train_loss /= max(nt, 1)
        val_loss = evaluate(model, val_loader, device)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        print(f"epoch {epoch:3d}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    meta = {}
    if SPLITS_JSON.exists():
        meta = json.loads(SPLITS_JSON.read_text())
    payload = {
        "state_dict": best_state,
        "window": meta.get("window", WINDOW),
        "feat_dim": FEAT_DIM,
        "best_val_mse": best_val,
        "epochs": args.epochs,
    }
    torch.save(payload, CHECKPOINT_BEST)
    print(f"Saved {CHECKPOINT_BEST} (best val MSE {best_val:.6f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_train, label="train MSE")
    ax.plot(hist_val, label="val MSE")
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_title("LSTM training curves")
    fig.savefig(PLOTS / "train_val_loss.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {PLOTS / 'train_val_loss.png'}")


if __name__ == "__main__":
    main()
