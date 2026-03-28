"""
Train LSTM on metric_windows.npz. Same loop as train.py but for metric data.
Saves checkpoints/lstm_metric_best.pt
"""
from __future__ import annotations
import argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import CHECKPOINTS, PLOTS, TRAIN_EPOCHS, BATCH_SIZE, LR
from metric_features import METRIC_FEAT_DIM
from model import LSTMPredictor

METRIC_WINDOWS_NPZ  = "data/metric_windows.npz"
METRIC_CHECKPOINT   = CHECKPOINTS / "lstm_metric_best.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--batch",  type=int, default=BATCH_SIZE)
    ap.add_argument("--lr",     type=float, default=LR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data",   default=METRIC_WINDOWS_NPZ)
    args = ap.parse_args()

    device = torch.device(args.device)
    z = np.load(args.data)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(z["train_X"]), torch.from_numpy(z["train_y"])),
        batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(z["val_X"]), torch.from_numpy(z["val_y"])),
        batch_size=args.batch, shuffle=False)

    model = LSTMPredictor(METRIC_FEAT_DIM).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, nt = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward(); opt.step()
            train_loss += loss.item() * xb.size(0); nt += xb.size(0)
        train_loss /= max(nt, 1)

        model.eval(); val_loss, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += nn.functional.mse_loss(model(xb), yb).item() * xb.size(0)
                nv += xb.size(0)
        val_loss /= max(nv, 1)
        print(f"epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feat_dim": METRIC_FEAT_DIM,
                "best_val_mse": best_val}, METRIC_CHECKPOINT)
    print(f"Saved {METRIC_CHECKPOINT} (best val MSE {best_val:.6f})")

if __name__ == "__main__":
    main()
