"""
LSTM next-step predictor for log-derived feature vectors.

The network sees a sequence of past frames (batch, T, feat_dim) and predicts
the *next* frame; at runtime, anomaly score is MSE(prediction, actual next).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMPredictor(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, hidden, num_layers=num_layers, batch_first=True, dropout=0.0
        )
        self.head = nn.Linear(hidden, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        return self.head(last)


def bootstrap_train(
    model: nn.Module,
    device: torch.device,
    window: int,
    feat_dim: int,
    steps: int = 200,
    lr: float = 1e-3,
) -> None:
    """Synthetic pre-fit before live logs (not training on your real data).

    For ``steps`` iterations, builds random Gaussian sequences of shape
    (batch=32, time=window, feat_dim), scales them by 0.25, and trains with
    Adam to minimize MSE between the model output and the *last* timestep,
    given all but the last timestep as input—same layout as production.

    This only calibrates weights so prediction errors are in a sensible range
    at cold start; **online detection** then uses a rolling distribution of
    MSE on real traffic (see ``main.py`` thresholds).
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x = torch.randn(32, window, feat_dim, device=device) * 0.25
        target = x[:, -1, :]
        pred = model(x[:, :-1, :])
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
