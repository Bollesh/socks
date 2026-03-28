"""Same LSTM head as drain3.models.LSTMPredictor (duplicate to keep folder self-contained)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, hidden, num_layers=num_layers, batch_first=True, dropout=0.0
        )
        self.head = nn.Linear(hidden, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])
