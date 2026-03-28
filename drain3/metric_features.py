# drain3/metric_features.py
"""
Shared Prometheus query definitions and normalisation for the metric LSTM.
Must stay identical to offline_lstm/metric_features.py.
Any change here requires retraining the metric checkpoint.
"""
from __future__ import annotations
import math
import numpy as np

# Ordered list of PromQL queries — order defines the feature vector index.
METRIC_QUERIES: list[str] = [
    "rate(http_requests_total[1m])",           # 0: overall request rate
    "rate(http_errors_total[1m])",             # 1: error rate
    "histogram_quantile(0.99, rate(http_request_duration_ms_milliseconds_bucket[1m]))",  # 2: p99 latency ms
    "locust_active_users",                     # 3: active user count
    "rate(orders_placed_total[1m])",           # 4: order rate
    "rate(cart_operations_total[1m])",         # 5: cart op rate
]

METRIC_FEAT_DIM = len(METRIC_QUERIES)  # 6

# Per-feature normalisation caps — tune after seeing real traffic ranges.
# Values are clipped to [0, cap] then divided by cap → [0, 1].
# For latency we log-normalise instead (see normalise()).
_CAPS = [50.0, 10.0, 1.0, 200.0, 5.0, 20.0]  # indices match METRIC_QUERIES

def normalise(raw: list[float]) -> np.ndarray:
    """Convert a raw Prometheus result vector to a normalised float32 array."""
    out = []
    for i, v in enumerate(raw):
        if i == 2:  # p99 latency — log normalise
            out.append(math.log1p(max(v, 0.0)) / 10.0)
        else:
            out.append(min(max(v, 0.0), _CAPS[i]) / _CAPS[i])
    return np.array(out, dtype=np.float32)
