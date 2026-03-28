# drain3/metric_lstm.py
"""
Prometheus scrape loop + LSTM t+1 predictor for metric anomaly detection.
Runs as an asyncio task alongside the log pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Callable, Any

import httpx
import numpy as np
import torch

from metric_features import METRIC_QUERIES, METRIC_FEAT_DIM, normalise
from models import LSTMPredictor, bootstrap_train

log = logging.getLogger("metric_lstm")

PROM_URL              = os.environ.get("PROM_URL", "http://otel:9090").rstrip("/")
SCRAPE_INTERVAL       = int(os.environ.get("PROM_SCRAPE_INTERVAL", "15"))
METRIC_WINDOW         = int(os.environ.get("DRAIN3_METRIC_WINDOW", "24"))
METRIC_WARMUP         = int(os.environ.get("DRAIN3_METRIC_WARMUP", "32"))
METRIC_ERROR_RING     = int(os.environ.get("DRAIN3_METRIC_ERROR_RING", "200"))
METRIC_Z              = float(os.environ.get("DRAIN3_METRIC_Z", "4.0"))
METRIC_CHECKPOINT     = os.environ.get("DRAIN3_METRIC_CHECKPOINT", "").strip()


async def _scrape(client: httpx.AsyncClient) -> tuple[np.ndarray, float]:
    """Fetch all metric queries from Prometheus, return (feature_vec, timestamp_ns)."""
    raw = []
    for q in METRIC_QUERIES:
        try:
            r = await client.get(
                f"{PROM_URL}/api/v1/query",
                params={"query": q},
                timeout=8.0,
            )
            result = r.json().get("data", {}).get("result", [])
            val = float(result[0]["value"][1]) if result else 0.0
        except Exception:
            val = 0.0
        raw.append(val)
    return normalise(raw), time.time_ns()


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.quantile(arr, q)) if arr.size else 0.0


async def run_metric_lstm(
    on_anomaly: Callable,   # async callable: (source, entry, mse, reason, metric_snapshot, anomaly_time_ns)
    http_client: httpx.AsyncClient,
) -> None:
    """
    Main loop. Runs forever, calls on_anomaly when LSTM fires.
    Pass the shared httpx client from main.py.
    """
    device = torch.device("cpu")
    model = LSTMPredictor(METRIC_FEAT_DIM).to(device)

    if METRIC_CHECKPOINT:
        try:
            ckpt = torch.load(METRIC_CHECKPOINT, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(METRIC_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        log.info("Loaded metric LSTM from %s", METRIC_CHECKPOINT)
    else:
        log.info("Bootstrap training metric LSTM (200 synthetic steps)...")
        bootstrap_train(model, device, METRIC_WINDOW, METRIC_FEAT_DIM)

    model.eval()
    log.info("Metric LSTM ready; scrape_interval=%ss window=%s", SCRAPE_INTERVAL, METRIC_WINDOW)

    feat_buf: deque[np.ndarray] = deque(maxlen=METRIC_WINDOW + 5)
    err_ring: deque[float]      = deque(maxlen=METRIC_ERROR_RING)

    while True:
        await asyncio.sleep(SCRAPE_INTERVAL)
        try:
            vec, ts_ns = await _scrape(http_client)
        except Exception as exc:
            log.warning("Prometheus scrape error: %s", exc)
            continue

        feat_buf.append(vec)
        if len(feat_buf) < METRIC_WINDOW:
            continue

        win = np.stack(list(feat_buf)[-METRIC_WINDOW:], axis=0)
        x   = torch.from_numpy(win[:-1]).unsqueeze(0).to(device)
        y_t = torch.from_numpy(win[-1]).unsqueeze(0).to(device)

        with torch.inference_mode():
            pred = model(x)
            mse  = float(torch.mean((pred - y_t) ** 2).cpu())

        err_ring.append(mse)
        if len(err_ring) < METRIC_WARMUP:
            continue

        arr = np.array(err_ring, dtype=np.float64)
        thr = max(
            _percentile(arr, 0.985),
            float(np.mean(arr) + METRIC_Z * max(np.std(arr), 1e-6)),
        )

        if mse > thr:
            metric_snapshot = {
                METRIC_QUERIES[i]: float(vec[i]) for i in range(METRIC_FEAT_DIM)
            }
            # on_anomaly is handle_anomaly in main.py
            await on_anomaly(
                source="metric",
                entry={},                  # no log entry for metric anomalies
                mse=mse,
                reason=f"metric_lstm:mse={mse:.6f}>threshold={thr:.6f}",
                metric_snapshot=metric_snapshot,
                anomaly_time_ns=ts_ns,
            )
