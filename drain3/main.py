# drain3/main.py
"""
drain3: dual-pipeline anomaly detection.
Pipeline 1: log-stream SSE → Drain3 + Isolation Forest → handle_anomaly
Pipeline 2: Prometheus scrape → metric LSTM → handle_anomaly
Both pipelines call the shared handle_anomaly() which fetches context and calls the SLM.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import deque
from typing import Any, Optional

print(
    "drain3: starting (importing torch + sklearn may take 10-60s)...",
    file=sys.stderr, flush=True,
)

import httpx
import numpy as np
import torch

from models import LSTMPredictor, bootstrap_train
from trace_summary import summarize_trace_json
from log_anomaly import LogAnomalyDetector
from metric_lstm import run_metric_lstm
from trace_lookup import find_trace_id

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("drain3")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Environment ──────────────────────────────────────────────────────────────
LOG_STREAM_URL = os.environ["LOG_STREAM_URL"]
TEMPO_URL      = os.environ.get("TEMPO_URL", "http://otel:3200").rstrip("/")
LOKI_URL       = os.environ.get("LOKI_URL",  "http://otel:3100").rstrip("/")
PROM_URL       = os.environ.get("PROM_URL",  "http://otel:9090").rstrip("/")
SLM_URL        = os.environ.get("SLM_URL", "").strip()
SLM_MODEL      = os.environ.get("SLM_MODEL", "llama3.2")
DASHBOARD_URL  = os.environ.get("DASHBOARD_URL", "").strip()

# Log LSTM config (unchanged from original)
WINDOW         = int(os.environ.get("DRAIN3_WINDOW", "24"))
FEAT_DIM       = int(os.environ.get("DRAIN3_FEAT_DIM", "8"))
WARMUP_WINDOWS = int(os.environ.get("DRAIN3_WARMUP", "64"))
ERROR_RING     = int(os.environ.get("DRAIN3_ERROR_RING", "400"))
Z_MULT         = float(os.environ.get("DRAIN3_Z", "4.5"))
FORCE_RULES    = os.environ.get("DRAIN3_FORCE_RULES", "1") == "1"
CHECKPOINT     = os.environ.get("DRAIN3_CHECKPOINT", "").strip()


# ── Feature extraction (log pipeline, unchanged) ─────────────────────────────
def _featurize(entry: dict[str, Any]) -> np.ndarray:
    labels   = entry.get("labels") if isinstance(entry.get("labels"), dict) else {}
    level    = str(entry.get("level") or labels.get("level") or "info").lower()
    is_error = 1.0 if level == "error" else 0.0
    dur_raw  = labels.get("duration_ms")
    try:    dur = float(dur_raw)
    except: dur = 0.0
    dur = math.log1p(max(dur, 0.0)) / 10.0
    stat = labels.get("status") or entry.get("status")
    try:    sc = int(stat)
    except: sc = 0
    status_n = min(max(sc / 600.0, 0.0), 1.0)
    is_5xx   = 1.0 if sc >= 500 else 0.0
    url      = str(labels.get("url") or "")
    url_hash = (int(hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8], 16) % 10_001) / 10_000.0
    msg      = str(entry.get("message") or entry.get("line") or "")
    msg_len  = min(len(msg) / 400.0, 1.0)
    has_trace = 1.0 if entry.get("trace_id") else 0.0
    return np.array([dur, is_error, status_n, is_5xx, url_hash, msg_len, has_trace, 0.0], dtype=np.float32)


def _rule_anomaly(entry: dict[str, Any]) -> tuple[bool, str]:
    labels = entry.get("labels") if isinstance(entry.get("labels"), dict) else {}
    level  = str(entry.get("level") or labels.get("level") or "").lower()
    if level == "error":
        return True, "rule:log_level_error"
    stat = labels.get("status") or entry.get("status")
    try:    sc = int(stat)
    except: sc = 0
    if sc >= 500:
        return True, "rule:http_5xx"
    return False, ""


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.quantile(arr, q)) if arr.size else 0.0


# ── Tempo fetch (unchanged) ───────────────────────────────────────────────────
async def _fetch_trace(client: httpx.AsyncClient, trace_id: str) -> Optional[dict]:
    if not trace_id:
        return None
    tid = trace_id.strip().lower().replace("0x", "")
    try:
        r = await client.get(f"{TEMPO_URL}/api/traces/{tid}", timeout=10.0)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("Tempo fetch failed for %s: %s", tid[:16], exc)
        return None


# ── Prometheus metric snapshot ────────────────────────────────────────────────
async def _fetch_metric_snapshot(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch current values for all metric queries. Used to enrich log anomaly context."""
    from metric_features import METRIC_QUERIES
    snapshot = {}
    for q in METRIC_QUERIES:
        try:
            r = await client.get(f"{PROM_URL}/api/v1/query", params={"query": q}, timeout=5.0)
            result = r.json().get("data", {}).get("result", [])
            snapshot[q] = float(result[0]["value"][1]) if result else 0.0
        except Exception:
            snapshot[q] = 0.0
    return snapshot


# ── SLM call ─────────────────────────────────────────────────────────────────
async def _slm_call(client: httpx.AsyncClient, prompt: str) -> str:
    if not SLM_URL:
        return "(SLM_URL not set)"
    r = await client.post(
        SLM_URL,
        json={"model": SLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120.0,
    )
    r.raise_for_status()
    return str(r.json().get("response", r.json()))


# ── Shared anomaly handler ────────────────────────────────────────────────────
async def handle_anomaly(
    client: httpx.AsyncClient,
    source: str,                          # "log" or "metric"
    entry: dict[str, Any],               # log entry (empty dict for metric pipeline)
    mse: float,
    reason: str,
    metric_snapshot: Optional[dict] = None,
    anomaly_time_ns: Optional[int]  = None,
) -> None:
    if anomaly_time_ns is None:
        anomaly_time_ns = time.time_ns()

    # --- Find trace_id ---
    trace_id = str(entry.get("trace_id") or "")
    if not trace_id:
        trace_id = await find_trace_id(client, TEMPO_URL, LOKI_URL, anomaly_time_ns) or ""

    # --- Fetch trace graph ---
    trace_payload = await _fetch_trace(client, trace_id)
    trace_blurb   = summarize_trace_json(trace_payload)

    # --- Fetch metric snapshot if not already provided (log pipeline) ---
    if metric_snapshot is None:
        metric_snapshot = await _fetch_metric_snapshot(client)

    # --- Format metric block ---
    metric_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in metric_snapshot.items())

    # --- Format log entry block ---
    log_block = json.dumps(entry, indent=2, default=str) if entry else "(metric anomaly — no log entry)"

    # --- Build prompt ---
    prompt = f"""You are an SRE assistant. A production anomaly has been detected.

Anomaly source: {source}
Reason: {reason}
Model MSE: {mse:.6f}

Current Prometheus metrics:
{metric_lines}

Log entry:
{log_block}

Tempo trace digest:
{trace_blurb}

Respond with exactly three sections:
1. ROOT CAUSE: What is most likely wrong and why.
2. IMMEDIATE FIX: Specific commands, config changes, or restarts to resolve it.
3. VERIFY: Commands or checks to confirm the fix worked.
"""

    reply = ""
    try:
        reply = await _slm_call(client, prompt)
        log.warning(
            "SLM fix [source=%s trace=%s]\n%s",
            source, trace_id[:16] if trace_id else "-", reply,
        )
    except Exception as exc:
        log.error("SLM call failed: %s", exc)
        reply = f"(SLM error: {exc})"

    # --- Push to dashboard ---
    if DASHBOARD_URL:
        dashboard_event = {
            "source": source,
            "reason": reason,
            "mse": mse,
            "trace_id": trace_id or None,
            "log_entry": json.dumps(entry, default=str)[:500] if entry else None,
            "slm_response": reply,
            "timestamp": anomaly_time_ns / 1e9,
            "metric_snapshot": {k: round(v, 4) for k, v in (metric_snapshot or {}).items()},
        }
        try:
            await client.post(f"{DASHBOARD_URL}/api/anomalies", json=dashboard_event, timeout=5.0)
        except Exception as exc:
            log.debug("Dashboard push failed: %s", exc)


# ── Log pipeline ──────────────────────────────────────────────────────────────
async def run_log_pipeline(
    client: httpx.AsyncClient,
    on_anomaly,
) -> None:
    device    = torch.device("cpu")
    log_model = LSTMPredictor(FEAT_DIM).to(device)

    if CHECKPOINT:
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        log_model.load_state_dict(ckpt["state_dict"])
        log.info("Loaded log LSTM from %s", CHECKPOINT)
    else:
        log.info("Bootstrap training log LSTM...")
        bootstrap_train(log_model, device, WINDOW, FEAT_DIM)

    log_model.eval()

    detector  = LogAnomalyDetector()
    feat_buf  = deque(maxlen=WINDOW + 5)
    err_ring  = deque(maxlen=ERROR_RING)
    parsed    = 0

    while True:
        try:
            async with client.stream("GET", LOG_STREAM_URL) as resp:
                resp.raise_for_status()
                log.info("Log pipeline connected to %s", LOG_STREAM_URL)
                buf = ""
                async for chunk in resp.aiter_text():
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            entry = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue

                        vec = _featurize(entry)
                        feat_buf.append(vec)
                        parsed += 1
                        if parsed % 500 == 0:
                            log.info("Log pipeline: %s events processed", parsed)

                        # Rule check
                        ra, rreason = _rule_anomaly(entry)
                        if FORCE_RULES and ra:
                            await on_anomaly(
                                source="log", entry=entry,
                                mse=0.0, reason=rreason,
                                anomaly_time_ns=time.time_ns(),
                            )
                            continue

                        # Drain3 + Isolation Forest check
                        is_anom, dreason = detector.process(entry, vec)
                        if is_anom:
                            await on_anomaly(
                                source="log", entry=entry,
                                mse=0.0, reason=dreason,
                                anomaly_time_ns=time.time_ns(),
                            )
                            continue

                        # LSTM check (same as original)
                        if len(feat_buf) < WINDOW:
                            continue
                        win = np.stack(list(feat_buf)[-WINDOW:], axis=0)
                        x   = torch.from_numpy(win[:-1]).unsqueeze(0).to(device)
                        y_t = torch.from_numpy(win[-1]).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            pred = log_model(x)
                            mse  = float(torch.mean((pred - y_t) ** 2).cpu())
                        err_ring.append(mse)
                        if len(err_ring) < WARMUP_WINDOWS:
                            continue
                        arr = np.array(err_ring, dtype=np.float64)
                        thr = max(
                            _percentile(arr, 0.985),
                            float(np.mean(arr) + Z_MULT * max(np.std(arr), 1e-6)),
                        )
                        if mse > thr:
                            await on_anomaly(
                                source="log", entry=entry,
                                mse=mse, reason=f"log_lstm:mse>{thr:.6f}",
                                anomaly_time_ns=time.time_ns(),
                            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Log pipeline error, reconnecting in 2s: %s", exc)
            await asyncio.sleep(2.0)


# ── Entry point ───────────────────────────────────────────────────────────────
async def run() -> None:
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60.0)) as client:

        # Wrap handle_anomaly to inject the shared client
        async def on_anomaly(**kwargs):
            await handle_anomaly(client=client, **kwargs)

        await asyncio.gather(
            run_log_pipeline(client, on_anomaly),
            run_metric_lstm(on_anomaly, client),
        )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("drain3: stopped.", file=sys.stderr, flush=True)
