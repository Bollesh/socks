# Socks observability — dual-pipeline anomaly detection implementation

## Overview

Two independent detection pipelines feed a single SLM that outputs a fix:

- **Pipeline 1 (log):** log-stream SSE → Drain3 template clustering + Isolation Forest → fetch Tempo trace + Prometheus snapshot → SLM
- **Pipeline 2 (metric):** Prometheus scrape loop → LSTM t+1 predictor → Tempo search + Loki logs → SLM

Both pipelines share a single `handle_anomaly()` coroutine and the same Ollama-compatible SLM call.

---

## Repository layout after changes

```
socks/
├── docker-compose.yaml                  ← add env vars to drain3 service
├── drain3/
│   ├── Dockerfile                       ← add scikit-learn, drain3 to pip install
│   ├── requirements.txt                 ← add drain3, scikit-learn
│   ├── metric_features.py               ← NEW: shared query strings + normalisation
│   ├── log_anomaly.py                   ← NEW: Drain3 templates + Isolation Forest
│   ├── metric_lstm.py                   ← NEW: Prometheus scrape loop + LSTM scoring
│   ├── trace_lookup.py                  ← NEW: Tempo search + Loki fallback
│   ├── models.py                        ← unchanged
│   ├── trace_summary.py                 ← unchanged
│   └── main.py                          ← rewrite: orchestrator + shared handle_anomaly
└── offline_lstm/
    ├── fetch_prometheus.py              ← NEW: pull Prometheus range data
    ├── prepare_metric_data.py           ← NEW: normalise + window metric CSV
    ├── train_metric.py                  ← NEW: train LSTM on metric windows
    ├── metric_features.py               ← NEW: copy of drain3/metric_features.py
    ├── config.py                        ← add metric paths + hyperparams
    └── (existing files unchanged)
```

---

## Step 1 — `drain3/metric_features.py` (new)

This file is the single source of truth for which Prometheus queries are used and how their values are normalised. **Both the offline training pipeline and the live scraper must import from an identical copy of this file.** If they diverge, the checkpoint will be useless.

```python
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
    "histogram_quantile(0.99, rate(http_request_duration_ms_bucket[1m]))",  # 2: p99 latency ms
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
```

Copy this file verbatim to `offline_lstm/metric_features.py`. Do not have one import the other across the package boundary — keep them as identical duplicates so the offline package stays self-contained.

---

## Step 2 — `drain3/log_anomaly.py` (new)

Wraps the real `drain3` library for template clustering. Each unique log template gets its own `IsolationForest` instance trained incrementally on feature vectors from that template's historical events. Fires on two conditions: a brand-new template (structural anomaly) or a high isolation score on a known template (statistical anomaly).

```python
# drain3/log_anomaly.py
"""
Log anomaly detection using real Drain3 template clustering
combined with per-template Isolation Forest scoring.
"""
from __future__ import annotations

import logging
import numpy as np
from collections import defaultdict
from typing import Any

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.ensemble import IsolationForest

log = logging.getLogger("log_anomaly")

# Minimum number of events seen for a template before the IF model is fitted.
MIN_SAMPLES_FOR_MODEL = 30
# sklearn IsolationForest contamination assumption.
CONTAMINATION = 0.05
# Score threshold: IsolationForest returns negative scores; more negative = more anomalous.
# -0.1 is a reasonable starting point — tune after observing real score distributions.
SCORE_THRESHOLD = float("-0.1")


class LogAnomalyDetector:
    def __init__(self):
        cfg = TemplateMinerConfig()
        cfg.load_default()
        # Drain3 config tweaks for HTTP log lines
        cfg.drain_sim_th = 0.4        # lower = more aggressive clustering
        cfg.drain_depth = 4
        cfg.drain_max_children = 100
        self._miner = TemplateMiner(config=cfg)

        # template_id → list of feature vectors (numpy rows)
        self._history: dict[int, list[np.ndarray]] = defaultdict(list)
        # template_id → fitted IsolationForest (or None if not enough samples yet)
        self._models: dict[int, IsolationForest | None] = defaultdict(lambda: None)
        # template_ids seen so far
        self._known: set[int] = set()

    def process(self, entry: dict[str, Any], feature_vec: np.ndarray) -> tuple[bool, str]:
        """
        Process one log entry. Returns (is_anomaly, reason).
        feature_vec must be the same 8-D vector already computed by _featurize().
        """
        message = str(entry.get("message") or entry.get("line") or "")
        result = self._miner.add_log_message(message)
        if result is None:
            return False, ""

        cluster = result["cluster"]
        tid = cluster.cluster_id
        is_new = tid not in self._known
        self._known.add(tid)

        if is_new:
            log.info("New log template #%s: %s", tid, cluster.get_template())
            return True, f"drain3:new_template:{cluster.get_template()[:80]}"

        # Accumulate history for this template
        self._history[tid].append(feature_vec)
        n = len(self._history[tid])

        # Refit IsolationForest periodically
        if n >= MIN_SAMPLES_FOR_MODEL and n % 20 == 0:
            X = np.stack(self._history[tid], axis=0)
            self._models[tid] = IsolationForest(
                contamination=CONTAMINATION, random_state=42, n_estimators=50
            )
            self._models[tid].fit(X)
            log.debug("Refitted IF for template #%s (n=%s)", tid, n)

        model = self._models[tid]
        if model is None:
            return False, ""

        score = float(model.score_samples(feature_vec.reshape(1, -1))[0])
        if score < SCORE_THRESHOLD:
            return True, (
                f"isolation_forest:template#{tid}:"
                f"score={score:.4f}<{SCORE_THRESHOLD}"
            )

        return False, ""
```

---

## Step 3 — `drain3/trace_lookup.py` (new)

Given an anomaly timestamp, finds correlated traces. Tries Tempo search first (error traces in a ±30s window), falls back to Loki log query to extract `trace_id` from log lines in the same window.

```python
# drain3/trace_lookup.py
"""
Find a correlated trace_id when none is present on the triggering event.
Used by the metric LSTM pipeline where metrics carry no trace_id.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

log = logging.getLogger("trace_lookup")


async def find_trace_id(
    client: httpx.AsyncClient,
    tempo_url: str,
    loki_url: str,
    anomaly_time_ns: int,
    window_s: int = 30,
) -> Optional[str]:
    """
    Returns a trace_id string, or None if nothing found.
    anomaly_time_ns: unix nanoseconds of the anomaly event.
    """
    start_s = anomaly_time_ns // 1_000_000_000 - window_s
    end_s   = anomaly_time_ns // 1_000_000_000 + window_s

    # --- Attempt 1: Tempo search API ---
    try:
        r = await client.get(
            f"{tempo_url}/api/search",
            params={
                "start": str(start_s),
                "end":   str(end_s),
                "tags":  "error=true",
                "limit": "5",
            },
            timeout=10.0,
        )
        if r.status_code == 200:
            traces = r.json().get("traces") or []
            if traces:
                tid = traces[0].get("traceID", "")
                if tid:
                    log.info("trace_lookup: found via Tempo search: %s", tid[:16])
                    return tid
    except Exception as exc:
        log.warning("trace_lookup: Tempo search failed: %s", exc)

    # --- Attempt 2: Loki query_range fallback ---
    try:
        start_ns = anomaly_time_ns - window_s * 1_000_000_000
        end_ns   = anomaly_time_ns + window_s * 1_000_000_000
        r = await client.get(
            f"{loki_url}/loki/api/v1/query_range",
            params={
                "query":     '{service_name="locust"}',
                "start":     str(start_ns),
                "end":       str(end_ns),
                "limit":     "50",
                "direction": "backward",
            },
            timeout=10.0,
        )
        if r.status_code == 200:
            for stream in r.json().get("data", {}).get("result", []):
                tid = stream.get("stream", {}).get("trace_id", "")
                if tid:
                    log.info("trace_lookup: found via Loki fallback: %s", tid[:16])
                    return tid
                # also search in log line values
                for _ts, line in stream.get("values", []):
                    if "trace_id" in line:
                        import re
                        m = re.search(r'trace_id["\s:=]+([0-9a-f]{16,32})', line)
                        if m:
                            log.info("trace_lookup: extracted from log line: %s", m.group(1)[:16])
                            return m.group(1)
    except Exception as exc:
        log.warning("trace_lookup: Loki fallback failed: %s", exc)

    log.info("trace_lookup: no trace found in window")
    return None
```

---

## Step 4 — `drain3/metric_lstm.py` (new)

The async Prometheus scrape loop. Runs concurrently with the log pipeline. Polls Prometheus on a fixed interval, builds normalised feature vectors, feeds the LSTM, and fires `handle_anomaly` when MSE exceeds the rolling threshold.

```python
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
```

---

## Step 5 — `drain3/main.py` (rewrite)

The orchestrator. Launches both pipelines as concurrent asyncio tasks and provides the shared `handle_anomaly()`.

```python
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

    try:
        reply = await _slm_call(client, prompt)
        log.warning(
            "SLM fix [source=%s trace=%s]\n%s",
            source, trace_id[:16] if trace_id else "-", reply,
        )
    except Exception as exc:
        log.error("SLM call failed: %s", exc)


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
                                client=client, source="log", entry=entry,
                                mse=0.0, reason=rreason,
                                anomaly_time_ns=time.time_ns(),
                            )
                            continue

                        # Drain3 + Isolation Forest check
                        is_anom, dreason = detector.process(entry, vec)
                        if is_anom:
                            await on_anomaly(
                                client=client, source="log", entry=entry,
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
                                client=client, source="log", entry=entry,
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
```

---

## Step 6 — `drain3/requirements.txt` (updated)

```
httpx==0.28.1
numpy==2.1.3
drain3==0.9.11
scikit-learn==1.5.2
```

Torch is installed separately in the Dockerfile via the CPU wheel index (unchanged).

---

## Step 7 — `drain3/Dockerfile` (updated)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY metric_features.py log_anomaly.py metric_lstm.py trace_lookup.py \
     models.py trace_summary.py main.py .

CMD ["python", "-u", "main.py"]
```

---

## Step 8 — `docker-compose.yaml` (drain3 service block)

Add the new environment variables to the existing drain3 service:

```yaml
drain3:
  build: ./drain3
  depends_on:
    - log-stream
    - otel
  extra_hosts:
    - "host.docker.internal:host-gateway"
  environment:
    - LOG_STREAM_URL=http://log-stream:8080/v1/stream
    - TEMPO_URL=http://otel:3200
    - LOKI_URL=http://otel:3100
    - PROM_URL=http://otel:9090
    - SLM_URL=http://host.docker.internal:11434/api/generate
    - SLM_MODEL=llama3.2
    # Log LSTM (unchanged)
    - DRAIN3_WINDOW=24
    - DRAIN3_FEAT_DIM=8
    - DRAIN3_WARMUP=64
    - DRAIN3_Z=4.5
    - DRAIN3_FORCE_RULES=1
    # - DRAIN3_CHECKPOINT=/checkpoints/lstm_best.pt
    # Metric LSTM
    - PROM_SCRAPE_INTERVAL=15
    - DRAIN3_METRIC_WINDOW=24
    - DRAIN3_METRIC_WARMUP=32
    - DRAIN3_METRIC_Z=4.0
    # - DRAIN3_METRIC_CHECKPOINT=/checkpoints/lstm_metric_best.pt
  # volumes:
  #   - ./offline_lstm/checkpoints:/checkpoints:ro
```

---

## Step 9 — offline training pipeline for metric LSTM

### `offline_lstm/fetch_prometheus.py` (new)

```python
"""
Pull Prometheus range data and write data/metric_raw.csv.
Run after the stack has been running for at least --minutes minutes.
"""
from __future__ import annotations
import argparse, time, math
import pandas as pd
import requests
from config import DATA, LOKI_URL
from metric_features import METRIC_QUERIES

PROM_URL = "http://localhost:9090"

def fetch_range(base: str, query: str, start: int, end: int, step: int) -> list[tuple[float, float]]:
    r = requests.get(f"{base}/api/v1/query_range", params={
        "query": query, "start": start, "end": end, "step": step,
    }, timeout=30)
    r.raise_for_status()
    results = r.json()["data"]["result"]
    if not results:
        return []
    return [(float(ts), float(val)) for ts, val in results[0]["values"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default=PROM_URL)
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--step", type=int, default=15, help="Prometheus step in seconds")
    ap.add_argument("--out", default=str(DATA / "metric_raw.csv"))
    args = ap.parse_args()

    end   = int(time.time())
    start = end - args.minutes * 60

    frames = []
    for q in METRIC_QUERIES:
        pairs = fetch_range(args.prom, q, start, end, args.step)
        df = pd.DataFrame(pairs, columns=["timestamp", q])
        frames.append(df.set_index("timestamp"))

    out = pd.concat(frames, axis=1).reset_index()
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows → {args.out}")

if __name__ == "__main__":
    main()
```

### `offline_lstm/prepare_metric_data.py` (new)

```python
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
    feats = np.stack([normalise(row[1:].tolist()) for row in df[METRIC_QUERIES].itertuples()], axis=0)

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
```

### `offline_lstm/train_metric.py` (new)

```python
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
```

---

## Step 10 — offline training workflow

```bash
cd offline_lstm
source .venv/bin/activate

# 1. Pull metric history (stack must be running with traffic)
export PROM_URL=http://localhost:9090
python fetch_prometheus.py --minutes 120 --step 15

# 2. Build windowed dataset
python prepare_metric_data.py

# 3. Train
python train_metric.py --epochs 40

# 4. Mount checkpoint into drain3 container
#    Add to docker-compose.yaml drain3 service:
#    volumes:
#      - ./offline_lstm/checkpoints:/checkpoints:ro
#    environment:
#      - DRAIN3_METRIC_CHECKPOINT=/checkpoints/lstm_metric_best.pt
```

For periodic retraining, add a cron job or a simple script that runs the three steps above on a schedule (e.g. nightly) and does `docker compose restart drain3` to pick up the new checkpoint.

---

## Key invariants to maintain

| Contract | Where enforced |
|---|---|
| `METRIC_QUERIES` order and strings identical in live and offline | `metric_features.py` is copied verbatim — never import across packages |
| `normalise()` function identical in live and offline | Same file, same copy |
| `DRAIN3_METRIC_WINDOW` matches `--window` used in `prepare_metric_data.py` | Set both to 24 (default) or override both consistently |
| MD5 URL hashing in log featurizer | Already shared, unchanged |
| Log LSTM `WINDOW` / `FEAT_DIM` match offline log training | Unchanged from original |

---

## Environment variables — complete reference

| Variable | Service | Default | Purpose |
|---|---|---|---|
| `LOG_STREAM_URL` | drain3 | required | SSE endpoint |
| `TEMPO_URL` | drain3 | `http://otel:3200` | Trace fetch |
| `LOKI_URL` | drain3 | `http://otel:3100` | Trace lookup fallback |
| `PROM_URL` | drain3 | `http://otel:9090` | Metric scrape + snapshot |
| `SLM_URL` | drain3 | — | Ollama API endpoint |
| `SLM_MODEL` | drain3 | `llama3.2` | Ollama model tag |
| `PROM_SCRAPE_INTERVAL` | drain3 | `15` | Seconds between metric polls |
| `DRAIN3_METRIC_WINDOW` | drain3 | `24` | Metric LSTM window length |
| `DRAIN3_METRIC_WARMUP` | drain3 | `32` | Samples before metric alerts fire |
| `DRAIN3_METRIC_Z` | drain3 | `4.0` | Metric threshold Z multiplier |
| `DRAIN3_METRIC_CHECKPOINT` | drain3 | — | Path to `lstm_metric_best.pt` |
| `DRAIN3_CHECKPOINT` | drain3 | — | Path to `lstm_best.pt` (log) |
| `DRAIN3_FORCE_RULES` | drain3 | `1` | Rule anomalies bypass model |
| `DRAIN3_Z` | drain3 | `4.5` | Log LSTM threshold Z multiplier |
