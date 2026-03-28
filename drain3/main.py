"""
drain3: consumes the log-stream SSE, scores windows with an LSTM predictor,
enriches anomalies with Tempo trace summaries, forwards to an SLM (e.g. Ollama).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sys
from collections import deque
from typing import Any

# Unbuffered banner before heavy imports so `docker logs` is never silent at boot.
print(
    "drain3: starting (importing torch may take 10-60s on first run)...",
    file=sys.stderr,
    flush=True,
)

import httpx
import numpy as np
import torch

from models import LSTMPredictor, bootstrap_train
from trace_summary import summarize_trace_json

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("drain3")
log.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

LOG_STREAM_URL = os.environ["LOG_STREAM_URL"]
TEMPO_URL = os.environ.get("TEMPO_URL", "http://otel:3200").rstrip("/")
SLM_URL = os.environ.get("SLM_URL", "").strip()
SLM_MODEL = os.environ.get("SLM_MODEL", "llama3.2")
WINDOW = int(os.environ.get("DRAIN3_WINDOW", "24"))
FEAT_DIM = int(os.environ.get("DRAIN3_FEAT_DIM", "8"))
WARMUP_WINDOWS = int(os.environ.get("DRAIN3_WARMUP", "64"))
ERROR_RING = int(os.environ.get("DRAIN3_ERROR_RING", "400"))
Z_MULT = float(os.environ.get("DRAIN3_Z", "4.5"))
FORCE_RULES = os.environ.get("DRAIN3_FORCE_RULES", "1") == "1"
# Optional: path to offline_lstm/checkpoints/lstm_best.pt (bind-mount into container)
CHECKPOINT_PATH = os.environ.get("DRAIN3_CHECKPOINT", "").strip()


def _featurize(entry: dict[str, Any]) -> np.ndarray:
    labels = entry.get("labels") if isinstance(entry.get("labels"), dict) else {}
    level = str(entry.get("level") or labels.get("level") or "info").lower()
    is_error = 1.0 if level == "error" else 0.0

    dur_raw = labels.get("duration_ms")
    try:
        dur = float(dur_raw)
    except (TypeError, ValueError):
        dur = 0.0
    dur = math.log1p(max(dur, 0.0)) / 10.0

    stat = labels.get("status") or entry.get("status")
    try:
        sc = int(stat)
    except (TypeError, ValueError):
        sc = 0
    status_n = min(max(sc / 600.0, 0.0), 1.0)
    is_5xx = 1.0 if sc >= 500 else 0.0

    url = str(labels.get("url") or "")
    # Stable across runs (must match offline_lstm/features.py for trained checkpoints).
    url_hash = (
        int(hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8], 16) % 10_001
    ) / 10_000.0
    msg = str(entry.get("message") or entry.get("line") or "")
    msg_len = min(len(msg) / 400.0, 1.0)
    has_trace = 1.0 if entry.get("trace_id") else 0.0

    return np.array(
        [dur, is_error, status_n, is_5xx, url_hash, msg_len, has_trace, 0.0],
        dtype=np.float32,
    )


def _rule_anomaly(entry: dict[str, Any]) -> tuple[bool, str]:
    labels = entry.get("labels") if isinstance(entry.get("labels"), dict) else {}
    level = str(entry.get("level") or labels.get("level") or "").lower()
    if level == "error":
        return True, "rule:log_level_error"
    stat = labels.get("status") or entry.get("status")
    try:
        sc = int(stat)
    except (TypeError, ValueError):
        sc = 0
    if sc >= 500:
        return True, "rule:http_5xx"
    if entry.get("force_anomaly"):
        return True, "rule:force_anomaly"
    return False, ""


def _percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


async def _fetch_trace(client: httpx.AsyncClient, trace_id: str) -> dict | None:
    if not trace_id:
        return None
    tid = trace_id.strip().lower().replace("0x", "")
    r = await client.get(f"{TEMPO_URL}/api/traces/{tid}")
    if r.status_code == 404:
        log.info("Tempo 404 for trace_id=%s", tid[:16])
        return None
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None


async def _slm_ollama(client: httpx.AsyncClient, prompt: str) -> str:
    if not SLM_URL:
        return "(SLM_URL not set; skipping inference)"
    body = {
        "model": SLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = await client.post(SLM_URL, json=body, timeout=120.0)
    r.raise_for_status()
    data = r.json()
    return str(data.get("response", data))


async def handle_anomaly(
    client: httpx.AsyncClient,
    entry: dict[str, Any],
    mse: float,
    reason: str,
) -> None:
    trace_id = str(entry.get("trace_id") or "")
    trace_payload = await _fetch_trace(client, trace_id)
    trace_blurb = summarize_trace_json(trace_payload)

    payload = json.dumps(entry, indent=2, default=str)
    prompt = (
        "You are an SRE assistant. A log anomaly was flagged.\n"
        f"Reason: {reason}\n"
        f"Model MSE (window predictor): {mse:.6f}\n\n"
        "Log entry:\n"
        f"{payload}\n\n"
        "Tempo trace digest (if any):\n"
        f"{trace_blurb}\n\n"
        "Give a short diagnosis and next debugging steps."
    )
    try:
        reply = await _slm_ollama(client, prompt)
        log.warning("SLM anomaly output trace=%s\n%s", trace_id[:16] if trace_id else "-", reply)
    except Exception as exc:
        log.error("SLM call failed: %s", exc)


async def run() -> None:
    device = torch.device("cpu")
    log.info("Building LSTM (feat_dim=%s)...", FEAT_DIM)
    model = LSTMPredictor(FEAT_DIM).to(device)
    if CHECKPOINT_PATH:
        try:
            ckpt = torch.load(
                CHECKPOINT_PATH, map_location=device, weights_only=False
            )
        except TypeError:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        w = ckpt.get("window")
        if w is not None and int(w) != WINDOW:
            log.warning("Checkpoint window %s != env WINDOW %s", w, WINDOW)
        model.load_state_dict(ckpt["state_dict"])
        log.info("Loaded LSTM weights from %s", CHECKPOINT_PATH)
    else:
        log.info("Bootstrap training (200 synthetic steps)...")
        bootstrap_train(model, device, WINDOW, FEAT_DIM)
    log.info("LSTM ready on cpu; window=%s", WINDOW)

    feat_buf: deque[np.ndarray] = deque(maxlen=WINDOW + 5)
    err_ring: deque[float] = deque(maxlen=ERROR_RING)
    parsed = 0

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60.0)) as client:
        while True:
            try:
                async with client.stream("GET", LOG_STREAM_URL) as resp:
                    resp.raise_for_status()
                    log.info("Connected to log stream: %s", LOG_STREAM_URL)
                    buf = ""
                    async for chunk in resp.aiter_text():
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            json_text = line[5:].strip()
                            try:
                                entry = json.loads(json_text)
                            except json.JSONDecodeError:
                                continue
                            v = _featurize(entry)
                            feat_buf.append(v)
                            parsed += 1
                            if parsed == 1 or parsed % 500 == 0:
                                log.info("parsed %s log events from stream", parsed)

                            ra, rreason = _rule_anomaly(entry)
                            if FORCE_RULES and ra:
                                await handle_anomaly(client, entry, 0.0, rreason)
                                continue

                            if len(feat_buf) < WINDOW:
                                continue

                            win = np.stack(list(feat_buf)[-WINDOW:], axis=0)
                            x = torch.from_numpy(win[:-1]).unsqueeze(0).to(device)
                            y_t = torch.from_numpy(win[-1]).unsqueeze(0).to(device)
                            with torch.inference_mode():
                                pred = model(x)
                                mse = float(torch.mean((pred - y_t) ** 2).cpu())

                            err_ring.append(mse)
                            if len(err_ring) < WARMUP_WINDOWS:
                                continue

                            arr = np.array(err_ring, dtype=np.float64)
                            thr = max(
                                _percentile(arr, 0.985),
                                float(np.mean(arr) + Z_MULT * max(np.std(arr), 1e-6)),
                            )
                            if mse > thr:
                                await handle_anomaly(
                                    client,
                                    entry,
                                    mse,
                                    f"model:lstm_mse>{thr:.6f}",
                                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("stream error, reconnecting in 2s: %s", exc, exc_info=True)
                await asyncio.sleep(2.0)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("drain3: stopped.", file=sys.stderr, flush=True)
