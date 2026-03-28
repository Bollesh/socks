"""
Anomaly Detection Dashboard — backend.
Receives anomaly events from drain3, streams them to the frontend via SSE,
and proxies Prometheus for live metrics.
"""
import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

PROM_URL = os.environ.get("PROM_URL", "http://otel:9090").rstrip("/")
LOG_STREAM_URL = os.environ.get("LOG_STREAM_URL", "http://log-stream:8080").rstrip("/")

MAX_HISTORY = 200
anomaly_history: deque[dict] = deque(maxlen=MAX_HISTORY)
subscribers: set[asyncio.Queue] = set()
stats = {
    "log_anomalies": 0,
    "metric_anomalies": 0,
    "slm_calls": 0,
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    subscribers.clear()

app = FastAPI(title="Anomaly Dashboard", lifespan=lifespan)


# ── Anomaly ingestion (from drain3) ──────────────────────────────────────────
@app.post("/api/anomalies")
async def ingest_anomaly(event: dict[str, Any]):
    event["timestamp"] = event.get("timestamp") or time.time()
    event["id"] = f"{event['timestamp']:.6f}-{stats['log_anomalies'] + stats['metric_anomalies']}"
    anomaly_history.append(event)

    source = event.get("source", "unknown")
    if source == "log":
        stats["log_anomalies"] += 1
    elif source == "metric":
        stats["metric_anomalies"] += 1
    if event.get("slm_response"):
        stats["slm_calls"] += 1

    data = json.dumps(event, default=str)
    dead = []
    for q in subscribers:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        subscribers.discard(q)

    return Response(status_code=204)


# ── SSE stream for frontend ─────────────────────────────────────────────────
@app.get("/api/anomalies/stream")
async def anomaly_stream(request: Request):
    async def gen():
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        subscribers.add(q)
        try:
            ping = 0
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=10.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    ping += 1
                    yield f": keepalive {ping}\n\n"
        finally:
            subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


# ── History + stats ──────────────────────────────────────────────────────────
@app.get("/api/anomalies/history")
async def get_history():
    return list(anomaly_history)


@app.get("/api/stats")
async def get_stats():
    return {
        **stats,
        "uptime_s": round(time.time() - stats["start_time"], 1),
        "total_anomalies": stats["log_anomalies"] + stats["metric_anomalies"],
        "subscribers": len(subscribers),
        "history_size": len(anomaly_history),
    }


# ── Prometheus proxy ─────────────────────────────────────────────────────────
@app.get("/api/prom/query")
async def prom_query(query: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=8.0)
        return r.json()


@app.get("/api/prom/query_range")
async def prom_query_range(query: str, start: str, end: str, step: str = "15s"):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{PROM_URL}/api/v1/query_range", params={
            "query": query, "start": start, "end": end, "step": step
        }, timeout=8.0)
        return r.json()


# ── Log-stream health proxy ─────────────────────────────────────────────────
@app.get("/api/logstream/health")
async def logstream_health():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{LOG_STREAM_URL}/health", timeout=5.0)
        return r.json()


# ── Serve static files ──────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("/app/static/index.html")

app.mount("/static", StaticFiles(directory="/app/static"), name="static")
