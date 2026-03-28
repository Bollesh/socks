"""
HTTP log hub: producers POST JSON lines, consumers open an SSE stream.
Designed for real-time fan-out to processors like drain3.
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("log-stream")

MAX_PENDING_PER_CLIENT = int(os.environ.get("SSE_QUEUE_MAX", "256"))
subscribers: set[asyncio.Queue] = set()
ingests_total = 0


async def _broadcast(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"))
    dead: list[asyncio.Queue] = []
    for q in subscribers:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        subscribers.discard(q)
        log.warning("Dropped slow SSE client")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("log-stream ready, path /v1/logs (POST), /v1/stream (GET SSE)")
    yield
    subscribers.clear()


app = FastAPI(title="Log stream", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "subscribers": len(subscribers), "ingests_total": ingests_total}


@app.post("/v1/logs")
async def ingest(entry: dict[str, Any]):
    """Accept a single log object (arbitrary JSON). Pushes to all SSE subscribers."""
    global ingests_total
    ingests_total += 1
    if ingests_total == 1 or ingests_total % 500 == 0:
        log.info("ingested %s log payloads (SSE clients: %s)", ingests_total, len(subscribers))
    await _broadcast(entry)
    return Response(status_code=204)


@app.get("/v1/stream")
async def stream(request: Request):
    """Server-Sent Events of JSON log entries, one per `data:` line."""

    async def gen():
        q: asyncio.Queue = asyncio.Queue(maxsize=MAX_PENDING_PER_CLIENT)
        subscribers.add(q)
        log.info("SSE client connected; total=%s", len(subscribers))
        try:
            ping = 0
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    ping += 1
                    yield f": keepalive {ping}\n\n"
                    continue
                yield f"data: {line}\n\n"
        finally:
            subscribers.discard(q)
            log.info("SSE client left; total=%s", len(subscribers))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
