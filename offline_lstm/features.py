"""Feature vectors aligned with drain3 LSTM input (must stay in sync)."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

FEAT_NAMES = (
    "dur_log_norm",
    "is_error",
    "status_norm",
    "is_5xx",
    "url_bucket",
    "msg_len_norm",
    "has_trace",
    "reserved",
)


def stable_url_bucket(url: str) -> float:
    h = int(hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:8], 16) % 10_001
    return h / 10_000.0


def featurize(entry: dict[str, Any]) -> np.ndarray:
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
        sc = int(float(stat))
    except (TypeError, ValueError):
        sc = 0
    status_n = min(max(sc / 600.0, 0.0), 1.0)
    is_5xx = 1.0 if sc >= 500 else 0.0

    url = str(labels.get("url") or "")
    url_b = stable_url_bucket(url)
    msg = str(entry.get("message") or entry.get("line") or "")
    msg_len = min(len(msg) / 400.0, 1.0)
    tid = entry.get("trace_id") or labels.get("trace_id") or ""
    has_trace = 1.0 if str(tid).strip() else 0.0

    return np.array(
        [dur, is_error, status_n, is_5xx, url_b, msg_len, has_trace, 0.0],
        dtype=np.float32,
    )


def row_to_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Build drain3-shaped dict from a CSV / DataFrame row."""
    labels = {
        "method": row.get("method", "") or "",
        "url": row.get("url", "") or "",
        "status": row.get("status", "") or "",
        "duration_ms": row.get("duration_ms", "") or "",
        "trace_id": row.get("trace_id", "") or "",
    }
    return {
        "message": row.get("message", "") or "",
        "level": row.get("level", "info") or "info",
        "labels": labels,
        "trace_id": row.get("trace_id", "") or "",
    }
