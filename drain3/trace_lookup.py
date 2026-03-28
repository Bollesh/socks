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
