"""
loki_poller.py
==============
Handles polling Loki for new log entries.
"""

import httpx

LOKI_URL   = "http://localhost:3100"
LOKI_QUERY = '{service_name=~"locust|catalogue|carts|orders|payment|user|shipping"}'


def poll_loki(client, start_ns, end_ns):
    try:
        response = client.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={
                "query":     LOKI_QUERY,
                "start":     start_ns,
                "end":       end_ns,
                "limit":     1000,
                "direction": "forward",
            },
            timeout=5.0,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[LOKI] Poll failed: {e}")
        return []

    entries = []

    for stream_block in response.json().get("data", {}).get("result", []):
        service = stream_block.get("stream", {}).get("service_name", "unknown")
        for timestamp_ns, line in stream_block.get("values", []):
            entries.append({
                "service":      service,
                "line":         line,
                "timestamp_ns": timestamp_ns,
            })

    return entries