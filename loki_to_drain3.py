"""
Loki → Drain3 Pipeline
======================
Just does one thing: polls Loki every 2 seconds, feeds each log line
into Drain3, and prints what Drain3 thinks about it.

Install: pip install httpx drain3
Run:     python loki_to_drain3.py
"""

import time
import httpx
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LOKI_URL       = "http://localhost:3100"
POLL_INTERVAL  = 2   # seconds
TEMPO_URL             = "http://localhost:3200"
ANOMALY_DURATION_MS   = 1000   # traces slower than this are flagged as real
VALIDATION_WINDOW_S   = 3      # +/- seconds around the log line to search Tempo
# All your services in one regex query
# =~ means "regex match" in LogQL
# The | between names means OR
LOKI_QUERY = '{service_name=~"locust|catalogue|carts|orders|payment|user|shipping"}'

# ─────────────────────────────────────────────────────────────────────────────
# DRAIN3 SETUP — one instance per service
# ─────────────────────────────────────────────────────────────────────────────
# Why one per service?
# Each service has different log formats.
# Mixing them confuses Drain3's template learning.

SERVICES = ["locust", "catalogue", "carts", "orders", "payment", "user", "shipping"]

def make_miner():
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4   # similarity threshold — lower = more sensitive
    config.drain_depth  = 4
    return TemplateMiner(config=config)

miners = {service: make_miner() for service in SERVICES}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: POLL LOKI
# ─────────────────────────────────────────────────────────────────────────────
# We ask Loki: "give me all logs between last_timestamp and now"
# Loki returns them as a JSON list grouped by service
# We flatten that into a simple list of {service, line, timestamp}

def poll_loki(client, start_ns, end_ns):
    try:
        response = client.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={
                "query":     LOKI_QUERY,
                "start":     start_ns,   # nanosecond unix timestamp
                "end":       end_ns,     # nanosecond unix timestamp
                "limit":     1000,       # max lines per poll
                "direction": "forward",  # oldest first
            },
            timeout=5.0,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[LOKI] Poll failed: {e}")
        return []

    entries = []

    # Loki response looks like:
    # {
    #   "data": {
    #     "result": [
    #       {
    #         "stream": { "service_name": "carts", "level": "ERROR" },
    #         "values": [
    #           ["1711234567000000000", "ERROR NullPointer at CartService.java:42"],
    #           ["1711234568000000000", "INFO  GET /cart 200 12ms"],
    #         ]
    #       },
    #       ...one block per unique set of stream labels
    #     ]
    #   }
    # }
    for stream_block in response.json().get("data", {}).get("result", []):
        service = stream_block.get("stream", {}).get("service_name", "unknown")
        for timestamp_ns, line in stream_block.get("values", []):
            entries.append({
                "service":      service,
                "line":         line,
                "timestamp_ns": timestamp_ns,
            })

    return entries

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEED INTO DRAIN3
# ─────────────────────────────────────────────────────────────────────────────
# Drain3 reads each log line and decides:
#   "cluster_created"          → never seen this pattern before → NOVEL
#   "cluster_template_changed" → pattern evolved significantly  → NOVEL
#   "none"                     → seen this before               → ROUTINE
def process_line(service, line, client, timestamp_ns):
    miner = miners.get(service)
    if not miner:
        return

    result = miner.add_log_message(line)

    if result is None:
        return

    # FIX: newer drain3 returns a dict, not an object
    if isinstance(result, dict):
        change = result.get("change_type", "none")
        template = result.get("template", line)
    else:
        change = result.change_type.name
        template = result.cluster.get_template()

    if change == "none":
        return

    print(f"[NOVEL] service={service}")
    print(f"        change={change}")
    print(f"        raw line : {line}")
    print(f"        template : {template}")

    # Validate against Tempo before flagging as real anomaly
    validation = validate_with_tempo(client, service, timestamp_ns)
    if validation["valid"]:
        print(f"        ✓ CONFIRMED by Tempo: {validation['reason']}")
        print(f"        → Send to remediation")
    else:
        print(f"        ✗ SUPPRESSED: {validation['reason']}")
    print()
def _search(client, query, start_s, end_s, limit=20):
    """
    One Tempo search. Tempo has no `service.name` query parameter — you filter
    either with `tags=` (logfmt) or, as here, with `q=` (a TraceQL query).
    Raises on transport/HTTP failure; the caller decides what that means.
    """
    response = client.get(
        f"{TEMPO_URL}/api/search",
        params={
            "q":     query,
            "start": start_s,   # Tempo wants plain unix seconds
            "end":   end_s,
            "limit": limit,
        },
        timeout=5.0,
    )
    response.raise_for_status()
    return response.json().get("traces", [])


def validate_with_tempo(client, service: str, timestamp_ns: str) -> dict:
    """
    When Drain3 flags a novel pattern in `service` at timestamp T,
    ask Tempo whether anything actually went wrong in that service
    within a 3-second window around T.

    Returns:
        {"valid": True,  "reason": "slow trace found: 1240ms"}  → real anomaly
        {"valid": False, "reason": "all traces normal in window"} → suppress it
    """
    # Convert nanoseconds to seconds for Tempo
    ts_seconds = int(timestamp_ns) / 1e9
    start_s = int(ts_seconds - VALIDATION_WINDOW_S)
    end_s   = int(ts_seconds + VALIDATION_WINDOW_S)

    # Let Tempo do the filtering: one query that matches only traces which are
    # slow, errored, or carry an HTTP 5xx. Doing it server-side also means the
    # matched attributes come back in the response, which a bare tag search
    # would not give us.
    service_filter = f'resource.service.name = "{service}"'
    anomaly_query = (
        "{ " + service_filter + " && ("
        f"trace:duration > {ANOMALY_DURATION_MS}ms"
        " || status = error"
        " || span.http.status_code >= 500) }"
    )

    try:
        traces = _search(client, anomaly_query, start_s, end_s)
    except Exception as e:
        # If Tempo is unreachable, let the anomaly through
        return {"valid": True, "reason": f"Tempo unavailable: {e}"}

    if traces:
        trace = traces[0]
        duration_ms = trace.get("durationMs", 0)
        if duration_ms > ANOMALY_DURATION_MS:
            reason = f"slow trace found: {duration_ms}ms"
        else:
            reason = "error span found in trace"
        return {
            "valid":    True,
            "reason":   reason,
            "trace_id": trace.get("traceID", ""),
        }

    # Nothing anomalous matched. Was there any traffic at all for this service?
    # "no traces" and "traces, all healthy" are different stories.
    try:
        any_traces = _search(client, "{ " + service_filter + " }", start_s, end_s, limit=1)
    except Exception as e:
        return {"valid": True, "reason": f"Tempo unavailable: {e}"}

    if not any_traces:
        return {"valid": False, "reason": f"no {service} traces found in window"}

    return {"valid": False, "reason": "all traces normal in window"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Starting Loki → Drain3 pipeline...")
    print(f"Query: {LOKI_QUERY}")
    print(f"Polling every {POLL_INTERVAL}s\n")

    # Start cursor: 1 minute ago in nanoseconds
    # So on startup we catch any recent logs immediately
    last_ts = str(time.time_ns() - 60 * 1_000_000_000)

    with httpx.Client() as client:
        while True:
            loop_start = time.time()

            now_ns = str(time.time_ns())

            # Step 1: get new logs from Loki
            entries = poll_loki(client, last_ts, now_ns)

            if entries:
                print(f"[POLL] Got {len(entries)} log lines")

                # Advance cursor past the last log we received
                # +1 so we don't fetch the same line again next poll
                last_ts = str(int(entries[-1]["timestamp_ns"]) + 1)

                # Step 2: feed each line into its service's Drain3
                for entry in entries:
                    process_line(entry["service"], entry["line"], client, entry["timestamp_ns"])
            else:
                print("[POLL] No new logs")
                last_ts = now_ns

            # Sleep for the remainder of the 2-second interval
            elapsed = time.time() - loop_start
            time.sleep(max(0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    main()