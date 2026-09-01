"""
analyzer.py
===========
Handles Drain3 pattern learning and Tempo trace validation together.
"""

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

TEMPO_URL             = "http://localhost:3200"
ANOMALY_DURATION_MS   = 1000
VALIDATION_WINDOW_S   = 3

SERVICES = ["locust", "catalogue", "carts", "orders", "payment", "user", "shipping"]


def make_miner():
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth  = 4
    return TemplateMiner(config=config)

miners = {service: make_miner() for service in SERVICES}


def _search(client, query, start_s, end_s, limit=20):
    """One Tempo TraceQL search. Raises on transport/HTTP failure."""
    response = client.get(
        f"{TEMPO_URL}/api/search",
        params={
            "q":     query,
            "start": start_s,   # unix epoch seconds
            "end":   end_s,
            "limit": limit,
        },
        timeout=5.0,
    )
    response.raise_for_status()
    return response.json().get("traces", [])


def validate_with_tempo(client, service: str, timestamp_ns: str) -> dict:
    """
    Ask Tempo whether anything actually went wrong in `service` around the
    time this log line was written. A trace counts as anomalous if it ran
    longer than ANOMALY_DURATION_MS or carries an error / HTTP 5xx span.
    """
    ts_seconds = int(timestamp_ns) / 1e9
    start_s = int(ts_seconds - VALIDATION_WINDOW_S)
    end_s   = int(ts_seconds + VALIDATION_WINDOW_S)

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
        # Fail open: a broken validator must not hide real problems.
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

    # Nothing anomalous. Distinguish "traces were fine" from "no traces at all".
    try:
        any_traces = _search(client, "{ " + service_filter + " }", start_s, end_s, limit=1)
    except Exception as e:
        return {"valid": True, "reason": f"Tempo unavailable: {e}"}

    if not any_traces:
        return {"valid": False, "reason": f"no {service} traces found in window"}

    return {"valid": False, "reason": "all traces normal in window"}


def process_line(service, line, client, timestamp_ns):
    miner = miners.get(service)
    if not miner:
        return

    result = miner.add_log_message(line)

    if result is None:
        return

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

    validation = validate_with_tempo(client, service, timestamp_ns)
    if validation["valid"]:
        print(f"        ✓ CONFIRMED by Tempo: {validation['reason']}")
        print(f"        → Send to remediation")
    else:
        print(f"        ✗ SUPPRESSED: {validation['reason']}")
    print()
