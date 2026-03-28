"""
analyzer.py
===========
Handles Drain3 pattern learning and Tempo trace validation together.
"""

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

TEMPO_URL             = "http://localhost:3200"
ANOMALY_DURATION_MS   = 1000

SERVICES = ["locust", "catalogue", "carts", "orders", "payment", "user", "shipping"]


def make_miner():
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth  = 4
    return TemplateMiner(config=config)

miners = {service: make_miner() for service in SERVICES}


def validate_with_tempo(client, timestamp_ns: str) -> dict:
    ts_seconds = int(timestamp_ns) / 1e9
    start_s = int(ts_seconds - 3)
    end_s   = int(ts_seconds + 3)

    try:
        response = client.get(
            f"{TEMPO_URL}/api/search",
            params={
                "service.name": "locust",
                "start":        start_s,
                "end":          end_s,
                "limit":        20,
            },
            timeout=5.0,
        )
        response.raise_for_status()
        traces = response.json().get("traces", [])
    except Exception as e:
        return {"valid": True, "reason": f"Tempo unavailable: {e}"}

    if not traces:
        return {"valid": False, "reason": "no traces found in window"}

    for t in traces:
        duration_ms = t.get("durationMs", 0)

        if duration_ms > ANOMALY_DURATION_MS:
            return {
                "valid":    True,
                "reason":   f"slow trace found: {duration_ms}ms",
                "trace_id": t.get("traceID", ""),
            }

        span_set = t.get("spanSet", {})
        spans    = span_set.get("spans", []) if span_set else []
        for span in spans:
            attrs = span.get("attributes", [])
            for attr in attrs:
                if attr.get("key") == "http.status_code":
                    val = attr.get("value", {}).get("intValue", 0)
                    if int(val) >= 500:
                        return {
                            "valid":    True,
                            "reason":   f"error span found: HTTP {val}",
                            "trace_id": t.get("traceID", ""),
                        }

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

    validation = validate_with_tempo(client, timestamp_ns)
    if validation["valid"]:
        print(f"        ✓ CONFIRMED by Tempo: {validation['reason']}")
        print(f"        → Send to remediation")
    else:
        print(f"        ✗ SUPPRESSED: {validation['reason']}")
    print()