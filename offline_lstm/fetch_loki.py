"""
Pull historical log lines from Grafana Loki and write data/raw_logs.csv.

Requires Loki reachable (e.g. docker compose otel on :3100).
Use --demo to synthesize a CSV Sample without Loki.
"""
from __future__ import annotations

import argparse
import time
import urllib.parse

import pandas as pd
import requests

from config import DATA, DEFAULT_QUERY, LOKI_URL, RAW_CSV


def fetch_range(
    base_url: str,
    query: str,
    start_ns: int,
    end_ns: int,
    chunk_s: int = 120,
    limit: int = 5000,
) -> list[dict]:
    rows: list[dict] = []
    t = start_ns
    while t < end_ns:
        chunk_end = min(t + chunk_s * 1_000_000_000, end_ns)
        params = {
            "query": query,
            "start": str(t),
            "end": str(chunk_end),
            "limit": str(limit),
            "direction": "FORWARD",
        }
        url = f"{base_url}/loki/api/v1/query_range?{urllib.parse.urlencode(params)}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for stream in payload.get("data", {}).get("result", []):
            labels = stream.get("stream", {}) or {}
            for ts_str, line in stream.get("values", []) or []:
                row = {
                    "timestamp_ns": int(ts_str),
                    "message": line,
                    "level": labels.get("level", ""),
                    "service_name": labels.get("service_name", ""),
                    "method": labels.get("method", ""),
                    "url": labels.get("url", ""),
                    "status": labels.get("status", ""),
                    "duration_ms": labels.get("duration_ms", ""),
                    "trace_id": labels.get("trace_id", ""),
                }
                for k, v in labels.items():
                    if k not in row:
                        row[f"label_{k}"] = v
                rows.append(row)
        t = chunk_end + 1
    return rows


def write_demo_csv(path, n: int = 4000) -> None:
    rnd = __import__("random")
    rnd.seed(42)
    urls = ["/catalogue", "/cart [view]", "/cart [add]", "/orders", "/"]
    levels = ["info"] * 85 + ["warning"] * 10 + ["error"] * 5
    t0 = int(time.time() * 1e9)
    rows = []
    for i in range(n):
        lev = rnd.choice(levels)
        st = rnd.choice([200, 200, 200, 404, 500] if lev != "info" else [200, 201])
        dur = rnd.expovariate(1 / 50) * 1000 if lev == "info" else rnd.uniform(2000, 12000)
        rows.append(
            {
                "timestamp_ns": t0 + i * 5_000_000,
                "message": f"demo request {i}",
                "level": lev,
                "service_name": "locust",
                "method": "GET",
                "url": rnd.choice(urls),
                "status": str(st),
                "duration_ms": str(round(dur, 2)),
                "trace_id": f"trace{i:06x}" if i % 17 == 0 else "",
            }
        )
    pd.DataFrame(rows).sort_values("timestamp_ns").to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Fetch Loki logs into CSV")
    ap.add_argument("--demo", action="store_true", help="Write synthetic demo CSV (no Loki)")
    ap.add_argument("--loki", default=LOKI_URL, help="Loki base URL")
    ap.add_argument("--query", default=DEFAULT_QUERY, help="LogQL query")
    ap.add_argument("--minutes", type=int, default=60, help="Lookback minutes from now (non-demo)")
    ap.add_argument("--out", type=str, default=str(RAW_CSV), help="Output CSV path")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if args.demo:
        write_demo_csv(args.out, n=4000)
        print(f"Wrote demo {args.out}")
        return

    end = time.time()
    start = end - args.minutes * 60
    print(f"Querying Loki {args.loki} query={args.query!r} ({args.minutes}m window UTC)")
    rows = fetch_range(args.loki, args.query, int(start * 1e9), int(end * 1e9))
    if not rows:
        print("No rows returned; extend --minutes or check query / Loki.")
        return
    df = pd.DataFrame(rows).sort_values("timestamp_ns")
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
