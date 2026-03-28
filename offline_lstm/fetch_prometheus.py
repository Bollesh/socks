"""
Pull Prometheus range data and write data/metric_raw.csv.
Run after the stack has been running for at least --minutes minutes.
"""
from __future__ import annotations
import argparse, time, math
import pandas as pd
import requests
# from config import DATA, LOKI_URL
from config import DATA
from metric_features import METRIC_QUERIES

PROM_URL = "http://localhost:9090"

def fetch_range(base: str, query: str, start: int, end: int, step: int) -> list[tuple[float, float]]:
    r = requests.get(f"{base}/api/v1/query_range", params={
        "query": query, "start": start, "end": end, "step": step,
    }, timeout=30)
    r.raise_for_status()
    results = r.json()["data"]["result"]
    if not results:
        return []
    return [(float(ts), float(val)) for ts, val in results[0]["values"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default=PROM_URL)
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--step", type=int, default=15, help="Prometheus step in seconds")
    ap.add_argument("--out", default=str(DATA / "metric_raw.csv"))
    args = ap.parse_args()

    end   = int(time.time())
    start = end - args.minutes * 60

    frames = []
    for q in METRIC_QUERIES:
        pairs = fetch_range(args.prom, q, start, end, args.step)
        df = pd.DataFrame(pairs, columns=["timestamp", q])
        frames.append(df.set_index("timestamp"))

    out = pd.concat(frames, axis=1).reset_index()
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows → {args.out}")

if __name__ == "__main__":
    main()
