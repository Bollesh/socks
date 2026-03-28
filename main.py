"""
main.py
=======
Entry point. Wires loki_poller and analyzer together.
"""

import time
import httpx

from loki_poller import poll_loki
from analyzer import process_line

POLL_INTERVAL = 2


def main():
    print("Starting Loki → Drain3 pipeline...")
    print(f"Polling every {POLL_INTERVAL}s\n")

    last_ts = str(time.time_ns() - 60 * 1_000_000_000)

    with httpx.Client() as client:
        while True:
            loop_start = time.time()

            now_ns = str(time.time_ns())

            entries = poll_loki(client, last_ts, now_ns)

            if entries:
                print(f"[POLL] Got {len(entries)} log lines")

                last_ts = str(int(entries[-1]["timestamp_ns"]) + 1)

                for entry in entries:
                    process_line(entry["service"], entry["line"], client, entry["timestamp_ns"])
            else:
                print("[POLL] No new logs")
                last_ts = now_ns

            elapsed = time.time() - loop_start
            time.sleep(max(0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    main()