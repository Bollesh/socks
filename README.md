# Sock Shop Anomaly Detection Pipeline

A log-anomaly detection experiment built on top of the [Weaveworks Sock Shop](https://microservices-demo.github.io/) microservices demo.

The idea is simple: **logs tell you something *new* happened, traces tell you whether it *mattered*.** Drain3 learns the normal shape of every service's log lines and flags novel templates; each flagged line is then cross-checked against Tempo traces from the same service at the same moment. Novel-but-boring lines get suppressed, novel-and-slow/erroring lines get confirmed as real anomalies.

## Architecture

```
                 ┌──────────────────────────────────────────┐
                 │  Sock Shop microservices (docker-compose) │
                 │  front-end, catalogue, carts, orders,     │
                 │  payment, user, shipping, queue-master    │
                 └──────────────┬───────────────────────────┘
                                │ traffic
                    ┌───────────┴───────────┐
                    │  Locust load generator │
                    │  (OTel-instrumented)   │
                    └───────────┬───────────┘
                                │ logs (Loki push) + traces (OTLP)
                 ┌──────────────┴───────────────────────────┐
                 │  grafana/otel-lgtm                        │
                 │  Loki :3100  Tempo :3200  Prom :9090      │
                 │  Grafana :3000                            │
                 └──────┬────────────────────┬──────────────┘
                        │ query_range        │ /api/search (TraceQL)
                        ▼                    ▼
                 ┌────────────────────────────────────┐
                 │  main.py                            │
                 │   loki_poller  →  analyzer          │
                 │   (Drain3 per service)              │
                 │   novel template? → validate w/Tempo│
                 │   → CONFIRMED  or  SUPPRESSED       │
                 └────────────────────────────────────┘
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `docker-compose.yaml` | Sock Shop services, the Locust container, and the `grafana/otel-lgtm` all-in-one observability stack. |
| `locust/locustfile.py` | Load test simulating a full shopping journey (browse → register/login → cart → checkout → order history). Instrumented with OpenTelemetry; also pushes structured logs directly to Loki under `service_name="locust"`. |
| `locust/Dockerfile` | Locust image with the OTel SDK and exporters. |
| `main.py` | Entry point for the detection pipeline. Polls Loki on a fixed interval and feeds each line to the analyzer. |
| `loki_poller.py` | Loki `query_range` client. Flattens the response into `{service, line, timestamp_ns}` entries. |
| `analyzer.py` | Drain3 template mining (one miner per service) plus Tempo-based validation of novel patterns. |
| `loki_to_drain3.py` | The original single-file version of the pipeline, kept with its explanatory comments. `main.py` + `loki_poller.py` + `analyzer.py` are the split-up refactor of this file; both are kept in sync. |
| `export_observability.sh` | Dumps Loki logs, Tempo traces, and Prometheus metrics/histograms from the last hour into `exports/` for offline analysis. |

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ on the host (the pipeline runs outside Docker and talks to the stack over `localhost`)

## Running it

### 1. Bring up the stack

```bash
docker compose up -d
```

Wait a minute for the services to settle, then open:

| Service | URL | Notes |
| --- | --- | --- |
| Sock Shop front-end | http://localhost:8079 | |
| Locust web UI | http://localhost:8089 | Target host is preset to `http://front-end:8079` |
| Grafana | http://localhost:3000 | `admin` / `admin` |
| Loki API | http://localhost:3100 | |
| Tempo API | http://localhost:3200 | |
| Prometheus | http://localhost:9090 | |

### 2. Generate load

Open the Locust UI at http://localhost:8089, set a user count and spawn rate, and start the swarm. Locust writes logs to Loki and traces to the OTLP HTTP endpoint as it runs.

### 3. Run the detection pipeline

```bash
pip install httpx drain3
python main.py
```

Output looks like:

```
Starting Loki → Drain3 pipeline...
Polling every 2s

[POLL] Got 143 log lines
[NOVEL] service=locust
        change=cluster_created
        raw line : Request completed | GET /orders 500
        template : Request completed | GET /orders <*>
        ✓ CONFIRMED by Tempo: error span found in trace
        → Send to remediation

[NOVEL] service=locust
        change=cluster_template_changed
        raw line : Viewing item 3395a43e-2d88-40de-b95f-e00e1502085b
        template : Viewing item <*>
        ✗ SUPPRESSED: all traces normal in window
```

### 4. (Optional) Export the raw telemetry

```bash
./export_observability.sh
```

Writes to `exports/`: `loki_logs.json`, one JSON file per trace under `tempo_traces/`, and Prometheus instant + range queries under `prometheus/`.

## How detection works

**Template mining.** Each service gets its own `drain3.TemplateMiner` — mixing log formats from different services degrades template quality. Configured with `drain_sim_th = 0.4` (fairly sensitive) and `drain_depth = 4`. Drain3 reports one of three change types per line:

- `cluster_created` — a pattern never seen before → **novel**
- `cluster_template_changed` — an existing pattern evolved → **novel**
- `none` — routine, ignored

**Trace validation.** A novel template alone is a weak signal; a new log line often just means a new but harmless code path. So `validate_with_tempo()` searches Tempo for traces **from the same service that produced the log line**, within a ±`VALIDATION_WINDOW_S` window around its timestamp. The filtering is pushed into Tempo as a single TraceQL query rather than done client-side:

```traceql
{ resource.service.name = "carts" && (trace:duration > 1000ms
                                      || status = error
                                      || span.http.status_code >= 500) }
```

If that returns anything, the anomaly is **confirmed** (and the matching `traceID` is attached). If it returns nothing, a second cheap query checks whether the service produced *any* traces in the window at all, so the two very different outcomes stay distinguishable:

- `no <service> traces found in window` — nothing to correlate against
- `all traces normal in window` — traffic was healthy, suppress

If Tempo itself is unreachable the anomaly is let through (fail-open), so a broken validator never silently hides real problems.

> Note on the Tempo API: `/api/search` has no `service.name` query parameter. Filtering must go through either `tags=` (logfmt) or `q=` (TraceQL). Passing `service.name` directly is silently ignored and returns *every* trace in the window — this pipeline uses `q=`.

**Polling cursor.** `main.py` starts one minute in the past and, after each poll, advances the cursor to the last entry's timestamp + 1 ns so lines are never processed twice.

## Configuration

There is no config file — the knobs are module-level constants:

| Constant | File | Default | Meaning |
| --- | --- | --- | --- |
| `POLL_INTERVAL` | `main.py` | `2` | Seconds between Loki polls. |
| `LOKI_URL` | `loki_poller.py` | `http://localhost:3100` | Loki base URL. |
| `LOKI_QUERY` | `loki_poller.py` | regex over the 7 services | LogQL stream selector. |
| `TEMPO_URL` | `analyzer.py` | `http://localhost:3200` | Tempo base URL. |
| `ANOMALY_DURATION_MS` | `analyzer.py` | `1000` | Traces slower than this confirm an anomaly. |
| `VALIDATION_WINDOW_S` | `analyzer.py` | `3` | Half-width, in seconds, of the Tempo search window around a log line. |
| `SERVICES` | `analyzer.py` | 7 services | One Drain3 miner is created per name here. |
| `drain_sim_th` | `analyzer.py` | `0.4` | Drain3 similarity threshold — lower means more clusters, more novelty. |

The Loki stream label `service_name` and the OTel resource attribute `service.name` are expected to carry the same value for a given service — that is what lets a log line be matched to its own traces.

## Load test coverage

`locust/locustfile.py` weights its tasks toward realistic browsing, with a smaller share of write-heavy flows:

- **catalogue** — paged listing, tag filters, item detail, size
- **cart** — add, view, quantity update, delete
- **user** — profile, addresses, cards, and writes to both
- **orders** — order history, plus a full checkout that fans out to payment, shipping, and queue-master
- **frontend** — home page and static assets

Every user registers a fresh account (`locust_<random>`) in `on_start` and logs in with HTTP basic auth. All request outcomes are pushed to Loki with `method`, `url`, `status`, and `duration_ms` labels.

## Notes and known rough edges

- **Only Locust is instrumented.** The Sock Shop images in this compose file emit neither OTLP traces nor logs to Loki, so in practice only the `locust` Drain3 miner sees traffic and only `locust` traces exist to validate against. The other six miners are wired up and ready but stay empty until log shipping / instrumentation is added for those services.
- The pipeline runs on the host, not in Compose, and hardcodes `localhost` ports — running it inside the Compose network would need the URLs changed to `otel:3100` / `otel:3200`.
- Drain3 state is in-memory only. Restarting `main.py` relearns every template from scratch, which produces a burst of `cluster_created` events on startup.
- Validation costs one Tempo query per novel line, plus a second only when the first finds nothing. On a cold start (when everything looks novel) that is a lot of queries in a short window.
- The TraceQL uses the scoped intrinsic `trace:duration`, which needs a reasonably recent Tempo. `grafana/otel-lgtm:latest` ships one; if you pin an older image and searches come back empty, that is the first thing to check.
