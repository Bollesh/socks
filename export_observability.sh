#!/bin/bash
set -e
mkdir -p exports/tempo_traces

LOKI="http://localhost:3100"
TEMPO="http://localhost:3200"
PROM="http://localhost:9090"

# ── Loki logs ────────────────────────────────────────────────────────────────
echo "Exporting Loki logs..."
curl -s "${LOKI}/loki/api/v1/query_range" \
  --data-urlencode 'query={service_name="locust"}' \
  --data-urlencode "start=$(date -d '1 hour ago' +%s%N)" \
  --data-urlencode "end=$(date +%s%N)" \
  --data-urlencode "limit=5000" \
  | python3 -m json.tool > exports/loki_logs.json
echo "  → exports/loki_logs.json"

# ── Tempo traces ─────────────────────────────────────────────────────────────
echo "Exporting Tempo traces..."
curl -s "${TEMPO}/api/search?limit=1000" > exports/tempo_index.json
python3 -c "
import json, urllib.request
with open('exports/tempo_index.json') as f:
    data = json.load(f)
for t in data.get('traces', []):
    tid = t['traceID']
    urllib.request.urlretrieve(
        f'${TEMPO}/api/traces/{tid}',
        f'exports/tempo_traces/{tid}.json'
    )
print(f'  → {len(data[\"traces\"])} traces saved to exports/tempo_traces/')
"

# ── Prometheus metrics ────────────────────────────────────────────────────────
echo "Exporting Prometheus metrics..."

# 1. All metric names
curl -s "${PROM}/api/v1/label/__name__/values" \
  | python3 -m json.tool > exports/prometheus_metric_names.json
echo "  → exports/prometheus_metric_names.json"

# 2. Instant snapshot of every locust metric
LOCUST_METRICS=(
  "http_requests_total"
  "http_errors_total"
  "http_request_duration_ms"
  "locust_active_users"
  "orders_placed_total"
  "cart_operations_total"
)

mkdir -p exports/prometheus
for METRIC in "${LOCUST_METRICS[@]}"; do
  curl -s "${PROM}/api/v1/query?query=${METRIC}" \
    | python3 -m json.tool > "exports/prometheus/${METRIC}.json"
  echo "  → exports/prometheus/${METRIC}.json"
done

# 3. Range data for the last hour (useful for graphing / replay)
START=$(date -d '1 hour ago' +%s)
END=$(date +%s)
STEP="15"   # 15-second resolution — lower = more data points

mkdir -p exports/prometheus/range
for METRIC in "${LOCUST_METRICS[@]}"; do
  curl -s "${PROM}/api/v1/query_range" \
    --data-urlencode "query=${METRIC}" \
    --data-urlencode "start=${START}" \
    --data-urlencode "end=${END}" \
    --data-urlencode "step=${STEP}" \
    | python3 -m json.tool > "exports/prometheus/range/${METRIC}_range.json"
  echo "  → exports/prometheus/range/${METRIC}_range.json"
done

# 4. HTTP duration histogram buckets (full label set — matches what Grafana shows)
curl -s "${PROM}/api/v1/query?query=http_request_duration_ms_bucket" \
  | python3 -m json.tool > exports/prometheus/http_duration_buckets.json
echo "  → exports/prometheus/http_duration_buckets.json"

echo "Done."