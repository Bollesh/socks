#!/bin/bash
set -e
mkdir -p exports/tempo_traces

echo "Exporting Loki logs..."
curl -s "http://localhost:3100/loki/api/v1/query_range" \
  --data-urlencode 'query={service_name="locust"}' \
  --data-urlencode "start=$(date -d '1 hour ago' +%s%N)" \
  --data-urlencode "end=$(date +%s%N)" \
  --data-urlencode "limit=5000" \
  | python3 -m json.tool > exports/loki_logs.json
echo "  → exports/loki_logs.json"

echo "Exporting Tempo traces..."
curl -s "http://localhost:3200/api/search?limit=1000" > exports/tempo_index.json
python3 -c "
import json, urllib.request
with open('exports/tempo_index.json') as f:
    data = json.load(f)
for t in data.get('traces', []):
    tid = t['traceID']
    urllib.request.urlretrieve(
        f'http://localhost:3200/api/traces/{tid}',
        f'exports/tempo_traces/{tid}.json'
    )
print(f'  → {len(data[\"traces\"])} traces saved to exports/tempo_traces/')
"

echo "Exporting Prometheus metrics..."
curl -s "http://localhost:9090/api/v1/label/__name__/values" \
  | python3 -m json.tool > exports/prometheus_metric_names.json
echo "  → exports/prometheus_metric_names.json"

echo "Done."
