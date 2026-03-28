FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY metric_features.py log_anomaly.py metric_lstm.py trace_lookup.py \
     models.py trace_summary.py main.py .

CMD ["python", "-u", "main.py"]
