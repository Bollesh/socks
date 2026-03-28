FROM python:3.11-slim

WORKDIR /app

RUN pip install \
    locust \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation-requests

COPY locustfile.py .

CMD ["locust", "-f", "locustfile.py", "--host=http://front-end:8079"]