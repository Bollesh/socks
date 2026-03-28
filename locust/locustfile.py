import logging
import time
import threading
import requests as req_lib
import json

from locust import HttpUser, task, between, events

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

OTEL_ENDPOINT = "http://otel:4318"
LOKI_ENDPOINT = "http://otel:3100/loki/api/v1/push"
RESOURCE = Resource({"service.name": "locust"})

# ── Traces ───────────────────────────────────────────────────────────────────
trace_provider = TracerProvider(resource=RESOURCE)
trace_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces"))
)
trace.set_tracer_provider(trace_provider)
RequestsInstrumentor().instrument()

# ── Loki log buffer ──────────────────────────────────────────────────────────
_log_buffer = []
_log_lock = threading.Lock()


def _push_to_loki(level: str, message: str, labels: dict = None):
    """Buffer a log entry and push to Loki."""
    ns = str(time.time_ns())
    stream = {"service_name": "locust", "level": level}
    if labels:
        stream.update(labels)
    entry = [ns, message]
    with _log_lock:
        _log_buffer.append((stream, entry))
    # Push immediately in a background thread to avoid blocking locust
    threading.Thread(target=_flush_loki, daemon=True).start()


def _flush_loki():
    with _log_lock:
        if not _log_buffer:
            return
        items = list(_log_buffer)
        _log_buffer.clear()

    # Group by stream labels
    streams = {}
    for stream, entry in items:
        key = json.dumps(stream, sort_keys=True)
        if key not in streams:
            streams[key] = {"stream": stream, "values": []}
        streams[key]["values"].append(entry)

    payload = {"streams": list(streams.values())}
    try:
        req_lib.post(
            LOKI_ENDPOINT,
            json=payload,
            timeout=2,
        )
    except Exception:
        pass  # Never let log failures crash locust


# ── Simple logger wrapper ────────────────────────────────────────────────────
class LokiLogger:
    def debug(self, msg, **kwargs):
        _push_to_loki("debug", msg, kwargs)

    def info(self, msg, **kwargs):
        _push_to_loki("info", msg, kwargs)

    def warning(self, msg, **kwargs):
        _push_to_loki("warning", msg, kwargs)

    def error(self, msg, **kwargs):
        _push_to_loki("error", msg, kwargs)


logger = LokiLogger()

# Also keep standard logging for locust console output
logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("locust.sockshop")


# ── Request event hook ───────────────────────────────────────────────────────
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response,
               context, exception, **kwargs):
    if exception:
        msg = f"Request failed | method={request_type} url={name} error={exception}"
        console_logger.error(msg)
        logger.error(msg, url=name, method=request_type)
    else:
        msg = (f"Request completed | method={request_type} url={name} "
               f"status={response.status_code} duration_ms={round(response_time, 2)}")
        console_logger.info(msg)
        logger.info(msg, url=name, method=request_type,
                    status=str(response.status_code),
                    duration_ms=str(round(response_time, 2)))


class SockShopUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def home(self):
        logger.debug("Fetching home page", endpoint="/")
        self.client.get("/")

    @task(2)
    def catalogue(self):
        logger.info("Fetching catalogue", endpoint="/catalogue")
        self.client.get("/catalogue")

    @task(1)
    def item(self):
        logger.info("Fetching catalogue item", endpoint="/catalogue/1", item_id="1")
        self.client.get("/catalogue/1")