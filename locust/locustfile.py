"""
Sock Shop - Complex Load Test
Telemetry:
  - Traces  → Tempo   (via OTLP)
  - Logs    → Loki    (via native push API)
  - Metrics → Prometheus (via OTLP → otel-lgtm collector)
"""
import json
import logging
import os
import threading
import time
import uuid
import random

import requests as req_lib
from locust import HttpUser, task, between, events, tag

# ── OpenTelemetry: Traces ─────────────────────────────────────────────────────
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# ── OpenTelemetry: Metrics ────────────────────────────────────────────────────
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

# ── Config ────────────────────────────────────────────────────────────────────
OTEL_ENDPOINT = "http://otel:4318"
LOKI_ENDPOINT = "http://otel:3100/loki/api/v1/push"
LOG_STREAM_URL = os.environ.get("LOG_STREAM_URL", "").strip()
RESOURCE = Resource({"service.name": "locust"})

# ── Trace provider ────────────────────────────────────────────────────────────
trace_provider = TracerProvider(resource=RESOURCE)
trace_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces"))
)
trace.set_tracer_provider(trace_provider)
RequestsInstrumentor().instrument()

# ── Metric provider ───────────────────────────────────────────────────────────
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=f"{OTEL_ENDPOINT}/v1/metrics"),
    export_interval_millis=5000,
)
meter_provider = MeterProvider(resource=RESOURCE, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)

meter = metrics.get_meter("locust.sockshop")

# Metric instruments
request_counter = meter.create_counter(
    "http_requests_total",
    description="Total HTTP requests made by locust",
)
error_counter = meter.create_counter(
    "http_errors_total",
    description="Total HTTP errors (status >= 400 or exception)",
)
duration_histogram = meter.create_histogram(
    "http_request_duration_ms",
    description="HTTP request duration in milliseconds",
    unit="ms",
)
active_users_gauge = meter.create_up_down_counter(
    "locust_active_users",
    description="Number of active simulated users",
)
orders_counter = meter.create_counter(
    "orders_placed_total",
    description="Total orders successfully placed",
)
cart_operations_counter = meter.create_counter(
    "cart_operations_total",
    description="Total cart operations (add/remove/update)",
)

# ── Loki logger ───────────────────────────────────────────────────────────────
_log_buffer = []
_log_lock = threading.Lock()


def _trace_id_hex() -> str:
    try:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return ""


def _push_log_stream(message: str, level: str, labels: dict | None = None):
    if not LOG_STREAM_URL:
        return
    lab = {k: str(v) for k, v in (labels or {}).items()}
    body: dict = {"message": message, "level": level, "labels": lab}
    tid = lab.get("trace_id")
    if tid:
        body["trace_id"] = tid

    def _post():
        try:
            req_lib.post(LOG_STREAM_URL, json=body, timeout=2)
        except Exception:
            pass

    threading.Thread(target=_post, daemon=True).start()


def _push_to_loki(level: str, message: str, labels: dict = None):
    ns = str(time.time_ns())
    stream = {"service_name": "locust", "level": level}
    if labels:
        stream.update({k: str(v) for k, v in labels.items()})
    with _log_lock:
        _log_buffer.append((stream, [ns, message]))
    threading.Thread(target=_flush_loki, daemon=True).start()


def _flush_loki():
    with _log_lock:
        if not _log_buffer:
            return
        items = list(_log_buffer)
        _log_buffer.clear()
    streams = {}
    for stream, entry in items:
        key = json.dumps(stream, sort_keys=True)
        if key not in streams:
            streams[key] = {"stream": stream, "values": []}
        streams[key]["values"].append(entry)
    try:
        req_lib.post(LOKI_ENDPOINT, json={"streams": list(streams.values())}, timeout=2)
    except Exception:
        pass


class LokiLogger:
    def debug(self, msg, **kw):   _push_to_loki("debug",   msg, kw)
    def info(self, msg, **kw):    _push_to_loki("info",    msg, kw)
    def warning(self, msg, **kw): _push_to_loki("warning", msg, kw)
    def error(self, msg, **kw):   _push_to_loki("error",   msg, kw)


logger = LokiLogger()
logging.basicConfig(level=logging.INFO)
console = logging.getLogger("locust.sockshop")
if LOG_STREAM_URL:
    console.info("Streaming request logs to %s (for log-stream / drain3)", LOG_STREAM_URL)
else:
    console.warning(
        "LOG_STREAM_URL is unset — log-stream will get no Locust HTTP events "
        "(expected in Docker: LOG_STREAM_URL=http://log-stream:8080/v1/logs)."
    )

# ── Known catalogue item IDs ──────────────────────────────────────────────────
CATALOGUE_IDS = [
    "03fef6ac-1896-4ce8-bd69-b798f85c6e0b",
    "3395a43e-2d88-40de-b95f-e00e1502085b",
    "510a0d7e-8e83-4193-b483-e27e09ddc34f",
    "522c3f13-8650-4d5a-a3ef-3f4be5d32e5a",
    "6d212ab8-4681-4dca-bcbf-42d16abfa20e",
    "808a2de1-1aaa-4c25-a9b9-6612e8f29a38",
    "819e1fbf-8b7e-4f6d-811f-693534916a8b",
    "d3588630-ad8e-49df-bbd7-3167f7efb246",
]


# ── Locust event hooks ────────────────────────────────────────────────────────
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response,
               context, exception, **kwargs):
    labels = {"url": name, "method": request_type}
    tid = _trace_id_hex()
    if tid:
        labels["trace_id"] = tid

    if exception:
        labels["status"] = "error"
        console.error("FAIL %s %s — %s", request_type, name, exception)
        logger.error(f"Request failed | {request_type} {name}",
                     method=request_type, url=name, error=str(exception))
        _push_log_stream(
            f"Request failed | {request_type} {name} — {exception}",
            "error",
            labels,
        )
        request_counter.add(1, {**labels})
        error_counter.add(1, {**labels})
        duration_histogram.record(response_time, {**labels})
    else:
        status = response.status_code
        labels["status"] = str(status)
        level = "warning" if status >= 400 else "info"
        console.info("%s %s %s %.0fms", request_type, name, status, response_time)
        payload_labels = dict(
            method=request_type,
            url=name,
            status=str(status),
            duration_ms=str(round(response_time, 2)),
        )
        if tid:
            payload_labels["trace_id"] = tid
        _push_to_loki(level,
                      f"Request completed | {request_type} {name} {status}",
                      payload_labels)
        _push_log_stream(
            f"Request completed | {request_type} {name} {status}",
            level,
            payload_labels,
        )
        request_counter.add(1, {**labels})
        duration_histogram.record(response_time, {**labels})
        if status >= 400:
            error_counter.add(1, {**labels})


@events.spawning_complete.add_listener
def on_spawning_complete(user_count, **kwargs):
    active_users_gauge.add(user_count)
    logger.info("Spawning complete", user_count=user_count)


@events.quitting.add_listener
def on_quitting(**kwargs):
    logger.info("Locust quitting, flushing telemetry")
    trace_provider.force_flush()
    meter_provider.force_flush()


# ── User behaviour ────────────────────────────────────────────────────────────
class SockShopUser(HttpUser):
    """
    Simulates a realistic shopping journey:
      1. Browse catalogue (anonymous)
      2. Register + login
      3. Add items to cart, update quantities, remove items
      4. View & update address / card on profile
      5. Place order → triggers payment + shipping + queue-master
      6. Review order history
      7. Session ends
    """
    wait_time = between(1, 4)

    def on_start(self):
        self.user_id = None
        active_users_gauge.add(1)
        self._register_and_login()

    def on_stop(self):
        active_users_gauge.add(-1)
        logger.info("User session ending")

    # ── Catalogue ─────────────────────────────────────────────────────────────

    @tag("catalogue")
    @task(5)
    def browse_catalogue(self):
        page = random.randint(1, 3)
        size = random.choice([4, 8, 12])
        tags = random.choice(["", "brown", "geek", "formal", "blue", "magic"])
        url = f"/catalogue?page={page}&size={size}"
        if tags:
            url += f"&tags={tags}"
        logger.info("Browsing catalogue", page=page, size=size, tags=tags)
        self.client.get(url, name="/catalogue")

    @tag("catalogue")
    @task(4)
    def view_catalogue_item(self):
        item_id = random.choice(CATALOGUE_IDS)
        logger.info("Viewing item", item_id=item_id)
        self.client.get(f"/catalogue/{item_id}", name="/catalogue/{id}")

    @tag("catalogue")
    @task(2)
    def browse_by_tag(self):
        tag_name = random.choice(["brown", "geek", "formal", "blue", "magic", "red"])
        logger.info("Browsing by tag", tag=tag_name)
        self.client.get(f"/catalogue?tags={tag_name}", name="/catalogue?tags=")

    @tag("catalogue")
    @task(1)
    def get_catalogue_size(self):
        self.client.get("/catalogue/size", name="/catalogue/size")

    # ── Cart ──────────────────────────────────────────────────────────────────

    @tag("cart")
    @task(4)
    def add_to_cart(self):
        item_id = random.choice(CATALOGUE_IDS)
        logger.info("Adding item to cart", item_id=item_id)
        with self.client.post(
            "/cart",
            json={"id": item_id},
            name="/cart [add]",
            catch_response=True,
        ) as resp:
            cart_operations_counter.add(1, {"operation": "add"})
            if resp.status_code not in (200, 201):
                logger.warning("Add to cart failed",
                               item_id=item_id, status=resp.status_code)
                resp.failure(f"add to cart returned {resp.status_code}")

    @tag("cart")
    @task(3)
    def view_cart(self):
        logger.debug("Viewing cart")
        self.client.get("/cart", name="/cart [view]")

    @tag("cart")
    @task(2)
    def update_cart_item(self):
        with self.client.get("/cart", name="/cart [view]",
                             catch_response=True) as resp:
            if resp.status_code == 200:
                try:
                    items = resp.json()
                    if items:
                        item = random.choice(items)
                        new_qty = random.randint(1, 5)
                        self.client.post(
                            "/cart/update",
                            json={"id": item.get("itemId", item.get("id")),
                                  "quantity": new_qty},
                            name="/cart/update",
                        )
                        cart_operations_counter.add(1, {"operation": "update"})
                        logger.info("Updated cart item", quantity=new_qty)
                except Exception:
                    pass
                resp.success()

    @tag("cart")
    @task(1)
    def delete_cart_item(self):
        item_id = random.choice(CATALOGUE_IDS)
        self.client.post("/cart", json={"id": item_id}, name="/cart [add]")
        with self.client.get("/cart", name="/cart [view]",
                             catch_response=True) as resp:
            if resp.status_code == 200:
                try:
                    items = resp.json()
                    if items:
                        item = random.choice(items)
                        iid = item.get("itemId", item.get("id"))
                        self.client.delete(f"/cart/{iid}",
                                           name="/cart/{id} [delete]")
                        cart_operations_counter.add(1, {"operation": "delete"})
                        logger.info("Deleted cart item", item_id=iid)
                except Exception:
                    pass
                resp.success()

    # ── User / profile ────────────────────────────────────────────────────────

    @tag("user")
    @task(2)
    def view_profile(self):
        logger.debug("Viewing profile")
        self.client.get("/profile", name="/profile")

    @tag("user")
    @task(1)
    def view_addresses(self):
        self.client.get("/addresses", name="/addresses")

    @tag("user")
    @task(1)
    def view_cards(self):
        self.client.get("/cards", name="/cards")

    @tag("user")
    @task(1)
    def add_address(self):
        addr = {
            "number":   str(random.randint(1, 999)),
            "street":   random.choice(["Main St", "Oak Ave", "Elm Rd", "Park Blvd"]),
            "city":     random.choice(["London", "New York", "Berlin", "Tokyo"]),
            "postcode": f"{random.randint(10000, 99999)}",
            "country":  random.choice(["UK", "US", "DE", "JP"]),
        }
        logger.info("Adding address", city=addr["city"])
        self.client.post("/addresses", json=addr, name="/addresses [add]")

    @tag("user")
    @task(1)
    def add_card(self):
        card = {
            "longNum": (f"{random.randint(1000,9999)}{random.randint(1000,9999)}"
                        f"{random.randint(1000,9999)}{random.randint(1000,9999)}"),
            "expires": f"{random.randint(1,12):02d}/{random.randint(25,30)}",
            "ccv":     f"{random.randint(100,999)}",
        }
        logger.info("Adding card")
        self.client.post("/cards", json=card, name="/cards [add]")

    # ── Orders ────────────────────────────────────────────────────────────────

    @tag("orders")
    @task(2)
    def view_orders(self):
        logger.debug("Viewing order history")
        self.client.get("/orders", name="/orders")

    @tag("orders")
    @task(1)
    def place_order(self):
        """Full checkout: add to cart → POST /orders → poll status.
        Triggers: orders → payment → shipping → queue-master → rabbitmq."""
        item_id = random.choice(CATALOGUE_IDS)
        logger.info("Starting checkout", item_id=item_id)

        self.client.post("/cart", json={"id": item_id}, name="/cart [add]")

        with self.client.post("/orders", name="/orders [place]",
                              catch_response=True) as resp:
            if resp.status_code in (200, 201):
                try:
                    order = resp.json()
                    order_id = order.get("id", "")
                    orders_counter.add(1, {"status": "success"})
                    logger.info("Order placed", order_id=order_id, item_id=item_id)
                    if order_id:
                        time.sleep(0.5)
                        self.client.get(f"/orders/{order_id}",
                                        name="/orders/{id}")
                except Exception:
                    pass
                resp.success()
            else:
                orders_counter.add(1, {"status": "failed"})
                logger.warning("Order failed", status=resp.status_code)
                resp.failure(f"order returned {resp.status_code}")

    # ── Frontend ──────────────────────────────────────────────────────────────

    @tag("frontend")
    @task(3)
    def home(self):
        logger.debug("Fetching home page")
        self.client.get("/", name="/")

    @tag("frontend")
    @task(1)
    def load_static_assets(self):
        for path in ["/topbar.css", "/style.css"]:
            self.client.get(path, name="/static")

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _register_and_login(self):
        suffix = uuid.uuid4().hex[:8]
        username = f"locust_{suffix}"
        password = "Password1"

        resp = self.client.post(
            "/register",
            json={"username": username, "password": password,
                  "email": f"{username}@locust.test",
                  "firstName": "Load", "lastName": "Tester"},
            name="/register",
        )
        if resp.status_code in (200, 201):
            try:
                self.user_id = resp.json().get("id", "")
            except Exception:
                pass
            logger.info("Registered", username=username)
        else:
            logger.warning("Registration failed",
                           status=resp.status_code, username=username)

        resp = self.client.get("/login", auth=(username, password), name="/login")
        if resp.status_code == 200:
            logger.info("Logged in", username=username)
        else:
            logger.warning("Login failed",
                           status=resp.status_code, username=username)