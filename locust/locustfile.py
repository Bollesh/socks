"""
Sock Shop - Complex Load Test
Covers: catalogue, carts, orders, user, payment, shipping, queue-master
User flows: browse → register/login → add to cart → checkout → view orders
"""
import json
import logging
import threading
import time
import uuid
import random

import requests as req_lib
from locust import HttpUser, task, between, events, tag

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# ── OTel setup ───────────────────────────────────────────────────────────────
OTEL_ENDPOINT = "http://otel:4318"
LOKI_ENDPOINT = "http://otel:3100/loki/api/v1/push"
RESOURCE = Resource({"service.name": "locust"})

trace_provider = TracerProvider(resource=RESOURCE)
trace_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTEL_ENDPOINT}/v1/traces"))
)
trace.set_tracer_provider(trace_provider)
RequestsInstrumentor().instrument()

# ── Loki logger ───────────────────────────────────────────────────────────────
_log_buffer = []
_log_lock = threading.Lock()


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
    def debug(self, msg, **kw): _push_to_loki("debug", msg, kw)
    def info(self, msg, **kw):  _push_to_loki("info",  msg, kw)
    def warning(self, msg, **kw): _push_to_loki("warning", msg, kw)
    def error(self, msg, **kw): _push_to_loki("error", msg, kw)


logger = LokiLogger()
logging.basicConfig(level=logging.INFO)
console = logging.getLogger("locust.sockshop")

# ── Known catalogue IDs (fetched at startup if possible) ──────────────────────
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
    if exception:
        console.error("FAIL %s %s — %s", request_type, name, exception)
        logger.error(f"Request failed | {request_type} {name}",
                     method=request_type, url=name, error=str(exception))
    else:
        status = response.status_code
        level = "warning" if status >= 400 else "info"
        console.info("%s %s %s %.0fms", request_type, name, status, response_time)
        _push_to_loki(level,
                      f"Request completed | {request_type} {name} {status}",
                      dict(method=request_type, url=name,
                           status=str(status), duration_ms=str(round(response_time, 2))))


# ── User behaviour ────────────────────────────────────────────────────────────
class SockShopUser(HttpUser):
    """
    Simulates a realistic shopping journey:
      1. Browse catalogue (anonymous)
      2. Register or login
      3. Add items to cart, update quantities, remove items
      4. View & update address / card on profile
      5. Place order → triggers payment + shipping + queue-master
      6. Review order history
      7. Logout
    """
    wait_time = between(1, 4)

    def on_start(self):
        """Called once per simulated user. Register and log in."""
        self.user_id = None
        self.session_cookie = None
        self._register_and_login()

    # ── Catalogue (hits catalogue service) ───────────────────────────────────

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

    # ── Cart (hits carts service) ─────────────────────────────────────────────

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
            if resp.status_code not in (200, 201):
                logger.warning("Add to cart failed", item_id=item_id, status=resp.status_code)
                resp.failure(f"add to cart returned {resp.status_code}")

    @tag("cart")
    @task(3)
    def view_cart(self):
        logger.debug("Viewing cart")
        self.client.get("/cart", name="/cart [view]")

    @tag("cart")
    @task(2)
    def update_cart_item(self):
        """View cart then update quantity of a random item."""
        with self.client.get("/cart", name="/cart [view]", catch_response=True) as resp:
            if resp.status_code == 200:
                try:
                    items = resp.json()
                    if items:
                        item = random.choice(items)
                        new_qty = random.randint(1, 5)
                        self.client.post(
                            "/cart/update",
                            json={"id": item.get("itemId", item.get("id")), "quantity": new_qty},
                            name="/cart/update",
                        )
                        logger.info("Updated cart item quantity", quantity=new_qty)
                except Exception:
                    pass
                resp.success()

    @tag("cart")
    @task(1)
    def delete_cart_item(self):
        """Add then remove an item — exercises cart delete endpoint."""
        item_id = random.choice(CATALOGUE_IDS)
        self.client.post("/cart", json={"id": item_id}, name="/cart [add]")
        with self.client.get("/cart", name="/cart [view]", catch_response=True) as resp:
            if resp.status_code == 200:
                try:
                    items = resp.json()
                    if items:
                        item = random.choice(items)
                        iid = item.get("itemId", item.get("id"))
                        self.client.delete(f"/cart/{iid}", name="/cart/{id} [delete]")
                        logger.info("Deleted cart item", item_id=iid)
                except Exception:
                    pass
                resp.success()

    # ── User / profile (hits user service) ───────────────────────────────────

    @tag("user")
    @task(2)
    def view_profile(self):
        logger.debug("Viewing profile")
        self.client.get("/profile", name="/profile")

    @tag("user")
    @task(1)
    def view_addresses(self):
        if self.user_id:
            self.client.get(f"/addresses", name="/addresses")

    @tag("user")
    @task(1)
    def view_cards(self):
        if self.user_id:
            self.client.get(f"/cards", name="/cards")

    @tag("user")
    @task(1)
    def add_address(self):
        """Add a new shipping address — exercises user service write path."""
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
        """Add a payment card — exercises user service write path."""
        card = {
            "longNum":  f"{random.randint(1000,9999)}{random.randint(1000,9999)}"
                        f"{random.randint(1000,9999)}{random.randint(1000,9999)}",
            "expires":  f"{random.randint(1,12):02d}/{random.randint(25,30)}",
            "ccv":      f"{random.randint(100,999)}",
        }
        logger.info("Adding card")
        self.client.post("/cards", json=card, name="/cards [add]")

    # ── Orders (hits orders + payment + shipping + queue-master) ─────────────

    @tag("orders")
    @task(2)
    def view_orders(self):
        logger.debug("Viewing order history")
        self.client.get("/orders", name="/orders")

    @tag("orders")
    @task(1)
    def place_order(self):
        """
        Full checkout flow:
          1. Add item to cart
          2. POST /orders → triggers payment service + shipping service + queue-master
          3. Poll order status
        """
        item_id = random.choice(CATALOGUE_IDS)
        logger.info("Starting checkout flow", item_id=item_id)

        # Add item to cart
        self.client.post("/cart", json={"id": item_id}, name="/cart [add]")

        # Place order (triggers payment + shipping internally)
        with self.client.post(
            "/orders",
            name="/orders [place]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 201):
                try:
                    order = resp.json()
                    order_id = order.get("id", "")
                    logger.info("Order placed", order_id=order_id, item_id=item_id)
                    # Poll order status
                    if order_id:
                        time.sleep(0.5)
                        self.client.get(f"/orders/{order_id}", name="/orders/{id}")
                except Exception:
                    pass
                resp.success()
            else:
                logger.warning("Order placement failed", status=resp.status_code)
                resp.failure(f"order returned {resp.status_code}")

    # ── Static / frontend ─────────────────────────────────────────────────────

    @tag("frontend")
    @task(3)
    def home(self):
        logger.debug("Fetching home page")
        self.client.get("/", name="/")

    @tag("frontend")
    @task(1)
    def load_static_assets(self):
        """Simulate browser loading JS/CSS."""
        for path in ["/topbar.css", "/style.css"]:
            self.client.get(path, name="/static")

    # ── Auth helpers ──────────────────────────────────────────────────────────

    def _register_and_login(self):
        """Register a fresh user, then log in to get a session."""
        suffix = uuid.uuid4().hex[:8]
        username = f"locust_{suffix}"
        password = "Password1"

        # Register
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
            logger.info("Registered new user", username=username)
        else:
            logger.warning("Registration failed", status=resp.status_code, username=username)

        # Login
        resp = self.client.get(
            "/login",
            auth=(username, password),
            name="/login",
        )
        if resp.status_code == 200:
            logger.info("Logged in", username=username)
        else:
            logger.warning("Login failed", status=resp.status_code, username=username)

    def on_stop(self):
        logger.info("User session ending")