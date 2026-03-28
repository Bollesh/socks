"""Summarize Grafana Tempo trace JSON for LLM context."""
from __future__ import annotations

from typing import Any


def _attrs(kv_list: list[dict] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not kv_list:
        return out
    for it in kv_list:
        k = it.get("key", "")
        v = it.get("value", {})
        if "stringValue" in v:
            out[k] = str(v["stringValue"])
        elif "intValue" in v:
            out[k] = str(v["intValue"])
        elif "boolValue" in v:
            out[k] = str(v["boolValue"])
    return out


def summarize_trace_json(data: dict[str, Any] | None, max_spans: int = 80) -> str:
    if not data:
        return "(no trace payload)"

    batches = data.get("batches")
    if not batches and data.get("resourceSpans"):
        batches = data["resourceSpans"] if isinstance(data["resourceSpans"], list) else []
    batches = batches or []

    spans_total = 0
    services: set[str] = set()
    rows: list[str] = []
    errors = 0

    def walk_resource_block(resource: dict, scope_spans: list):
        nonlocal spans_total, errors
        res_attr = _attrs(resource.get("attributes"))
        svc = res_attr.get("service.name", "?")
        if svc:
            services.add(svc)
        for ss in scope_spans or []:
            for sp in ss.get("spans") or []:
                if spans_total >= max_spans:
                    return
                spans_total += 1
                name = sp.get("name", "")
                kind = sp.get("kind", "")
                st = int(sp.get("startTimeUnixNano", 0))
                et = int(sp.get("endTimeUnixNano", 0))
                dur_ns = max(et - st, 0)
                attr = _attrs(sp.get("attributes"))
                st_code = attr.get("otel.status_code", "")
                if st_code == "ERROR" or sp.get("status", {}).get("code") == 2:
                    errors += 1
                rows.append(
                    f"- {svc} | {name} | {dur_ns // 1_000_000}ms | {kind} | {st_code or 'OK'}"
                )

    for batch in batches:
        if "resourceSpans" in batch:
            for block in batch.get("resourceSpans") or []:
                walk_resource_block(block.get("resource") or {}, block.get("scopeSpans") or [])
        else:
            walk_resource_block(batch.get("resource") or {}, batch.get("scopeSpans") or [])

    head = (
        f"services={sorted(services)} spans_sampled={spans_total} "
        f"span_errors={errors}\n"
    )
    return head + "\n".join(rows[:max_spans])
