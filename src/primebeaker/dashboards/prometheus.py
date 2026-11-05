"""vLLM counters, engine gauges and interval latencies; no generation calls."""
from __future__ import annotations

import math
import re
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit

from .store import utc_now


def read_url(url: str, timeout: float, headers: dict | None = None) -> str:
    with urlopen(Request(url, headers=headers or {}), timeout=timeout) as response:
        return response.read(8 * 1024 * 1024).decode()


def parse(text: str) -> tuple[dict, dict]:
    totals, engines = {}, {}
    for line in text.splitlines():
        match = re.match(r"(vllm:[\w:]+)(\{.*\})?\s+([-+\w.]+)(?:\s+\S+)?$", line)
        if not match:
            continue
        try:
            value = float(match[3])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        name, labels = match[1].split(":", 1)[1], match[2] or ""
        totals[name] = totals.get(name, 0) + value
        engine = re.search(r'(?:engine|engine_index)="([^"]+)"', labels)
        if engine and name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "gpu_cache_usage_perc"):
            engines.setdefault(engine[1], {})[name] = value
    return totals, engines


class VLLMMetrics:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.previous = {}

    def scrape(self, endpoint: str, headers: dict | None = None) -> dict:
        endpoint = endpoint.rstrip("/").removesuffix("/v1")
        sampled = utc_now()
        try:
            parsed = urlsplit(endpoint)
            display = urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                endpoint = display
                raise ValueError("Use HTTP(S) endpoints without credentials/query strings; configure headers through environment variables")
            totals, engines = parse(read_url(endpoint + "/metrics", self.timeout, headers))
            if "num_requests_running" not in totals or "num_requests_waiting" not in totals:
                raise ValueError("vLLM request gauges are not exposed")
            stamp = time.monotonic()
            old = self.previous.get(endpoint)
            self.previous[endpoint] = (stamp, totals)

            def delta(key):
                if old is None or key not in old[1] or key not in totals:
                    return None
                value = totals[key] - old[1][key]
                return value if value >= 0 else None

            def rate(key):
                value = delta(key)
                return value / (stamp - old[0]) if value is not None else None

            def mean(key):
                count, total = delta(key + "_count"), delta(key + "_sum")
                return total / count if count and total is not None else None

            cache_key = "kv_cache_usage_perc" if "kv_cache_usage_perc" in totals else "gpu_cache_usage_perc"
            caches = [e[cache_key] for e in engines.values() if cache_key in e]
            # Some vLLM versions do not label engine gauges. A single unlabeled
            # cache gauge can still be shown without inventing an engine count.
            if not caches and cache_key in totals:
                caches = [totals[cache_key]]
            return {"endpoint": endpoint, "available": True, "sampled_at": sampled, "active": totals["num_requests_running"], "queued": totals["num_requests_waiting"], "engines": engines,
                    "prompt_tokens_per_second": rate("prompt_tokens_total"), "output_tokens_per_second": rate("generation_tokens_total"), "completed_requests_per_second": rate("request_success_total"),
                    "completed_requests_total": totals.get("request_success_total"), "request_latency_seconds_interval_mean": mean("e2e_request_latency_seconds"),
                    "queue_seconds_interval_mean": mean("request_queue_time_seconds"), "first_token_seconds_interval_mean": mean("time_to_first_token_seconds"),
                    "kv_cache_percent_mean": 100 * sum(caches) / len(caches) if caches else None, "kv_cache_percent_max": 100 * max(caches) if caches else None}
        except Exception as exc:
            self.previous.pop(endpoint, None)
            return {"endpoint": endpoint, "available": False, "sampled_at": sampled, "active": None, "queued": None, "error": f"Metrics unavailable ({type(exc).__name__})"}


def discover(registry: str, model: str, timeout: float) -> list[str]:
    import asyncio
    from literegistry import get_kvstore
    from literegistry.client import RegistryClient

    async def fetch():
        client = RegistryClient(get_kvstore(registry))
        try:
            return await asyncio.wait_for(client.get_all(model, force=True), timeout=timeout)
        finally:
            await client.close()
    return list(dict.fromkeys(asyncio.run(fetch())))
