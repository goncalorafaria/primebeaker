#!/usr/bin/env python3
"""Watch live load and request latency on the four registered inference replicas."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import math
import re
import sys
import time
import urllib.request


ENDPOINTS = {
    "a": "http://holmes-cs-aus-540.reviz.ai2.in:34695",
    "b": "http://holmes-cs-aus-538.reviz.ai2.in:57257",
    "c": "http://holmes-cs-aus-514.reviz.ai2.in:49313",
    "d": "http://holmes-cs-aus-541.reviz.ai2.in:50889",
}
SAMPLE = re.compile(r"^(\w+:[\w:]+)(?:\{([^}]*)\})?\s+([^\s]+)")
WANTED = {
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
    "sglang:generation_tokens_total",
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:e2e_request_latency_seconds_count",
}


def scrape(name, url, timeout):
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=timeout) as response:
            body = response.read().decode()
        values = {}
        for line in body.splitlines():
            match = SAMPLE.match(line)
            if not match:
                continue
            metric, labels, raw_value = match.groups()
            if metric not in WANTED:
                continue
            # Gauge values are replicated across tensor ranks; count them once.
            if metric in {
                "sglang:num_running_reqs",
                "sglang:num_queue_reqs",
                "sglang:token_usage",
            } and labels and 'tp_rank="0"' not in labels:
                continue
            value = float(raw_value)
            if math.isfinite(value):
                values[metric] = values.get(metric, 0.0) + value
        return name, {"time": time.monotonic(), "metrics": values}
    except Exception as exc:
        return name, {"error": f"{type(exc).__name__}: {exc}"}


def delta(sample, previous, metric):
    now = sample["metrics"].get(metric)
    before = previous.get("metrics", {}).get(metric)
    if now is None or before is None or now < before:
        return None
    return now - before


def fmt(value, precision=0):
    return "—" if value is None else f"{value:,.{precision}f}"


def render(samples, previous, clear):
    if clear:
        print("\033[2J\033[H", end="")
    print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + " | Inference replica load")
    print(f"{'SERVER':5} {'RUNNING':>9} {'QUEUED':>9} {'CACHE':>8} {'TOK/S':>10} {'MEAN LATENCY':>14} {'DONE':>8}")

    totals = {"running": 0.0, "queued": 0.0, "tokens": 0.0, "latency_sum": 0.0, "latency_count": 0.0, "done": 0.0}
    healthy = 0
    for name in ENDPOINTS:
        sample = samples.get(name, {"error": "no response"})
        if "error" in sample:
            print(f"{name:5} ERROR: {sample['error']}")
            continue
        healthy += 1
        metrics = sample["metrics"]
        prev = previous.get(name, {})
        elapsed = sample["time"] - prev.get("time", sample["time"])
        tok_delta = delta(sample, prev, "sglang:generation_tokens_total") if "metrics" in prev else None
        sum_delta = delta(sample, prev, "sglang:e2e_request_latency_seconds_sum") if "metrics" in prev else None
        count_delta = delta(sample, prev, "sglang:e2e_request_latency_seconds_count") if "metrics" in prev else None
        tok_rate = tok_delta / elapsed if tok_delta is not None and elapsed > 0 else None
        mean_latency = sum_delta / count_delta if sum_delta is not None and count_delta and count_delta > 0 else None
        running = metrics.get("sglang:num_running_reqs")
        queued = metrics.get("sglang:num_queue_reqs")
        cache = metrics.get("sglang:token_usage")
        completed = count_delta
        for key, value in (("running", running), ("queued", queued), ("tokens", tok_rate), ("done", completed)):
            if value is not None:
                totals[key] += value
        if sum_delta is not None and count_delta is not None and count_delta > 0:
            totals["latency_sum"] += sum_delta
            totals["latency_count"] += count_delta
        print(f"{name:5} {fmt(running):>9} {fmt(queued):>9} {fmt(cache * 100 if cache is not None else None, 1):>7}% {fmt(tok_rate):>10} {fmt(mean_latency, 2):>13}s {fmt(completed):>8}")

    overall_latency = totals["latency_sum"] / totals["latency_count"] if totals["latency_count"] else None
    print("-" * 76)
    print(f"TOTAL {fmt(totals['running'] if healthy else None):>9} {fmt(totals['queued'] if healthy else None):>9} {'':>8} {fmt(totals['tokens'] if healthy else None):>10} {fmt(overall_latency, 2):>13}s {fmt(totals['done'] if healthy else None):>8}")
    print(f"Endpoints responding: {healthy}/{len(ENDPOINTS)}. Latency, output rate, and DONE are measured between samples; — means no interval sample. Ctrl-C exits.\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=5, help="sampling interval in seconds (default: 5)")
    parser.add_argument("--timeout", type=float, default=5, help="per-endpoint HTTP timeout (default: 5)")
    parser.add_argument("--once", action="store_true", help="take two samples, print interval metrics, and exit")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        parser.error("--interval and --timeout must be positive")

    previous = {}
    iteration = 0
    try:
        with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as pool:
            while True:
                started = time.monotonic()
                samples = dict(pool.map(lambda item: scrape(item[0], item[1], args.timeout), ENDPOINTS.items()))
                render(samples, previous, sys.stdout.isatty())
                previous = samples
                iteration += 1
                if args.once and iteration >= 2:
                    break
                time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
