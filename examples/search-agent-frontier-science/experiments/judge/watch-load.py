#!/usr/bin/env python3
"""Live Pengin vLLM load, discovered through LiteRegistry's Redis head resolver."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import math
import re
import sys
import time
import urllib.request

DEFAULT_REGISTRY = 'head+sqlite:///weka/gfaria/search-agent-repr/registry/control.sqlite3'
DEFAULT_MODEL = '/weka/gfaria/launches/pengin-9b-rl-benchmarks-100step-20261025/9b/step_400/hf-checkpoint'
METRIC = re.compile(r'^(vllm:\w+)(?:\{.*\})?\s+([^\s]+)')


def scrape(url):
    try:
        with urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=4) as response:
            payload = response.read().decode()
        metrics = {}
        for line in payload.splitlines():
            match = METRIC.match(line)
            if not match:
                continue
            name, value = match.groups()
            if name.endswith('_bucket'):
                continue
            value = float(value)
            if math.isfinite(value):
                metrics[name] = metrics.get(name, 0.0) + value
        return {'time': time.monotonic(), 'metrics': metrics}
    except Exception as exc:
        return {'error': str(exc)}


def metric(sample, name):
    return sample.get('metrics', {}).get('vllm:' + name)


def delta(now, before, name):
    a, b = metric(now, name), metric(before, name)
    if a is None or b is None or a < b:
        return None
    return a - b


def fmt(value, precision=0):
    return '—' if value is None else f'{value:,.{precision}f}'


def display(current, previous, discovery_error, clear):
    rows = []
    totals = {'active': 0, 'queued': 0, 'tok_s': 0, 'lat_sum': 0, 'lat_count': 0, 'req_s': 0}
    coverage = dict.fromkeys(('active', 'queued', 'rate', 'requests'), 0)
    kv_values = []
    for url, sample in sorted(current.items()):
        if 'error' in sample:
            rows.append(f'{url.removeprefix("http://"):51} ERROR: {sample["error"]}')
            continue
        active = metric(sample, 'num_requests_running')
        queued = metric(sample, 'num_requests_waiting')
        kv = metric(sample, 'kv_cache_usage_perc')
        rate = latency = req_rate = None
        before = previous.get(url, {})
        if 'time' in before:
            tokens = delta(sample, before, 'generation_tokens_total')
            elapsed = sample['time'] - before['time']
            if tokens is not None and elapsed > 0:
                rate = tokens / elapsed
                totals['tok_s'] += rate
                coverage['rate'] += 1
            requests = delta(sample, before, 'request_success_total')
            if requests is not None and elapsed > 0:
                req_rate = requests / elapsed
                totals['req_s'] += req_rate
                coverage['requests'] += 1
            duration = delta(sample, before, 'e2e_request_latency_seconds_sum')
            count = delta(sample, before, 'e2e_request_latency_seconds_count')
            if duration is not None and count is not None and count > 0:
                latency = duration / count
                totals['lat_sum'] += duration
                totals['lat_count'] += count
        for name, value in [('active', active), ('queued', queued)]:
            if value is not None:
                totals[name] += value
                coverage[name] += 1
        if kv is not None:
            kv_values.append(kv * 100)
        rows.append(f'{url.removeprefix("http://"):51} {fmt(active):>7} {fmt(queued):>7} {fmt(req_rate, 2):>9} {fmt(rate):>10} {fmt(latency, 2):>10} {fmt(kv * 100 if kv is not None else None, 1):>8}')
    if clear:
        print('\033[2J\033[H', end='')
    print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') + '  |  Pengin judge load')
    healthy = sum('error' not in x for x in current.values())
    print(f'Endpoints responding: {healthy}/{len(current)} | rate samples: {coverage["rate"]}/{len(current)}')
    avg_latency = totals['lat_sum'] / totals['lat_count'] if totals['lat_count'] else None
    print(f'ACTIVE {fmt(totals["active"] if coverage["active"] else None)}  |  QUEUED {fmt(totals["queued"] if coverage["queued"] else None)}  |  OUTPUT {fmt(totals["tok_s"] if coverage["rate"] else None)} tok/s  |  AVG LATENCY {fmt(avg_latency, 2)} s')
    print(f'COMPLETED MODEL REQ/S {fmt(totals["req_s"] if coverage["requests"] else None, 2)} | request-rate coverage {coverage["requests"]}/{len(current)} | gauge coverage {coverage["active"]}/{len(current)}')
    print(f'KV CACHE: mean {fmt(sum(kv_values) / len(kv_values) if kv_values else None, 1)}% / max {fmt(max(kv_values) if kv_values else None, 1)}% ({len(kv_values)} engines)')
    if discovery_error:
        print('Discovery warning (retaining previous endpoints):', discovery_error)
    print(f'{"ENGINE":51} {"ACTIVE":>7} {"QUEUED":>7} {"REQ/S":>9} {"TOK/S":>10} {"LATENCY s":>10} {"KV %":>8}')
    print('\n'.join(rows) if rows else 'No active judge endpoints registered.')
    print('REQ/S = successful model completions, not rubric episodes. ACTIVE/QUEUED = vLLM only, not JudgeAPI queues. Partial coverage sums healthy samples only. — = no sample. Ctrl-C exits.\n', flush=True)


async def main(args):
    # Use the same deployed LiteRegistry package as the service stack.
    sys.path.insert(0, '/weka/gfaria/search-agent-repr/services/runtime')
    from literegistry import get_kvstore
    from literegistry.client import RegistryClient
    logging.getLogger().setLevel(logging.WARNING)
    client = RegistryClient(get_kvstore(args.registry))
    endpoints, previous = [], {}
    discovered = float('-inf')
    discovery_error = None
    iteration = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        while True:
            started = time.monotonic()
            if started - discovered >= 30:
                try:
                    roster = await asyncio.wait_for(client.roster(), timeout=10)
                    endpoints = sorted({x['uri'] for x in roster['servers'] if x.get('status') == 'active' and x.get('metadata', {}).get('model_path') == args.model})
                    discovery_error = None
                except Exception as exc:
                    discovery_error = str(exc) or type(exc).__name__
                discovered = started
            results = list(pool.map(scrape, endpoints))
            current = dict(zip(endpoints, results))
            if not args.once or iteration > 0:
                display(current, previous, discovery_error, sys.stdout.isatty() and not args.once)
            previous = current
            iteration += 1
            if args.once and iteration >= 2:
                return
            await asyncio.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', default=DEFAULT_REGISTRY)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--interval', type=float, default=5, help='Refresh seconds (default: 5)')
    parser.add_argument('--once', action='store_true', help='Take two samples, print rates, and exit')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
