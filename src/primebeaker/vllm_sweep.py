"""Repeatable, isolated vLLM serving throughput experiments (vLLM 0.22 CLI)."""
from __future__ import annotations

import csv
import io
import tempfile
import threading
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request


def atomic_text(path, content):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dump(path, value):
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item
    atomic_text(path, json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def load(config):
    c = json.loads(Path(config).read_text())
    allowed = {'model', 'server', 'grid', 'concurrency', 'workloads', 'requests', 'warmup_requests',
               'repeats', 'seed', 'startup_timeout', 'benchmark_timeout', 'latency_limits_ms', 'sample_settings', 'measurement_seconds'}
    if set(c) - allowed:
        raise ValueError(f'Unknown config keys: {sorted(set(c)-allowed)}')
    if not isinstance(c.get('model'), str) or not c['model']:
        raise ValueError('model is required')
    c = {'server': {}, 'grid': {}, 'requests': 128, 'warmup_requests': 8, 'repeats': 2,
         'seed': 42, 'startup_timeout': 1800, 'benchmark_timeout': 1800, 'latency_limits_ms': {}, **c}
    for key in ('requests', 'warmup_requests', 'repeats', 'startup_timeout', 'benchmark_timeout'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not c.get('concurrency') or any(type(n) is not int or n < 1 for n in c['concurrency']):
        raise ValueError('concurrency must contain positive integers')
    if max(c['concurrency']) > c['requests']:
        raise ValueError('requests must be at least the largest concurrency')
    reserved = {'model', 'host', 'port', 'served_model_name', 'api_key', 'config', 'uds'}
    for key in set(c['server']) | set(c['grid']):
        if key in reserved or not key.replace('_', '').isalnum():
            raise ValueError(f'Reserved or invalid engine parameter: {key}')
    for values in c['grid'].values():
        if not isinstance(values, list) or not values:
            raise ValueError('grid dimensions must be nonempty lists')
    if 'sample_settings' in c and (type(c['sample_settings']) is not int or c['sample_settings'] < 1):
        raise ValueError('sample_settings must be a positive integer')
    if 'measurement_seconds' in c and (type(c['measurement_seconds']) is not int or c['measurement_seconds'] < 1):
        raise ValueError('measurement_seconds must be a positive integer')
    names = set()
    if not c.get('workloads'):
        raise ValueError('workloads is required')
    for w in c['workloads']:
        if set(w) - {'name', 'input_tokens', 'output_tokens', 'prefix_tokens', 'replay_manifest'}:
            raise ValueError('Unknown workload field')
        if not w.get('name') or w['name'] in names:
            raise ValueError('Workload names must be nonempty and unique')
        names.add(w['name'])
        if w.get('replay_manifest'):
            continue
        for key in ('input_tokens', 'output_tokens'):
            if type(w.get(key)) is not int or w[key] < 1:
                raise ValueError(f'{key} must be positive')
        if type(w.get('prefix_tokens', 0)) is not int or w.get('prefix_tokens', 0) < 0:
            raise ValueError('prefix_tokens must be nonnegative')
    for key, value in c['latency_limits_ms'].items():
        if key not in {'p99_ttft_ms', 'p99_tpot_ms', 'p99_e2el_ms'} or value <= 0:
            raise ValueError('Latency limits must be positive p99_*_ms values')
    return c


def trials(c):
    keys = sorted(c['grid'])
    result = []
    for values in itertools.product(*(c['grid'][key] for key in keys)):
        engine = {**c['server'], **dict(zip(keys, values))}
        for w, concurrency, repeat in itertools.product(c['workloads'], c['concurrency'], range(c['repeats'])):
            length = w.get('input_tokens', 0) + w.get('output_tokens', 0) + w.get('prefix_tokens', 0)
            if length > int(engine.get('max_model_len', 2**63)):
                raise ValueError(f"Workload {w['name']} exceeds max_model_len (including prefix and output)")
            t = {'engine': engine, 'workload': w, 'concurrency': concurrency, 'repeat': repeat,
                 'seed': c['seed'] + repeat}
            t['id'] = hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()[:16]
            result.append(t)
    if 'sample_settings' in c:
        # Sample engine + client settings together; retain every workload/repeat.
        identities = sorted({json.dumps([t['engine'], t['concurrency']], sort_keys=True) for t in result})
        chosen = set(random.Random(c['seed']).sample(identities, min(c['sample_settings'], len(identities))))
        result = [t for t in result if json.dumps([t['engine'], t['concurrency']], sort_keys=True) in chosen]
    random.Random(c['seed']).shuffle(result)
    return result


def flags(parameters):
    args = []
    for key, value in sorted(parameters.items()):
        if value is None:
            continue
        key = key.replace('_', '-')
        if isinstance(value, bool):
            args.append('--' + ('' if value else 'no-') + key)
        else:
            args.extend(['--' + key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)])
    return args


def commands(c, t, port, folder, warmup=False):
    w = t['workload']
    server = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', c['model'],
              '--host', '127.0.0.1', '--port', str(port), '--served-model-name', 'sweep-model', *flags(t['engine'])]
    if w.get('replay_manifest'):
        replay = Path(__file__).with_name('trace_replay.py')
        return server, [sys.executable, str(replay), '--manifest='+w['replay_manifest'],
            '--port='+str(port), '--output='+str(folder), '--concurrency='+str(t['concurrency']),
            '--seed='+str(t['seed']), '--warmup='+str(warmup)]
    bench = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'bench', 'serve', '--backend', 'openai',
             '--base-url', f'http://127.0.0.1:{port}', '--endpoint', '/v1/completions',
             '--model', 'sweep-model', '--tokenizer', c['model'], '--dataset-name', 'random',
             '--random-input-len', str(w['input_tokens']), '--random-output-len', str(w['output_tokens']),
             '--random-prefix-len', str(w.get('prefix_tokens', 0)), '--random-range-ratio', '0',
             '--seed', str(t['seed'] + (1000000 if warmup else 0)), '--request-rate', 'inf',
             '--max-concurrency', str(t['concurrency']), '--num-prompts', str(c['warmup_requests'] if warmup else c['requests']),
             '--ignore-eos', '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,95,99',
             '--save-result', '--save-detailed', '--result-dir', str(folder), '--result-filename', 'warmup.json' if warmup else 'benchmark.json']
    return server, bench


def stop(process):
    # The leader may have exited while a vLLM worker remains in its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(command, log, timeout):
    with Path(log).open('w') as stream:
        p = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'Command exited {code}; see {log}')
        finally:
            stop(p)


def http(port, path, method='GET'):
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}{path}', method=method), timeout=3) as r:
        return r.read()


def token_counters(payload):
    totals = {}
    for line in payload.decode().splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        name = parts[0].split('{', 1)[0]
        if name in {'vllm:generation_tokens_total', 'vllm:prompt_tokens_total'}:
            # Label values can contain spaces; metric value is at the end.
            value = float(parts[-1])
            if math.isfinite(value):
                totals[name] = totals.get(name, 0.0) + value
    return totals


def measure_compute(port, folder, command, seconds):
    """Persist engine counters while load runs; completion is not required."""
    samples = []
    finished = threading.Event()
    sample_lock = threading.Lock()

    def sample():
        with sample_lock:
            timestamp = time.monotonic()
            try:
                payload = http(port, '/metrics')
                record = {'monotonic_seconds': timestamp, 'unix_seconds': time.time(),
                          'counters': token_counters(payload)}
                atomic_text(folder/f'metrics-sample-{len(samples):05d}.txt', payload.decode())
            except Exception as error:
                record = {'monotonic_seconds': timestamp, 'error': str(error)}
            try:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
                record['gpu_csv'] = gpu.stdout.strip()
            except (OSError, subprocess.SubprocessError) as error:
                record['gpu_sampling_error'] = str(error)
            samples.append(record)
            dump(folder/'compute-samples.json', samples)

    def poll():
        while not finished.wait(2):
            sample()

    sample()
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        try:
            execute(command, folder/'benchmark.log', seconds)
        except subprocess.TimeoutExpired:
            # A bounded observation window is a normal completion condition.
            pass
    finally:
        finished.set()
        thread.join(timeout=5)
        sample()
        summary = summarize_compute(samples)
        dump(folder/'compute.json', summary)
    return summary


def summarize_compute(samples):
    good = [s for s in samples if s.get('counters')]
    result = {'sample_count': len(good), 'measurement_source': 'engine_token_counters'}
    if len(good) < 2:
        return result
    first, last = good[0], good[-1]
    elapsed = last['monotonic_seconds'] - first['monotonic_seconds']
    result['observed_seconds'] = elapsed
    if elapsed <= 0:
        return result
    for name, label in [('vllm:generation_tokens_total', 'output'), ('vllm:prompt_tokens_total', 'input')]:
        if all(name in s['counters'] for s in good):
            values = [s['counters'][name] for s in good]
            if any(b < a for a, b in zip(values, values[1:])):
                result['counter_reset'] = True
                continue
            delta = values[-1] - values[0]
            result[f'{label}_tokens'] = delta
            result[f'{label}_tokens_per_second'] = delta / elapsed
    return result


def eligible(raw, c):
    if raw.get('completed') != c['requests']:
        return False
    rate = raw.get('output_throughput')
    if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
        return False
    return all(isinstance(raw.get(key), (float, int)) and math.isfinite(raw[key]) and raw[key] <= limit
               for key, limit in c['latency_limits_ms'].items())


def check_interactive_slots(limit=2):
    """Count queued as well as running interactive allocations; fail closed."""
    import pwd
    user = pwd.getpwuid(os.getuid()).pw_name
    output = subprocess.check_output(
        ['squeue', '--noheader', '--user', user, '--format=%i|%P'], text=True, timeout=30)
    jobs = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if len(fields) != 2:
            raise RuntimeError('Cannot parse squeue; refusing submission')
        if 'interactive' in fields[1].lower():
            jobs.add(fields[0])
    if len(jobs) >= limit:
        raise RuntimeError(f'Interactive submission limit {limit} reached (queued/running jobs: {sorted(jobs)}). Resume after a slot is released.')
    return sorted(jobs)


class SweepCLI:
    def prepare_replay(self, traces, tools, output, conversations=32, turns=4, seed=42, runtime_traces=None):
        from primebeaker.trace_replay import prepare
        return prepare(traces, tools, output, conversations, turns, seed, runtime_traces)

    def plan(self, config):
        c = load(config)
        ts = trials(c)
        return {'trials': len(ts), 'model': c['model'], 'schedule': ts,
                'note': 'One fresh server per trial. Startup is excluded. Run only on allocated GPUs.'}

    def run(self, config, output, max_trials=None, resume=False):
        c = load(config)
        ts = trials(c)
        root = Path(output).resolve()
        if resume:
            if load(root/'config.json') != c:
                raise ValueError('Resume requires the same configuration')
            pending = []
            for t in ts:
                folder = root/t['id']
                result = folder/'result.json'
                if result.exists() and json.loads(result.read_text()).get('status') in {'completed', 'failed', 'measured'}:
                    continue
                if folder.exists():
                    archive = root/'interrupted'
                    archive.mkdir(exist_ok=True)
                    folder.rename(archive/f"{t['id']}-{time.time_ns()}")
                pending.append(t)
            ts = pending
        else:
            root.mkdir(parents=True, exist_ok=False)
            dump(root/'config.json', c)
            dump(root/'plan.json', ts)
        if max_trials is not None:
            if int(max_trials) < 1:
                raise ValueError('max_trials must be positive')
            ts = ts[:int(max_trials)]
        session_id = str(time.time_ns())
        dump(root/f'session-{session_id}.json', {'pending_trials': [t['id'] for t in ts],
             'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        from importlib.metadata import version
        dump(root/'environment.json', {'vllm': version('vllm'), 'torch': version('torch'),
             'python': sys.version, 'host': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
             'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        dump(root/f'environment-{session_id}.json', json.loads((root/'environment.json').read_text()))
        try:
            execute(['nvidia-smi', '-q'], root/'gpu.txt', 20)
        except Exception as error:
            (root/'gpu-error.txt').write_text(str(error))
        previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            for index, t in enumerate(ts):
                folder = root/t['id']
                folder.mkdir()
                dump(folder/'trial.json', {**t, 'session_id': session_id})
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                server, bench = commands(c, t, port, folder)
                dump(folder/'commands.json', {'server': server, 'benchmark': bench, 'warmup': commands(c, t, port, folder, True)[1], 'server_env': {'VLLM_SERVER_DEV_MODE': '1'}})
                row = {**t, 'status': 'failed', 'eligible': False, 'session_id': session_id,
                       'started_at_unix': time.time()}
                abort_sweep = False
                started = time.monotonic()
                print(f"[{index+1}/{len(ts)}] {t['id']} {t['engine']} concurrency={t['concurrency']}", flush=True)
                with (folder/'server.log').open('w') as stream:
                    # Cache reset is a dev endpoint; this server binds to loopback only.
                    server_env = {**os.environ, 'VLLM_SERVER_DEV_MODE': '1'}
                    p = subprocess.Popen(server, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=server_env)
                    try:
                        deadline = time.monotonic() + c['startup_timeout']
                        while True:
                            if p.poll() is not None:
                                raise RuntimeError(f'Engine exited {p.returncode} during startup; see server.log')
                            try:
                                models = json.loads(http(port, '/v1/models'))
                                if any(m['id'] == 'sweep-model' for m in models['data']):
                                    break
                            except (OSError, ValueError, KeyError):
                                pass
                            if time.monotonic() > deadline:
                                raise TimeoutError('Engine startup timeout')
                            time.sleep(1)
                        execute(commands(c, t, port, folder, True)[1], folder/'warmup.log', c['benchmark_timeout'])
                        warmup = json.loads((folder/'warmup.json').read_text())
                        if warmup.get('completed', 0) < 1:
                            replay_log = folder/'replay-warmup.jsonl'
                            if replay_log.exists():
                                failures = [json.loads(line) for line in replay_log.read_text().splitlines()]
                                abort_sweep = any(r.get('http_status') in {400, 401, 403, 404, 422} for r in failures)
                            raise RuntimeError('Warm-up requests failed; see replay-warmup.jsonl or warmup.log')
                        # Fresh server per trial plus distinct warm-up seeds prevent cache leakage.
                        # Keep any intended within-trial shared-prefix reuse in the measured workload.
                        http(port, '/reset_prefix_cache', method='POST')
                        (folder/'metrics-before.txt').write_bytes(http(port, '/metrics'))
                        compute = measure_compute(port, folder, bench, c.get('measurement_seconds', c['benchmark_timeout']))
                        row.update(status='measured', compute=compute)
                        if (folder/'benchmark.json').exists():
                            raw = json.loads((folder/'benchmark.json').read_text())
                            row.update(status='completed', eligible=eligible(raw, c), metrics=raw)
                        (folder/'metrics-after.txt').write_bytes(http(port, '/metrics'))
                    except KeyboardInterrupt:
                        row.update(status='interrupted')
                        raise
                    except Exception as error:
                        row.update(status='failed', eligible=False, error=f'{type(error).__name__}: {error}')
                    finally:
                        if (folder/'compute.json').exists():
                            row['compute'] = json.loads((folder/'compute.json').read_text())
                        stop(p)
                        row.update(finished_at_unix=time.time(), elapsed_seconds=time.monotonic()-started)
                        dump(folder/'result.json', row)
                self.report(str(root))
                if abort_sweep:
                    raise RuntimeError('Stopping sweep: replay request validation/authentication failed; see saved response body')
        finally:
            signal.signal(signal.SIGTERM, previous)
        return self.report(str(root))

    def report(self, output):
        root = Path(output)
        c = json.loads((root/'config.json').read_text())
        rows = [json.loads(p.read_text()) for p in root.glob('*/result.json')]
        # Recover sampled work even if allocation termination prevented final JSON.
        existing = {r['id'] for r in rows}
        for path in root.glob('*/trial.json'):
            if path.parent.name not in existing and (path.parent/'compute-samples.json').exists():
                t = json.loads(path.read_text())
                rows.append({**t, 'status': 'interrupted', 'eligible': False,
                             'compute': summarize_compute(json.loads((path.parent/'compute-samples.json').read_text()))})
        groups = {}
        for row in rows:
            if 'compute' not in row and (root/row['id']/'compute-samples.json').exists():
                row['compute'] = summarize_compute(json.loads((root/row['id']/'compute-samples.json').read_text()))
            key = json.dumps([row['workload'], row['engine'], row['concurrency']], sort_keys=True)
            groups.setdefault(key, []).append(row)
        ranking = []
        for group in groups.values():
            complete = len(group) == c['repeats'] and all(r['eligible'] for r in group)
            record = {k: group[0][k] for k in ('workload', 'engine', 'concurrency')}
            record.update(eligible=complete, trials=len(group), median_output_tokens_per_second=None)
            if complete:
                rates = [r['metrics']['output_throughput'] for r in group]
                record.update(median_output_tokens_per_second=statistics.median(rates),
                              min_output_tokens_per_second=min(rates), max_output_tokens_per_second=max(rates))
            ranking.append(record)
        ranking.sort(key=lambda r: (r['workload']['name'], not r['eligible'], -(r['median_output_tokens_per_second'] or 0)))
        dump(root/'ranking.json', ranking)
        compute_ranking = []
        for group in groups.values():
            usable = [r for r in group if r.get('compute', {}).get('output_tokens_per_second', 0) > 0]
            if usable:
                compute_ranking.append({**{k: group[0][k] for k in ('engine', 'workload', 'concurrency')},
                    'measured_repeats': len(usable), 'planned_repeats': c['repeats'],
                    'median_output_tokens_per_second': statistics.median(r['compute']['output_tokens_per_second'] for r in usable),
                    'observed_seconds': sum(r['compute'].get('observed_seconds', 0) for r in usable),
                    'statuses': [r['status'] for r in usable]})
        compute_ranking.sort(key=lambda r: (r['workload']['name'], -r['median_output_tokens_per_second']))
        dump(root/'compute-ranking.json', compute_ranking)
        with (root/'trials.csv').open('w', newline='') as stream:
            fields = ['id', 'workload', 'engine', 'concurrency', 'repeat', 'status', 'eligible',
                      'output_throughput', 'request_throughput', 'p99_ttft_ms', 'p99_tpot_ms', 'error']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = {k: row.get(k) for k in fields}
                for key in ('workload', 'engine'):
                    flat[key] = json.dumps(flat[key], sort_keys=True)
                flat.update({k: row.get('metrics', {}).get(k) for k in fields if k.endswith('_ms') or k.endswith('throughput')})
                writer.writerow(flat)
        # Analysis exports retain individual repeats and failures, independent of fitting.
        dump(root/'observations.json', {'schema_version': 1, 'observations': rows})
        flat_rows = []
        for row in rows:
            flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            flat['model'] = c['model']
            flat['requested_prompts'] = c['requests']
            flat['raw_result_path'] = str(root/row['id']/'result.json')
            for group in ('engine', 'workload', 'metrics', 'compute'):
                for key, value in row.get(group, {}).items():
                    if not isinstance(value, (dict, list)):
                        flat[f'{group}.{key}'] = value
            flat_rows.append(flat)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=sorted({key for row in flat_rows for key in row}))
        writer.writeheader()
        writer.writerows(flat_rows)
        atomic_text(root/'analysis.csv', buffer.getvalue())
        return {'finished_trials': len(rows), 'ranking': str(root/'ranking.json'), 'csv': str(root/'trials.csv'),
                'analysis': str(root/'analysis.csv'), 'observations': str(root/'observations.json')}


    def fit(self, output, source="compute"):
        """Fit a regularized quadratic response surface; suggest unmeasured settings.

        Runs on the analysis host with NumPy. Predictions are hypotheses, never
        promoted to measured winners. Cross-validation holds out whole settings.
        """
        import numpy as np
        root = Path(output)
        c = load(root/'config.json')
        self.report(output)
        if source not in {'compute', 'completed'}:
            raise ValueError('source must be compute or completed')
        ranking = json.loads((root/('compute-ranking.json' if source == 'compute' else 'ranking.json')).read_text())
        if source == 'compute':
            ranking = [{**r, 'eligible': True} for r in ranking]
        full = {**c}
        full.pop('sample_settings', None)
        candidates = trials(full)
        keys = sorted(c['grid']) + ['concurrency']
        levels = [c['grid'][key] if key != 'concurrency' else c['concurrency'] for key in keys]
        if any(any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in vs) for vs in levels):
            raise ValueError('fit currently supports numeric grid values (None means disabled/zero)')

        def identity(r):
            return json.dumps([r['engine'], r['concurrency']], sort_keys=True)

        def vector(r):
            return [float((r['concurrency'] if key == 'concurrency' else r['engine'][key]) or 0) for key in keys]

        bounds = np.array([[min(float(v or 0) for v in vs), max(float(v or 0) for v in vs)] for vs in levels])
        scale = np.maximum(bounds[:, 1] - bounds[:, 0], 1)

        def features(records):
            x = (np.array([vector(r) for r in records]) - bounds[:, 0]) / scale
            return np.column_stack([np.ones(len(x)), x, x*x] + [x[:, i]*x[:, j] for i in range(len(keys)) for j in range(i+1, len(keys))])

        def regress(x, y, alpha):
            penalty = np.eye(x.shape[1]) * alpha
            penalty[0, 0] = 0
            return np.linalg.solve(x.T @ x + penalty, x.T @ y)

        reports = []
        for workload in c['workloads']:
            observed = [r for r in ranking if r['workload'] == workload]
            valid = [r for r in observed if r['eligible']]
            result = {'workload': workload, 'measurement_source': source, 'successful_settings': len(valid),
                      'excluded_settings': len(observed)-len(valid)}
            if len(valid) < 8:
                reports.append({**result, 'status': 'insufficient_data', 'minimum_successful_settings': 8})
                continue
            x = features(valid)
            y = np.log([r['median_output_tokens_per_second'] for r in valid])
            # Leave-one-setting-out CV; repetitions never leak across folds.
            scores = []
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                predictions = []
                for i in range(len(y)):
                    mask = np.arange(len(y)) != i
                    predictions.append(float(x[i] @ regress(x[mask], y[mask], alpha)))
                scores.append((float(np.mean((np.array(predictions)-y)**2)), alpha))
            mse, alpha = min(scores)
            coefficients = regress(x, y, alpha)
            seen = {identity(r) for r in observed}
            pool = {identity(t): t for t in candidates if t['workload'] == workload and identity(t) not in seen}
            suggestions = []
            for t in pool.values():
                predicted = math.exp(float(np.clip(features([t])[0] @ coefficients, -700, 700)))
                suggestions.append({'engine': t['engine'], 'concurrency': t['concurrency'],
                                    'predicted_output_tokens_per_second': predicted})
            suggestions.sort(key=lambda r: -r['predicted_output_tokens_per_second'])
            result.update(status='fitted', ridge_alpha=alpha, cv_log_rmse=math.sqrt(mse),
                          coefficients=coefficients.tolist(), dimensions=keys, bounds=bounds.tolist(),
                          suggestions=suggestions[:10],
                          note='Predictions require measurement; failures excluded, so feasibility is not predicted. CV error is not a confidence interval.')
            reports.append(result)
        dump(root/'fit.json', reports)
        return {'fit': str(root/'fit.json'), 'workloads': reports}

    def rexs(self, config, profile, output, image, mounts=None, gpus=1, cpus=16, memory='128GiB', submit=False, max_trials=None, resume=False):
        """Snapshot a sweep into one Rexs-owned allocation; render unless submit=True."""
        from rexs.cli import Rexs
        c = load(config)
        planned = trials(c)
        for trial in planned:
            e = trial['engine']
            needed = int(e.get('tensor_parallel_size', 1)) * int(e.get('pipeline_parallel_size', 1)) * int(e.get('data_parallel_size', 1))
            if needed > int(gpus):
                raise ValueError(f'Engine requires {needed} GPUs but allocation requests {gpus}')
        root = Path(output).resolve()
        if resume:
            if load(root/'config.json') != c:
                raise ValueError('Resume requires the same configuration')
        else:
            root.mkdir(parents=True, exist_ok=False)
            dump(root/'config.json', c)
            (root/'worker.py').write_bytes(Path(__file__).read_bytes())
            (root/'trace_replay.py').write_bytes(Path(__file__).with_name('trace_replay.py').read_bytes())
        mounts = json.loads(mounts) if isinstance(mounts, str) else (mounts or {})
        datasets = [{'mountPath': '/sweep', 'source': {'hostPath': str(root)}}]
        for host, target in mounts.items():
            if not Path(host).is_absolute() or not target.startswith('/') or target == '/sweep' or target.startswith('/sweep/'):
                raise ValueError('Use absolute host/container mounts outside /sweep')
            datasets.append({'mountPath': target, 'source': {'hostPath': host}, 'readOnly': True})
        command = ['python', '/sweep/worker.py', 'run', '--config=/sweep/config.json', '--output=/sweep/results']
        if resume and (root/'results').exists():
            command.append('--resume=True')
        if max_trials is not None:
            command.append(f'--max_trials={int(max_trials)}')
        spec = {'version': 'v2', 'tasks': [{'name': 'vllm-sweep', 'image': {'beaker': image},
                'command': command, 'datasets': datasets, 'result': {'path': '/tmp/beaker-result'},
                'resources': {'gpuCount': int(gpus), 'cpuCount': int(cpus), 'memory': memory}}]}
        dump(root/'experiment.json', spec)
        rex = Rexs()
        rex.validate(str(root/'experiment.json'), profile=profile, strict=True)
        script = rex.render(str(root/'experiment.json'), profile=profile, output=str(root/'experiment.sbatch'), strict=True)
        result = {'script': script, 'submitted': False}
        if submit:
            import fcntl
            lock_path = Path.home()/'.cache'/'primebeaker-interactive-submit.lock'
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                check_interactive_slots()
                result.update(submitted=True, experiment=rex.submit(str(root/'experiment.json'), profile=profile, strict=True))
        dump(root/'receipt.json', result)
        return result


def main():
    import fire
    fire.Fire(SweepCLI())


if __name__ == '__main__':
    main()
