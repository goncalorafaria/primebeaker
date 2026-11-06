"""Shared worker for the mounted-runtime medical/science examples."""
from __future__ import annotations

import atexit
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib
from urllib.request import urlopen

root = Path(os.environ['LAUNCH_DIR'])
settings = json.loads((root / 'launch.json').read_text())
role = os.environ['WORKER_ROLE']
children: list[subprocess.Popen] = []


def cleanup():
    for process in reversed(children):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in children:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


atexit.register(cleanup)
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))


def start(command):
    process = subprocess.Popen(command, start_new_session=True)
    children.append(process)
    return process


def require_children():
    for process in children:
        if process.poll() is not None:
            raise RuntimeError(f'Child exited: {process.args!r}; exit={process.returncode}')


def wait_health(url):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        require_children()
        try:
            with urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError(f'Service did not become ready: {url}')


# Resolve Redis through the same stable head used by tools and model workers.
registry = subprocess.check_output(
    [sys.executable, str(root / 'discover_registry.py')], text=True).strip()
os.environ.update(REGISTRY=registry, REGISTRY_PATH=registry,
                  GATEWAY_LOAD_REDIS_URL=registry,
                  GATEWAY_LOAD_NAMESPACE=settings['judge_service'] + '-routing',
                  REGISTRY_CACHE_TTL_SECONDS='5', CHAT_SOFT_AFFINITY='false',
                  TIMEOUT='7500', JUDGE_TIMEOUT='7500')
marker = Path(settings['output_dir']) / ('.completed-' + settings['run_name'])
marker.parent.mkdir(parents=True, exist_ok=True)

if role == 'model':
    import vllm
    if vllm.__version__ != '0.29.0':
        raise RuntimeError('This serving recipe requires vLLM 0.29.0')
    command = ['literegistry', 'vllm', '--runtime=local',
               '--registry=' + registry, '--model=' + settings['judge_model'],
               '--host=0.0.0.0', '--port=28120', '--tensor-parallel-size=4',
               '--data-parallel-size=1', '--data-parallel-size-local=1',
               '--max-model-len=' + str(settings['judge_context']),
               '--max-num-batched-tokens=16384', '--max-num-seqs=1024',
               '--enable-prefix-caching', '--trust-remote-code',
               '--mamba-cache-mode=align', '--reasoning-parser=qwen3',
               '--enable-auto-tool-choice', '--tool-call-parser=qwen3_coder',
               '--language-model-only']
    if settings['judge'] == 'moe':
        command.append('--enable-expert-parallel')
    start(command)
elif role in ('api', 'trainer'):
    start([sys.executable, '-m', 'uvicorn', 'primebeaker.gateway:create_app',
           '--factory', '--host', '127.0.0.1', '--port', '1213', '--workers', '1'])
    wait_health('http://127.0.0.1:1213/health')
    if role == 'api':
        # Pengin's Verifiers environment must be constructed on its main thread.
        os.environ['JUDGE_PRELOAD_HARNESS'] = '1' if settings['judge'] == 'pengin9b' else '0'
        start([sys.executable, '-m', 'primebeaker.judge.judge_server',
               '--registry', registry, '--port', '28461',
               '--model_path', settings['judge_service'],
               '--model_gateway_url', 'http://127.0.0.1:1213',
               '--tool_server_url', 'http://127.0.0.1:1213',
               '--model_profiles_dir', str(root / 'profiles'),
               '--workers', '1' if settings['judge'] == 'pengin9b' else '4'])
    else:
        from prime_rl.configs.rl import RLConfig
        from primebeaker.config import RLTrainingToml
        from primebeaker.coordination import wait_for_services
        config = tomllib.loads((root / 'train.toml').read_text())
        run_dir = Path(config['output_dir']) / config['run']['name']
        if config.get('resume'):
            checkpoint = run_dir / 'checkpoints' / ('step_' + str(config['resume']['step']))
            for filename in ('trainer/.metadata', 'orchestrator/progress.pt'):
                if not (checkpoint / filename).is_file():
                    raise RuntimeError(f'Incomplete resume checkpoint: {checkpoint / filename}')
        elif (run_dir / 'checkpoints').exists():
            raise RuntimeError('Fresh run already has checkpoints; choose an explicit resume phase')
        RLConfig.model_validate(copy.deepcopy(config))
        resolved = RLTrainingToml.from_path(root / 'train.toml').with_environment_namespace()
        resolved.write('/tmp/medical-science-runtime.toml')
        requirements = {settings['judge_model']: 1, settings['judge_service']: 1}
        if settings['judge'] == 'pengin9b':
            requirements.update(terminal=1)
            requirements[settings['local_search_model']] = 1
        wait_for_services(registry, requirements, timeout=21600, poll_interval=5)
        start(['rl', '@', '/tmp/medical-science-runtime.toml'])
        policy = children[-1]
        while policy.poll() is None:
            for process in children[:-1]:
                if process.poll() is not None:
                    raise RuntimeError('Trainer-owned gateway exited')
            time.sleep(5)
        if policy.returncode:
            raise SystemExit(policy.returncode)
        # Base phase one deliberately leaves its judge pool available for resume.
        if config['max_steps'] >= 800:
            marker.touch()
        else:
            (marker.parent / ('.phase400-' + settings['run_name'])).touch()
        raise SystemExit(0)
else:
    raise ValueError('Unknown worker role: ' + role)

while not marker.exists():
    require_children()
    time.sleep(10)
