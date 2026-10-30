"""Preview by default; explicitly --submit to train a fresh terminal solver."""
from __future__ import annotations

import argparse
import json
import shlex
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from primebeaker.config import dump_toml_document
from primebeaker.rl import BeakerRLBackend, RLLaunchRequest

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, default=Path(__file__).with_name('mc32-fp8-4b.toml'))
p.add_argument('--image', required=True, help='Immutable Beaker image built from the pinned native MC32 forks')
p.add_argument('--gateway-url', required=True)
p.add_argument('--output-root', type=Path, required=True)
p.add_argument('--cluster', default='ai2/jupiter')
p.add_argument('--workspace', default='ai2/oe-agents')
p.add_argument('--budget', default='ai2/oe-omai')
p.add_argument('--train-data', type=Path)
p.add_argument('--validation-data', type=Path)
p.add_argument('--runtime-python', type=Path, default=Path('/app/.venv/bin/python'))
p.add_argument('--podman-client-source', type=Path, help='Optional shared application client source; otherwise use the installed package')
p.add_argument('--uv-executable', type=Path, default=Path('/usr/local/bin/uv'))
p.add_argument('--cuda-compat-dir', default='')
p.add_argument('--submit', action='store_true')
a = p.parse_args()
c = tomllib.loads(a.config.read_text())
name = c['run']['name'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6]
launch = a.output_root.resolve() / name
launch.mkdir(parents=True, exist_ok=False)
c['output_dir'] = str(launch / 'output')
c['run']['name'] = name
c['monitors']['wandb']['name'] = name
ids = {}
for split, override in [('train', a.train_data), ('eval', a.validation_data)]:
    taskset = c['orchestrator'][split]['source'][0]['env']['taskset']
    path = override or root / taskset['dataset']
    path = path.resolve(strict=True)
    taskset['dataset'] = str(path)
    taskset['task']['execution']['gateway_url'] = a.gateway_url
    # Split disjointness is by original task, not rollout/reference ID.
    ids[split] = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = str(row.get('task_id') or row['record_id'])
                if key in ids[split]:
                    raise ValueError(f'Duplicate task {key} in {split}')
                ids[split].add(key)
if ids['train'] & ids['eval']:
    raise ValueError('Training and validation task IDs overlap')
if not all(ids.values()):
    raise ValueError('Both splits must contain tasks')
config = launch / 'train.toml'
config.write_text(dump_toml_document(c))
env = {
    'PRIMEBEAKER_SOURCE': str(root),
    'MC32_PYTHON': str(a.runtime_python),
    'PODMAN_GATEWAY_URL': a.gateway_url, 'TERMINAL_RL_CONFIG': str(config),
    'CUDA_COMPAT_DIR': a.cuda_compat_dir, 'MC32_UV': str(a.uv_executable),
}
if a.podman_client_source is not None:
    env['PODMAN_CLIENT_SOURCE'] = str(a.podman_client_source.resolve(strict=True))
request = RLLaunchRequest.from_toml(
    config, image=a.image, workspace=a.workspace, clusters=(a.cluster,), budget=a.budget,
    priority='urgent', min_runtime_hours=1, working_dir=root, scratch_dir=launch,
    home_dir=launch, environment=env, wandb_secret='gfaria_WANDB_API_KEY', hf_secret='HF_TOKEN',
    setup_command='source ' + shlex.quote(str(Path(__file__).with_name('mc32-runtime.sh'))),
)
backend = BeakerRLBackend()
receipt = backend.submit(request) if a.submit else backend.preview(request)
(launch / 'receipt.json').write_text(receipt.model_dump_json(indent=2))
print(receipt.model_dump_json(indent=2))
