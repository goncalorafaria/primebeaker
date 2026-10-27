"""Require the previously verified MC32 source snapshot before solver startup."""
import importlib.util
import json
import os
from pathlib import Path

root = Path(os.environ['MC32_ROOT']).resolve()
spec = importlib.util.spec_from_file_location('mc32_snapshot', root / 'source_snapshot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if Path(module.VLLM).resolve() != Path(os.environ["MC32_VLLM_SOURCE"]).resolve():
    raise RuntimeError("Verification snapshot points at a different vLLM source")
fingerprint = module.fingerprint()
for report in root.glob('verification-*.json'):
    data = json.loads(report.read_text())
    if (isinstance(data, dict) and data.get('gpu_verified') and data.get('beaker_job_id')
            and data.get('exitstatus') == 0 and data.get('source_hashes') == fingerprint):
        print('MC32_GPU_VERIFIED', report, flush=True)
        break
else:
    raise RuntimeError('No successful GPU verification matching the MC32 source snapshot')

import packed_heads
import renderers.client
import verifiers.v1.trace
import prime_rl.trainer.batch
import prime_rl.trainer.rl.data
import vllm
for imported in (packed_heads, renderers.client, verifiers.v1.trace,
                 prime_rl.trainer.batch, prime_rl.trainer.rl.data):
    if not Path(imported.__file__).resolve().is_relative_to(root / 'runtime'):
        raise RuntimeError(f'Incorrect packed runtime: {imported.__file__}')
if not Path(vllm.__file__).resolve().is_relative_to(Path(os.environ['MC32_VLLM_SOURCE']).resolve()):
    raise RuntimeError(f'Incorrect MC32 vLLM: {vllm.__file__}')

# Schema validation is performed in the actual worker environment, before GPU launch.
import tomllib
from prime_rl.configs.rl import RLConfig
with open(os.environ['TERMINAL_RL_CONFIG'], 'rb') as handle:
    RLConfig.model_validate(tomllib.load(handle))
print('TERMINAL_MC32_CONFIG_READY', flush=True)
