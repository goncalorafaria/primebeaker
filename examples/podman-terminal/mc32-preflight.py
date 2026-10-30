"""Validate installed native MC32 packages and the solver's RL configuration."""
import json
import os
import tomllib
from importlib.metadata import distribution

from renderers import packed_heads
import renderers.client
import verifiers.v1.trace
import prime_rl.trainer.batch
import prime_rl.trainer.rl.data
import vllm.mc_logprobs
import vllm.v1.worker.gpu.sample.mc_logprobs
from prime_rl.configs.rl import RLConfig

# The standard image installs the native vLLM fork directly from this commit.
expected_vllm = "409c9efd1aeb0fdd3f32d06554e33f520534e273"
metadata = distribution("vllm").read_text("direct_url.json")
if metadata is None or json.loads(metadata).get("vcs_info", {}).get("commit_id") != expected_vllm:
    raise RuntimeError("Use the pinned native MC32 image; installed vLLM source revision differs")

with open(os.environ['TERMINAL_RL_CONFIG'], 'rb') as handle:
    RLConfig.model_validate(tomllib.load(handle))
print('NATIVE_MC32_PACKAGES', packed_heads.__file__, flush=True)
print('TERMINAL_MC32_CONFIG_READY', flush=True)
