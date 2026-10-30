#!/usr/bin/env bash
# Source inside the Beaker worker before PrimeBeaker starts native rl.
set -euo pipefail
: "${PRIMEBEAKER_SOURCE:?Set the checkout containing the native solver}"
: "${MC32_PYTHON:?Set the native MC32 image Python executable}"
: "${MC32_UV:?Set the uv executable available in the worker}"
: "${PODMAN_GATEWAY_URL:?Set the reachable gateway root URL}"
export PATH="$(dirname -- "$MC32_PYTHON"):$(dirname -- "$MC32_UV"):$PATH"
export PYTHONPATH="$PRIMEBEAKER_SOURCE/src"
if [[ -n "${PODMAN_CLIENT_SOURCE:-}" ]]; then
    export PYTHONPATH="$PODMAN_CLIENT_SOURCE:$PYTHONPATH"
fi
export VLLM_SCORE_CENTERING_MC32=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_COMPACT_LOGPROBS=1 VLLM_PRIME_COMPACT_LOGPROBS=0
export VLLM_PRIME_FLASHINFER_SAMPLER_HEAD=0 VLLM_PLUGINS=prime_rl
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
# Jupiter may need this; Holmes normally uses the native driver. Explicit opt-in.
if [[ "${CUDA_COMPAT_DIR:-}" != "" ]]; then
    export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
fi
"$MC32_UV" run --no-project --python "$MC32_PYTHON" "$PRIMEBEAKER_SOURCE/examples/podman-terminal/mc32-preflight.py"
