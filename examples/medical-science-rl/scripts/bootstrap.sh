#!/usr/bin/env bash
set -euo pipefail
: "${WORKER_PYTHON:?}" "${LAUNCH_DIR:?}"
if [[ "$WORKER_PYTHON" == */* ]]; then
  export PATH="$(dirname "$WORKER_PYTHON"):$PATH"
fi
mkdir -p /tmp/beaker-result
ulimit -n 65536
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export VLLM_ENABLE_CUDA_COMPATIBILITY=0
if [[ "$WORKER_ROLE" != api ]]; then
  selected="$("$WORKER_PYTHON" "$LAUNCH_DIR/select_cuda_driver.py")"
  export LD_LIBRARY_PATH="$selected"
fi
if [[ "$WORKER_ROLE" == trainer ]]; then
  export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_PLUGINS=prime_rl
  export VLLM_SCORE_CENTERING_MC32=1 VLLM_USE_COMPACT_LOGPROBS=1
  export VLLM_PRIME_COMPACT_LOGPROBS=0 VLLM_PRIME_FLASHINFER_SAMPLER_HEAD=0
fi
exec "$WORKER_PYTHON" "$LAUNCH_DIR/worker.py"
