#!/usr/bin/env bash
set -euo pipefail
# Native CUDA 13-capable drivers must not be shadowed by the older compat driver.
CUDA_DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)
if [[ "${USE_CUDA_COMPAT:-auto}" != "0" && "$CUDA_DRIVER_MAJOR" -lt 580 ]]; then
  export LD_LIBRARY_PATH="/weka/gfaria/search-agent-repr/judge/cuda13-compat/root/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
fi
ROOT=/weka/gfaria/search-agent-repr/evals/frontier-structured350-420-training-tools-seed1-20261104/runtime
OUT=${EVAL_OUTPUT_DIR:-/weka/gfaria/search-agent-repr/evals/frontier-science-step210}
CKPT=${CHECKPOINT_PATH:-/weka/gfaria/search-agent-repr/outputs/search-agent-mc32-16train-16infer-cp4-resume130/search-agent-mc32-16train-16infer-cp4-resume130/checkpoints/step_210}
export EVAL_SEARCH_MODEL_PATH="search:frontier430-jupiter-five-$(basename "$OUT")"
export MODEL_PORT=${MODEL_PORT:-28743} SEARCH_PORT=${SEARCH_PORT:-28741} GATEWAY_PORT=${GATEWAY_PORT:-28742}
MODEL=${MODEL_PATH:-$OUT/hf-checkpoint}
BASE=/weka/gfaria/launches/quokka9b-verifiable-score-centering-lag16-20261013
SOURCE="$BASE/runtime/prime-rl"
TRAIN_PY=/weka/gfaria/experiments/reverse-text-online-20261013/venv/bin/python
EVAL_PY=/weka/gfaria/experiments/reverse-text-online-20261013/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
SERVICE_PY=/weka/gfaria/experiments/reverse-text-online-20261013/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
TRAIN_PATH="/weka/gfaria/search-agent-repr/mc128-runtime/extra-deps:/weka/gfaria/prime-rl-mc32/experiments/mc32/runtime:/weka/gfaria/vllm-mc32:/weka/gfaria/launches/pengin-9b-holmes-rl-8gpu-20261025/modules:$SOURCE/packages/prime-rl-configs/src:$SOURCE/deps/pydantic-config/src:$BASE/runtime/tool-client"
EVAL_PATH="$ROOT/deps:/weka/gfaria/jtc/.venv/lib/python3.12/site-packages:/weka/gfaria/primebeaker/src:/weka/gfaria/jtc:$BASE/runtime/tool-client:/weka/gfaria/search-agent-repr/services/runtime"
export PATH=/weka/gfaria/experiments/reverse-text-online-20261013/venv/bin:/weka/gfaria/.local/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export REGISTRY_PATH=head+sqlite:///weka/gfaria/search-agent-repr/registry/control.sqlite3
export REGISTRY_CACHE_TTL_SECONDS=5 TIMEOUT=2700
ATTEMPT=${BEAKER_JOB_ID:-$(date -u +%Y%m%dT%H%M%S)-$$}
LOGS="$OUT/logs/$ATTEMPT"
export WEB_METRICS_DIR="$OUT/web-metrics/$ATTEMPT"
mkdir -p "$LOGS" /tmp/beaker-result
printf '%s\n' "$LOGS" > "$OUT/logs/latest-path.txt"
exec > >(tee -a "$LOGS/launch.log") 2>&1
pids=()
cleanup() { if (( ${#pids[@]} )); then kill "${pids[@]}" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 143' TERM INT
command -v ninja
ninja --version
# Verify the actual inference interpreter against the node's injected driver.
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
PYTHONPATH="$TRAIN_PATH" "$TRAIN_PY" - <<'CUDA_CHECK'
import json, torch
from pathlib import Path
import os
torch.cuda.init()
x = torch.ones(1, device='cuda')
torch.cuda.synchronize()
print('CUDA preflight passed:', json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0), devices=torch.cuda.device_count())), flush=True)
CUDA_CHECK
printf 'Exporting fixed checkpoint %s\n' "$CKPT"
if [[ ! -f "$MODEL/EXPORT_COMPLETE" ]]; then
  PYTHONPATH="$TRAIN_PATH" "$TRAIN_PY" "$ROOT/export.py" "$CKPT" "$MODEL" > "$LOGS/export.log" 2>&1
  touch "$MODEL/EXPORT_COMPLETE"
fi
printf 'Starting fixed policy and live-web services\n'
PYTHONPATH="$TRAIN_PATH" "$TRAIN_PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 127.0.0.1 --port "$MODEL_PORT" --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 262144 --max-num-batched-tokens 131072 --max-num-seqs 128 \
  --enable-prefix-caching --mamba-cache-mode align --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only \
  > "$LOGS/model.log" 2>&1 &
pids+=("$!")
PYTHONPATH="/weka/gfaria/search-agent-repr/services/runtime:/weka/gfaria/primebeaker/src:/weka/gfaria/primebeaker/.venv/lib/python3.12/site-packages" "$SERVICE_PY" "$ROOT/eval_search_server.py" \
  > "$LOGS/search.log" 2>&1 &
pids+=("$!")
PYTHONPATH="/weka/gfaria/search-agent-repr/services/runtime:/weka/gfaria/primebeaker/src:/weka/gfaria/primebeaker/.venv/lib/python3.12/site-packages" "$SERVICE_PY" -m uvicorn primebeaker.gateway:create_app --factory \
  --host 127.0.0.1 --port "$GATEWAY_PORT" --workers 1 > "$LOGS/gateway.log" 2>&1 &
pids+=("$!")
export EVAL_SERVICE_PIDS="${pids[*]}"
export PYTHONPATH="$EVAL_PATH"
"$EVAL_PY" "$ROOT/preflight.py" "$OUT"
"$EVAL_PY" "$ROOT/run.py" --model "$MODEL" --inference-url "http://127.0.0.1:$MODEL_PORT/v1" \
  --gateway-url "http://127.0.0.1:$GATEWAY_PORT" --seed "$EVAL_SEED" --output-dir "$OUT/smoke" --max-rows 1
"$EVAL_PY" - "$OUT/smoke/traces.jsonl" <<'PY'
import json,sys
r=json.loads(open(sys.argv[1]).readline())
assert not r.get('errors'), r.get('errors')
assert r['info']['search_agent']['final_response'], 'No final answer in smoke rollout'
print('Smoke rollout passed',flush=True)
PY
run_seed() {
  local seed="$1" seed_out="${OUT}-seed${1}"
  "$EVAL_PY" "$ROOT/run.py" --model "$MODEL" --inference-url "http://127.0.0.1:$MODEL_PORT/v1" \
    --gateway-url "http://127.0.0.1:$GATEWAY_PORT" --seed "$seed" --output-dir "$seed_out/full" > "$LOGS/seed-$seed.log" 2>&1
  touch "$seed_out/full/INFERENCE_COMPLETE"
}
seed_pids=()
for seed in 1 2 3 4 5; do
 run_seed "$seed" &
 seed_pids+=("$!");pids+=("$!")
done
for pid in "${seed_pids[@]}"; do wait "$pid"; done
printf 'All five Jupiter seeds completed\n'
