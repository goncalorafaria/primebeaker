#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
export LD_LIBRARY_PATH="/weka/gfaria/search-agent-repr/judge/cuda13-compat/root/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}"
/app/.venv/bin/python -c 'import torch; torch.cuda.init(); print("CUDA preflight:", torch.version.cuda, torch.cuda.get_device_name(0), flush=True)' 
ROOT=/weka/gfaria/search-agent-repr
export PYTHONPATH="/weka/gfaria/primebeaker/src:/weka/gfaria/jtc${PYTHONPATH:+:$PYTHONPATH}"
export HEAD_REGISTRY="${HEAD_REGISTRY:-sqlite:///weka/gfaria/search-agent-repr/registry/control.sqlite3}"
# Keep discovery indirect so Redis can move without restarting clients.
export REGISTRY="head+$HEAD_REGISTRY"
export REGISTRY_CACHE_TTL_SECONDS=5
export REGISTRY_PATH="$REGISTRY"
export TIMEOUT=6600 JUDGE_TIMEOUT=6600 PYTHONUNBUFFERED=1
MODEL=/weka/gfaria/launches/pengin-9b-rl-benchmarks-100step-20261025/9b/step_400/hf-checkpoint
SERVICE=judge:search-agent-pengin9b-rl400
JUDGE_PYTHON=${JUDGE_PYTHON:-/app/judge-venv/bin/python}
test -x "$JUDGE_PYTHON"
test -f "$MODEL/model.safetensors.index.json"
# Validate the actual JudgeAPI interpreter and mounted source before GPU startup.
"$JUDGE_PYTHON" - <<'PYCHECK'
from pathlib import Path
from primebeaker.judge.judge_server import JudgeServerConfig
from primebeaker.judge.model_profiles import load_model_profile
from primebeaker.judge.pengin_harness import validate_pengin_template
from literegistry_tool_client import WebTerminalExecutionClient
import json
root = Path('/weka/gfaria/search-agent-repr')
payload = json.loads((root / 'config/judge-profiles/pengin9b_rl400.json').read_text())
profile = load_model_profile(payload['model'], JudgeServerConfig(
    model_profiles_dir=str(root / 'config/judge-profiles')))
assert profile.rubric_mode == 'pengin_submit'
assert set(profile.tools) == {'webterminal', 'submit'}
assert profile.terminal_truncation == 2000
validate_pengin_template(profile.prompt_template_path)
print('Pengin JudgeAPI implementation, dependencies and canonical prompt loaded', flush=True)
PYCHECK
LOG_DIR="$ROOT/logs/judge-${BEAKER_JOB_ID:-local}"
mkdir -p "$LOG_DIR" /tmp/beaker-result
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT
# One TP4 engine across the four assigned GPUs; two HTTP API processes.
VLLM_PYTHON=${VLLM_PYTHON:-/app/.venv/bin/python}
JUDGE_API_PROCESSES=${JUDGE_API_PROCESSES:-2}
port=${JUDGE_PORT_BASE:-25200}
  "$VLLM_PYTHON" "$ROOT/judge/serve-registered-vllm.py" --runtime=local \
    --model="$MODEL" --registry="$REGISTRY" --host=0.0.0.0 --port="$port" \
    --tensor-parallel-size=4 --data-parallel-size=1 --data-parallel-size-local=1 \
    --api-server-count="$JUDGE_API_PROCESSES" \
    --dtype=bfloat16 --max-model-len=131072 --max-num-batched-tokens=16384 --gpu-memory-utilization=0.85 \
    --max-num-seqs=128 --enable-prefix-caching --trust-remote-code \
    --mamba-cache-mode=align --reasoning-parser=qwen3 --enable-auto-tool-choice \
    --tool-call-parser=qwen3_coder --language-model-only > "$LOG_DIR/model-tp4.log" 2>&1 &
  pids+=("$!")
# Extra model nodes reuse the existing Jupiter JudgeAPI and gateways.
if [[ "${JUDGE_START_LOCAL_SERVICES:-1}" == "1" ]]; then
"$JUDGE_PYTHON" -m uvicorn primebeaker.gateway:create_app --factory   --host 0.0.0.0 --port 1213 --workers 1 > "$LOG_DIR/gateway.log" 2>&1 &
pids+=("$!")
# The API and model share REGISTRY; train workers route /judge through their gateway.
"$JUDGE_PYTHON" -m primebeaker.judge.judge_server --registry "$REGISTRY"   --port 8090 --model_path "$SERVICE" --model_gateway_url http://127.0.0.1:1213   --model_profiles_dir "$ROOT/config/judge-profiles" --workers 4   > "$LOG_DIR/api.log" 2>&1 &
pids+=("$!")
fi
# Supervise every component. Readiness is checked by the run's service preflight.
set +e
wait -n "${pids[@]}"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then rc=1; fi
exit "$rc"
