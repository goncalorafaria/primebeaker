#!/usr/bin/env bash
set -Eeuo pipefail
export BEAKER_REPLICA_RANK=${PRL_NODE_RANK:-$BEAKER_REPLICA_RANK}
RUNTIME=/weka/gfaria/prime-rl-mc32/experiments/mc32
BASE=/weka/gfaria/launches/quokka9b-verifiable-score-centering-lag16-20261013
SOURCE="$BASE/runtime/prime-rl"
VENV=/weka/gfaria/experiments/reverse-text-online-20261013/venv
export PATH="$VENV/bin:/weka/gfaria/.local/bin:$PATH"
export PYTHONPATH="/weka/gfaria/search-agent-repr/mc128-runtime/extra-deps:/weka/gfaria/search-agent-repr/policy-resume450-to600-bf16-cp4-12infer-20261105/runtime:/weka/gfaria/vllm-mc32:/weka/gfaria/launches/pengin-9b-holmes-rl-8gpu-20261025/modules:$SOURCE/packages/prime-rl-configs/src:$SOURCE/deps/pydantic-config/src:$BASE/runtime/tool-client:/weka/gfaria/primebeaker/src:/weka/gfaria/search-agent-repr/prepared-runtime"
export INFERENCE_API_PROCESSES_PER_ENGINE=4
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_SCORE_CENTERING_MC32=1 VLLM_PLUGINS=prime_rl
export VLLM_USE_COMPACT_LOGPROBS=1 VLLM_PRIME_COMPACT_LOGPROBS=0 VLLM_PRIME_FLASHINFER_SAMPLER_HEAD=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export JUDGE_TIMEOUT=6600 TIMEOUT=2700
# Holmes uses the native CUDA driver.
mkdir -p /tmp/beaker-result
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export PRL_RUN_ID="$WANDB_RUN_ID" PRL_RUN_NAME=search-agent-mc32-resume450-to600-bf16-20261105
mkdir -p "$RENDEZVOUS_DIR"
if python3 /weka/gfaria/search-agent-repr/policy-12train-8infer-cp4-b256-mc32/check-mc32-receipt.py; then
    echo "MC32 source fingerprints verified locally; proceeding without waiting for a queued peer"
elif [[ "$BEAKER_REPLICA_RANK" == 1 ]]; then
    # Exercise the actual GPU kernel, mixed batches and full vLLM generation.
    # No policy training or inference server starts before this succeeds.
    if python3 /weka/gfaria/search-agent-repr/policy-12train-8infer-cp4-b256-mc32/check-mc32-receipt.py; then
        printf 'Verified unchanged MC32 runtime using completed GPU test receipt\n' > "$RENDEZVOUS_DIR/mc32-gpu.log"
    else
    CUDA_VISIBLE_DEVICES=0 bash "$RUNTIME/run.sh" -m pytest \
        "$RUNTIME/tests/test_mc32.py" -v --junitxml=/tmp/beaker-result/mc32.xml \
        > "$RENDEZVOUS_DIR/mc32-gpu.log" 2>&1
    python3 - "$RUNTIME" <<'CHECK'
import json, os, sys
from pathlib import Path
root=Path(sys.argv[1])
result=json.loads((root/f"verification-{os.environ['BEAKER_JOB_ID']}.json").read_text())
assert result['gpu_verified'], result
CHECK
    fi
    printf 'passed\n' > "$RENDEZVOUS_DIR/mc32-gpu.ok.tmp"
    mv "$RENDEZVOUS_DIR/mc32-gpu.ok.tmp" "$RENDEZVOUS_DIR/mc32-gpu.ok"
else
    deadline=$((SECONDS + ${RENDEZVOUS_TIMEOUT_SECONDS:-86400}))
    until [[ -s "$RENDEZVOUS_DIR/mc32-gpu.ok" ]]; do
        (( SECONDS < deadline )) || { echo "MC32 GPU verification did not complete" >&2; exit 1; }
        sleep 5
    done
fi
if [[ "$BEAKER_REPLICA_RANK" == 0 ]]; then
    CUDA_VISIBLE_DEVICES=0 python3 /weka/gfaria/search-agent-repr/policy-holmes-mc128/check-fa2.py
fi
exec bash /weka/gfaria/search-agent-repr/policy-resume450-to600-bf16-cp4-12infer-20261105/multinode_rl_role.sh
