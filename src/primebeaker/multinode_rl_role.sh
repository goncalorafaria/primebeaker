#!/usr/bin/env bash
# Heterogeneous Prime-RL on replicated Beaker tasks. Replica 0 owns the trainer,
# orchestrator, router, and gateway; remaining replicas are inference-only.

set -Eeuo pipefail

: "${BEAKER_REPLICA_RANK:?missing Beaker replica rank}"
required=(
    TOTAL_NODES TOTAL_INFER_RANKS GPUS_PER_NODE TRAINER_GPU_COUNT MIXED_INFER_GPU_COUNT
    CONFIG_PATH OUTPUT_DIR WORKING_DIR SCRATCH LAUNCH_ID RENDEZVOUS_DIR
    RENDEZVOUS_TIMEOUT_SECONDS REGISTRY GATEWAY_PORT GATEWAY_WORKERS
    REQUIRED_LITEREGISTRY_SERVICES LOCAL_SEARCH_MODEL_PATHS_JSON
)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment variable: $name" >&2; exit 2; }
done

RANK=$BEAKER_REPLICA_RANK
if (( RANK == 0 )); then
    TRAIN_GPU_COUNT=$TRAINER_GPU_COUNT
    if (( MIXED_INFER_GPU_COUNT > 0 )); then
        ROLE=mixed
        INFER_GPU_START=$TRAIN_GPU_COUNT
        INFER_GPU_COUNT=$MIXED_INFER_GPU_COUNT
    else
        ROLE=trainer
        INFER_GPU_START=0
        INFER_GPU_COUNT=0
    fi
else
    ROLE=inference
    TRAIN_GPU_COUNT=0
    INFER_GPU_START=0
    INFER_GPU_COUNT=$GPUS_PER_NODE
fi
(( RANK >= 0 && RANK < TOTAL_NODES )) || { echo "invalid node rank $RANK" >&2; exit 2; }
(( TRAIN_GPU_COUNT + INFER_GPU_COUNT <= GPUS_PER_NODE )) || { echo "GPU partition exceeds node" >&2; exit 2; }

CONFIG_DIR="$OUTPUT_DIR/configs"
RUNTIME_DIR="$RENDEZVOUS_DIR/runtime"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$RENDEZVOUS_DIR" "$RUNTIME_DIR" "$LOG_DIR/inference" "$LOG_DIR/trainer" /tmp/beaker-result

atomic_write() {
    local path=$1 value=$2 temporary="${1}.tmp.${RANK}.$$"
    printf '%s\n' "$value" > "$temporary"
    mv "$temporary" "$path"
}

on_exit() {
    local code=$?
    trap - EXIT INT TERM
    set +e
    if [[ "$ROLE" != inference && "$code" -eq 0 ]]; then
        atomic_write "$RENDEZVOUS_DIR/complete" "Prime-RL completed on node $RANK"
    elif (( code != 0 )) && [[ ! -s "$RENDEZVOUS_DIR/complete" && ! -s "$RENDEZVOUS_DIR/shutdown.requested" ]]; then
        atomic_write "$RENDEZVOUS_DIR/shutdown.requested" "Prime-RL node $RANK ($ROLE) exited with code $code"
    fi
    atomic_write "$RENDEZVOUS_DIR/status.$RANK" "$code $ROLE"
    local pids
    pids=$(jobs -pr)
    [[ -z "$pids" ]] || { kill -TERM $pids 2>/dev/null; wait $pids 2>/dev/null; }
    if (( RANK == 0 )) && [[ -n "${PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID:-}" ]]; then
        echo "Stopping managed service experiment $PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID"
        beaker experiment stop "$PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID" || \
            echo "warning: could not stop managed service experiment $PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID" >&2
    fi
    exit "$code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_file() {
    local path=$1 description=$2 deadline=$((SECONDS + RENDEZVOUS_TIMEOUT_SECONDS))
    until [[ -s "$path" ]]; do
        [[ ! -s "$RENDEZVOUS_DIR/shutdown.requested" ]] || {
            echo "shutdown while waiting for $description" >&2; return 1;
        }
        (( SECONDS < deadline )) || { echo "timed out waiting for $description" >&2; return 1; }
        sleep 2
    done
}

watch_peers() {
    local marker marker_rank code peer_role
    while true; do
        [[ ! -s "$RENDEZVOUS_DIR/complete" ]] || return 0
        [[ ! -s "$RENDEZVOUS_DIR/shutdown.requested" ]] || return 92
        for marker in "$RENDEZVOUS_DIR"/status.*; do
            [[ -f "$marker" ]] || continue
            marker_rank=${marker##*.}
            [[ "$marker_rank" == "$RANK" ]] && continue
            read -r code peer_role < "$marker"
            (( code == 0 )) || return 90
            [[ "$peer_role" == inference ]] || return 0
            return 91
        done
        sleep 3
    done
}

cd "$WORKING_DIR"
if [[ -n "${PRIMEBEAKER_SETUP_COMMAND:-}" ]]; then
    bash -lc "$PRIMEBEAKER_SETUP_COMMAND"
fi
python3 -c 'import prime_rl, primebeaker'
command -v inference >/dev/null
command -v orchestrator >/dev/null
command -v torchrun >/dev/null
if [[ -n "${PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID:-}" ]]; then
    command -v beaker >/dev/null
    [[ -n "${BEAKER_TOKEN:-}" ]] || { echo "missing BEAKER_TOKEN for managed service cleanup" >&2; exit 2; }
fi

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache}
export HF_HOME=${HF_HOME:-$XDG_CACHE_HOME/huggingface}
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export TMPDIR=${TMPDIR:-/tmp/primebeaker-rl-$LAUNCH_ID-$RANK}
mkdir -p "$OUTPUT_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
ulimit -n 65536

python3 -m primebeaker.coordination wait-redis \
    --registry "$REGISTRY" \
    --timeout "$RENDEZVOUS_TIMEOUT_SECONDS" \
    --stop-file "$RENDEZVOUS_DIR/shutdown.requested"

LOCAL_HOST=$(hostname -f 2>/dev/null || hostname)
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
LOCAL_IP=${LOCAL_IP:-$LOCAL_HOST}
atomic_write "$RENDEZVOUS_DIR/host.$RANK" "$LOCAL_IP"
atomic_write "$RENDEZVOUS_DIR/layout.$RANK" "$INFER_GPU_START $INFER_GPU_COUNT"
echo "rank=$RANK role=$ROLE train=$TRAIN_GPU_COUNT infer=$INFER_GPU_COUNT host=$LOCAL_IP"
for ((peer = 0; peer < TOTAL_NODES; peer++)); do
    wait_for_file "$RENDEZVOUS_DIR/host.$peer" "node $peer hostname"
    wait_for_file "$RENDEZVOUS_DIR/layout.$peer" "node $peer GPU layout"
done
MASTER_ADDR=$(<"$RENDEZVOUS_DIR/host.0")
export MASTER_ADDR

if [[ "$ROLE" != inference ]]; then
    python3 -m primebeaker.multinode prepare-rl \
        --config "$CONFIG_PATH" \
        --config-dir "$CONFIG_DIR" \
        --runtime-dir "$RUNTIME_DIR"
fi
wait_for_file "$RUNTIME_DIR/runtime.ready" "resolved Prime-RL component configs"
# shellcheck disable=SC1091
source "$RUNTIME_DIR/runtime.env"

if command -v ibv_devinfo >/dev/null 2>&1; then
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[:blank:]' | paste -sd, || true)
    [[ -z "$IB_HCA" ]] || export NCCL_IB_HCA=$IB_HCA
fi
if command -v ip >/dev/null 2>&1; then
    GLOO_IFNAME=$(ip -o -4 addr show to "$LOCAL_IP" 2>/dev/null | awk '{print $2; exit}' || true)
    [[ -z "$GLOO_IFNAME" ]] || export GLOO_SOCKET_IFNAME=$GLOO_IFNAME
fi

watch_peers &
PEER_WATCH_PID=$!

# Build the one global router worker list from every node's published GPU slice.
worker_urls=()
admin_urls=()
computed_ranks=0
for ((node = 0; node < TOTAL_NODES; node++)); do
    node_host=$(<"$RENDEZVOUS_DIR/host.$node")
    read -r node_infer_start node_infer_count < "$RENDEZVOUS_DIR/layout.$node"
    (( node_infer_count % INFERENCE_TP == 0 )) || { echo "inference slice not divisible by TP" >&2; exit 2; }
    node_rank_count=$((node_infer_count / INFERENCE_TP))
    for ((local_dp = 0; local_dp < node_rank_count; local_dp++)); do
        endpoint="http://$node_host:$((BACKEND_PORT + local_dp))"
        worker_urls+=("$endpoint")
        admin_urls+=("$endpoint/v1")
        computed_ranks=$((computed_ranks + 1))
    done
done
(( computed_ranks == TOTAL_INFER_RANKS )) || { echo "inference rank count mismatch" >&2; exit 2; }

if (( RANK == 0 )); then
    command -v vllm-router >/dev/null
    atomic_write "$RENDEZVOUS_DIR/router.url" "http://$LOCAL_IP:$ROUTER_PORT/v1"
    atomic_write "$RENDEZVOUS_DIR/admin.urls" "${admin_urls[*]}"
    vllm-router \
        --policy "$ROUTER_POLICY" \
        --host 0.0.0.0 \
        --port "$ROUTER_PORT" \
        --request-id-headers x-session-id \
        --prometheus-port "$((ROUTER_PORT + 21000))" \
        --worker-startup-timeout-secs 4200 \
        --worker-urls "${worker_urls[@]}" \
        >> "$LOG_DIR/inference/router.log" 2>&1 &
    ROUTER_PID=$!
fi

# shellcheck disable=SC1091
source "$RUNTIME_DIR/inference.env"
LOCAL_INFER_RANKS=$((INFER_GPU_COUNT / INFERENCE_TP))
INFERENCE_PIDS=()
for ((local_dp = 0; local_dp < LOCAL_INFER_RANKS; local_dp++)); do
    port=$((BACKEND_PORT + local_dp))
    gpu_start=$((INFER_GPU_START + local_dp * INFERENCE_TP))
    gpu_end=$((gpu_start + INFERENCE_TP - 1))
    gpus=$(seq -s, "$gpu_start" "$gpu_end")
    cache="$TMPDIR/inference/node_${RANK}_rank${local_dp}"
    mkdir -p "$cache/vllm" "$cache/inductor" "$cache/rpc"
    CUDA_VISIBLE_DEVICES="$gpus" \
        VLLM_RPC_BASE_PATH="$cache/rpc" \
        VLLM_CACHE_ROOT="$cache/vllm" \
        TORCHINDUCTOR_CACHE_DIR="$cache/inductor" \
        inference @ "$CONFIG_DIR/inference.toml" \
        --server.host 0.0.0.0 \
        --server.port "$port" \
        --parallel.dp 1 \
        --data-parallel-size-local 1 \
        --api-server-count 1 \
        2>&1 | tee -a "$LOG_DIR/inference/node_${RANK}_rank${local_dp}.log" &
    INFERENCE_PIDS+=("$!")
done

if [[ "$ROLE" == inference ]]; then
    set +e
    wait -n "$PEER_WATCH_PID" "${INFERENCE_PIDS[@]}"
    status=$?
    set -e
    (( status != 0 )) || [[ -s "$RENDEZVOUS_DIR/complete" ]] || status=91
    exit "$status"
fi

# The training replica owns the local tool gateway and readiness barrier.
python3 -m primebeaker.gateway \
    --registry "$REGISTRY" \
    --port "$GATEWAY_PORT" \
    --workers "$GATEWAY_WORKERS" \
    --timeout 800 \
    --judge-timeout 800 \
    > "$OUTPUT_DIR/literegistry-gateway.log" 2>&1 &
GATEWAY_PID=$!
sleep 5
kill -0 "$GATEWAY_PID"

if [[ "$REQUIRED_LITEREGISTRY_SERVICES" != "{}" ]]; then
    python3 -m primebeaker.coordination wait-services \
        --registry "$REGISTRY" \
        --requirements-json "$REQUIRED_LITEREGISTRY_SERVICES" \
        --timeout "$RENDEZVOUS_TIMEOUT_SECONDS" \
        --stop-file "$RENDEZVOUS_DIR/shutdown.requested"
fi

gateway_ready=false
for attempt in $(seq 1 35); do
    if python3 -m primebeaker.coordination probe-gateway \
        --url "http://127.0.0.1:$GATEWAY_PORT" \
        --model-paths-json "$LOCAL_SEARCH_MODEL_PATHS_JSON" \
        --required-path /judge \
        --timeout 70; then
        gateway_ready=true
        break
    fi
    kill -0 "$GATEWAY_PID" 2>/dev/null || { echo "gateway exited before readiness" >&2; exit 1; }
    sleep 2
done
[[ "$gateway_ready" == true ]] || { echo "gateway failed readiness" >&2; exit 1; }

wait_for_file "$RENDEZVOUS_DIR/router.url" "inference router URL"
wait_for_file "$RENDEZVOUS_DIR/admin.urls" "inference rank endpoints"
INFER_URL=$(<"$RENDEZVOUS_DIR/router.url")
read -r -a ADMIN_URLS < "$RENDEZVOUS_DIR/admin.urls"
TRAIN_GPUS=$(seq -s, 0 "$((TRAIN_GPU_COUNT - 1))")

(
    # shellcheck disable=SC1091
    source "$RUNTIME_DIR/trainer.env"
    trainer_args=(
        torchrun --role=trainer --standalone --nnodes=1
        --nproc-per-node="$TRAIN_GPU_COUNT" --node-rank=0
        --log-dir="$LOG_DIR/trainer/torchrun" --tee=3 --redirects=3
        --local-ranks-filter="$TRAINER_RANKS_FILTER"
        -m prime_rl.trainer.rl.train @ "$CONFIG_DIR/trainer.toml"
    )
    (( USE_ZMQ_TRANSPORT == 0 )) || trainer_args+=(--rollout_transport.host "$MASTER_ADDR")
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" WANDB_SHARED_LABEL=trainer "${trainer_args[@]}"
) 2>&1 | sed -u 's/^\[[a-zA-Z]*[0-9]*\]://' | tee -a "$LOG_DIR/trainer/node_0.log" &
TRAINER_PID=$!

(
    # shellcheck disable=SC1091
    source "$RUNTIME_DIR/orchestrator.env"
    orchestrator_args=(
        @ "$CONFIG_DIR/orchestrator.toml"
        --model.client.base-url "$INFER_URL"
        --model.client.admin-base-url "${ADMIN_URLS[@]}"
    )
    (( USE_NCCL_BROADCAST == 0 )) || orchestrator_args+=(--weight_broadcast.host "$MASTER_ADDR")
    (( USE_ZMQ_TRANSPORT == 0 )) || orchestrator_args+=(--rollout_transport.host "$MASTER_ADDR")
    WANDB_SHARED_LABEL=orchestrator orchestrator "${orchestrator_args[@]}"
) 2>&1 | tee "$LOG_DIR/orchestrator.log" &
ORCHESTRATOR_PID=$!

set +e
completed_pid=
critical=("$TRAINER_PID" "$ORCHESTRATOR_PID" "$GATEWAY_PID" "$ROUTER_PID" "$PEER_WATCH_PID" "${INFERENCE_PIDS[@]}")
wait -n -p completed_pid "${critical[@]}"
status=$?
if [[ "$completed_pid" != "$TRAINER_PID" && "$completed_pid" != "$ORCHESTRATOR_PID" ]]; then
    (( status != 0 )) || status=91
elif (( status == 0 )); then
    if [[ "$completed_pid" == "$TRAINER_PID" ]]; then
        wait "$ORCHESTRATOR_PID"
    else
        wait "$TRAINER_PID"
    fi
    status=$?
fi
set -e
exit "$status"
