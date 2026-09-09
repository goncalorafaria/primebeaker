#!/usr/bin/env bash
# One replicated Beaker task per SFT node. Shared Weka files provide discovery
# and failure propagation; torchrun owns the actual distributed process group.

set -Eeuo pipefail

: "${BEAKER_REPLICA_RANK:?missing Beaker replica rank}"
required=(TOTAL_NODES GPUS_PER_NODE CONFIG_PATH OUTPUT_DIR WORKING_DIR SCRATCH LAUNCH_ID RENDEZVOUS_DIR RENDEZVOUS_TIMEOUT_SECONDS)
for name in "${required[@]}"; do
    [[ -n "${!name:-}" ]] || { echo "missing required environment variable: $name" >&2; exit 2; }
done

RANK=$BEAKER_REPLICA_RANK
(( RANK >= 0 && RANK < TOTAL_NODES )) || { echo "invalid node rank $RANK" >&2; exit 2; }
RUNTIME_DIR="$RENDEZVOUS_DIR/runtime"
RUNTIME_CONFIG="$OUTPUT_DIR/configs/sft.toml"
LOG_DIR="$OUTPUT_DIR/logs/trainer"
mkdir -p "$RENDEZVOUS_DIR" "$RUNTIME_DIR" "$LOG_DIR" /tmp/beaker-result

atomic_write() {
    local path=$1 value=$2 temporary="${1}.tmp.${RANK}.$$"
    printf '%s\n' "$value" > "$temporary"
    mv "$temporary" "$path"
}

on_exit() {
    local code=$?
    trap - EXIT INT TERM
    set +e
    if (( code != 0 )) && [[ ! -s "$RENDEZVOUS_DIR/shutdown.requested" ]]; then
        atomic_write "$RENDEZVOUS_DIR/shutdown.requested" "SFT node $RANK exited with code $code"
    fi
    atomic_write "$RENDEZVOUS_DIR/status.$RANK" "$code sft"
    local pids
    pids=$(jobs -pr)
    [[ -z "$pids" ]] || { kill -TERM $pids 2>/dev/null; wait $pids 2>/dev/null; }
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

cd "$WORKING_DIR"
if [[ -n "${PRIMEBEAKER_SETUP_COMMAND:-}" ]]; then
    bash -lc "$PRIMEBEAKER_SETUP_COMMAND"
fi
command -v torchrun >/dev/null
python3 -c 'import prime_rl, primebeaker'

export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache}
export HF_HOME=${HF_HOME:-$XDG_CACHE_HOME/huggingface}
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export TMPDIR=${TMPDIR:-/tmp/primebeaker-sft-$LAUNCH_ID-$RANK}
mkdir -p "$OUTPUT_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
ulimit -n 65536

LOCAL_HOST=$(hostname -f 2>/dev/null || hostname)
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
LOCAL_IP=${LOCAL_IP:-$LOCAL_HOST}
atomic_write "$RENDEZVOUS_DIR/host.$RANK" "$LOCAL_IP"
for ((peer = 0; peer < TOTAL_NODES; peer++)); do
    wait_for_file "$RENDEZVOUS_DIR/host.$peer" "node $peer hostname"
done
MASTER_ADDR=$(<"$RENDEZVOUS_DIR/host.0")

if (( RANK == 0 )); then
    python3 -m primebeaker.multinode prepare-sft \
        --config "$CONFIG_PATH" \
        --output "$RUNTIME_CONFIG" \
        --runtime-dir "$RUNTIME_DIR"
fi
wait_for_file "$RUNTIME_DIR/runtime.ready" "resolved SFT config"
# shellcheck disable=SC1091
source "$RUNTIME_DIR/runtime.env"
# shellcheck disable=SC1091
source "$RUNTIME_DIR/trainer.env"

if command -v ibv_devinfo >/dev/null 2>&1; then
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[:blank:]' | paste -sd, || true)
    [[ -z "$IB_HCA" ]] || export NCCL_IB_HCA=$IB_HCA
fi
if command -v ip >/dev/null 2>&1; then
    GLOO_IFNAME=$(ip -o -4 addr show to "$LOCAL_IP" 2>/dev/null | awk '{print $2; exit}' || true)
    [[ -z "$GLOO_IFNAME" ]] || export GLOO_SOCKET_IFNAME=$GLOO_IFNAME
fi

torchrun \
    --role=trainer \
    --nnodes="$TOTAL_NODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --rdzv-endpoint="$MASTER_ADDR:29500" \
    --rdzv-id="primebeaker_$LAUNCH_ID" \
    --log-dir="$LOG_DIR/torchrun" \
    --tee=3 \
    --redirects=3 \
    --local-ranks-filter="$TRAINER_RANKS_FILTER" \
    -m prime_rl.trainer.sft.train \
    @ "$RUNTIME_CONFIG" \
    2>&1 | sed -u 's/^\[[a-zA-Z]*[0-9]*\]://' | tee -a "$LOG_DIR/node_${RANK}.log"
