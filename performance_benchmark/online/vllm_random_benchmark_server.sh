#!/bin/bash

# ----------------------------------------------------------------
# Bare-metal guard: if not inside Docker, re-exec inside container
# ----------------------------------------------------------------
if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    if nvidia-smi &>/dev/null; then
        CONTAINER_NAME="vllm-nv-container"
    else
        CONTAINER_NAME="lsv-container-b8"
    fi

    SCRIPT_IN_CONTAINER="/llm/performance_benchmark/online/$(basename "$0")"

    _SELF_DIR="$(dirname "$(realpath "$0")")"
    _WEIGHTS_DIR="${WEIGHTS_DIR:-$(dirname "$(dirname "$(dirname "$_SELF_DIR")")")/weights}"
    _WEIGHTS_DIR="$(realpath "$_WEIGHTS_DIR" 2>/dev/null || echo "${_WEIGHTS_DIR%/}")"

    TRANSLATED_ARGS=()
    for arg in "$@"; do
        arg_real="$(realpath "$arg" 2>/dev/null || echo "${arg%/}")"
        if [[ "$arg_real" == "$_WEIGHTS_DIR"* ]]; then
            rel="${arg_real#$_WEIGHTS_DIR}"
            arg="/llm/models${rel}"
        fi
        TRANSLATED_ARGS+=("$arg")
    done

    CONTAINER_STATE=$(sudo docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)
    if [ -z "$CONTAINER_STATE" ]; then
        echo "ERROR: Container '$CONTAINER_NAME' does not exist. Run setup_env.sh first."
        exit 1
    fi
    # echo "Bare-metal detected -- restarting container '$CONTAINER_NAME' to kill stale processes..."
    # sudo docker restart "$CONTAINER_NAME"
    # echo "Waiting for container to be ready..."
    # sleep 5
    # echo "Bare-metal detected -- re-executing inside container '$CONTAINER_NAME'..."
    exec sudo docker exec -it "$CONTAINER_NAME" bash "$SCRIPT_IN_CONTAINER" "${TRANSLATED_ARGS[@]}"
fi

# Ensure we run from the script's own directory so LOG/ lands in the right place
cd "$(dirname "$(realpath "$0")")"

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
PORT=8006
TP=4
MAX_BATCHED_TOKENS=8192
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.8

# Setup logging
mkdir -p $SERVER_MODEL_NAME
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
LOG_FILE="${SERVER_MODEL_NAME}/client_${CURRENT_TIME}_tp${TP}_mbt${MAX_BATCHED_TOKENS}_${GPU_TYPE}.log"
SERVER_LOG="${SERVER_MODEL_NAME}/server_${CURRENT_TIME}_tp${TP}_mbt${MAX_BATCHED_TOKENS}_${GPU_TYPE}.log"

echo "Test results will be saved to: $LOG_FILE"
echo "Server log will be saved to:   $SERVER_LOG"

echo "---------------------------------------------------"
echo "Starting ShareGPT Benchmark with the following parameters:"
echo "Server Model Path:  $SERVER_MODEL"
echo "Server Model Name:  $SERVER_MODEL_NAME"
echo "Port:               $PORT"
echo "TP:                 $TP"
echo "Max Batched Tokens: $MAX_BATCHED_TOKENS"
echo "Max Model Len:      $MAX_MODEL_LEN"
echo "GPU Mem Util:       $GPU_MEM_UTIL"
echo "GPU Type:           $GPU_TYPE"
echo "---------------------------------------------------"


# Start vllm server
echo "Starting vllm server..."

VLLM_SERVER_ARGS=(
    --model "$SERVER_MODEL"
    --served-model-name "$SERVER_MODEL_NAME"
    --allowed-local-media-path /llm/models
    --dtype=float16
    --port $PORT
    --host 0.0.0.0
    --trust-remote-code
    --disable-sliding-window
    --no-enable-prefix-caching
    --gpu-memory-util=$GPU_MEM_UTIL
    --max-num-batched-tokens=$MAX_BATCHED_TOKENS
    --limit-mm-per-prompt '{"image": 1}'
    --disable-log-requests
    --max-model-len=$MAX_MODEL_LEN
    --block-size 64
    --quantization fp8
    -tp=$TP
)

if [ "$GPU_TYPE" = "XPU" ]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1   
    VLLM_SERVER_ARGS+=(--enforce-eager)
else
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
fi

nohup vllm serve "${VLLM_SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
MAX_RETRIES=120
COUNT=0
SERVER_READY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
        SERVER_READY=1; break
    fi
    sleep 5
    COUNT=$((COUNT+1))
    echo "Waiting for server... ($COUNT/$MAX_RETRIES)"
done

if [ $SERVER_READY -eq 0 ]; then
    echo "Server failed to start within timeout. Checking logs:"
    tail -n 20 "$SERVER_LOG"
    kill $SERVER_PID
    exit 1
fi

# Run benchmarks
[ "$GPU_TYPE" = "XPU" ] && MAX_BSIZE=30 || MAX_BSIZE=40

run_benchmark() {
    local bsize=$1
    echo ">>> Running vllm bench serve with --num-prompts=$bsize" | tee -a "$LOG_FILE"
    vllm bench serve \
        --model "$SERVER_MODEL" \
        --served-model-name "$SERVER_MODEL_NAME" \
        --endpoint /v1/chat/completions \
        --dataset-name random-mm \
        --num-prompts $bsize \
        --max-concurrency $bsize \
        --ready-check-timeout-sec 1 --num-warmups 1 \
        --random-input-len 128 \
        --random-output-len 128 \
        --random-mm-base-items-per-request 1 \
        --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
        --random-mm-bucket-config '{(224,224, 1): 1.0}' \
        --request-rate inf \
        --backend openai-chat \
        --ignore-eos \
        --port=$PORT \
        --seed 42 2>&1 | tee -a "$LOG_FILE"
}

run_benchmark 1
for (( i=2; i<=MAX_BSIZE; i+=2 )); do
    run_benchmark $i
done

echo "All benchmark runs finished. Stopping server..."
kill $SERVER_PID
echo "Done."
