#!/bin/bash

# ----------------------------------------------------------------
# Bare-metal guard: if not inside Docker, re-exec inside container
# ----------------------------------------------------------------
if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    if nvidia-smi &>/dev/null; then
        CONTAINER_NAME="vllm-nv-container"
    else
        CONTAINER_NAME="lsv-container"
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
# Select model via first argument: "4b" for Qwen3-VL-4B-Instruct, default is 30B
MODEL_SELECT="${1:-30b}"

if [ "$MODEL_SELECT" = "4b" ]; then
    SERVER_MODEL="/llm/models/Qwen3-VL-4B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-4B-Instruct"
    TP=1
else
    SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
    TP=4
fi

PORT=8006
MAX_BATCHED_TOKENS=8192
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.8
MM_W=224
MM_H=224
INPUT_LEN=1024
OUTPUT_LEN=1024

# Setup logging
mkdir -p $SERVER_MODEL_NAME
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
LOG_FILE="${SERVER_MODEL_NAME}/${CURRENT_TIME}_client_tp${TP}_mbt${MAX_BATCHED_TOKENS}_${MM_W}x${MM_H}_in${INPUT_LEN}_out${OUTPUT_LEN}_${GPU_TYPE}.log"
SERVER_LOG="${SERVER_MODEL_NAME}/${CURRENT_TIME}_server_tp${TP}_mbt${MAX_BATCHED_TOKENS}_${MM_W}x${MM_H}_in${INPUT_LEN}_out${OUTPUT_LEN}_${GPU_TYPE}.log"

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
    export NCCL_P2P_LEVEL=SYS
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
if [ "$GPU_TYPE" = "XPU" ]; then
    MAX_BSIZE=150
elif [ "$GPU_TYPE" = "RTX5090" ]; then
    if [ "$MODEL_SELECT" = "4b" ]; then
        MAX_BSIZE=130
    else
        MAX_BSIZE=300
    fi
else
    MAX_BSIZE=200
fi
MM_BUCKET_CONFIG="{(${MM_W},${MM_H}, 1): 1.0}"

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
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --random-mm-base-items-per-request 1 \
        --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
        --random-mm-bucket-config "$MM_BUCKET_CONFIG" \
        --request-rate inf \
        --backend openai-chat \
        --ignore-eos \
        --port=$PORT \
        --seed 42 2>&1 | tee -a "$LOG_FILE"
}

run_benchmark 1
i=2
while [ $i -le $MAX_BSIZE ]; do
    run_benchmark $i
    i=$((i + 2))
done

echo "All benchmark runs finished. Stopping server..."
kill $SERVER_PID
echo "Done."

# Parse log and generate CSV
echo "Parsing log and generating CSV..."
python3 "$(dirname "$0")/parse_log.py" "$LOG_FILE"
echo "CSV saved."
