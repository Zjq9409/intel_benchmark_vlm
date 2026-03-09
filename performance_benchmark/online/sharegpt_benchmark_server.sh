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
    echo "Bare-metal detected -- restarting container '$CONTAINER_NAME' to kill stale processes..."
    sudo docker restart "$CONTAINER_NAME"
    echo "Waiting for container to be ready..."
    sleep 5
    echo "Bare-metal detected -- re-executing inside container '$CONTAINER_NAME'..."
    exec sudo docker exec -it "$CONTAINER_NAME" bash "$SCRIPT_IN_CONTAINER" "${TRANSLATED_ARGS[@]}"
fi

# Ensure we run from the script's own directory so LOG/ lands in the right place
cd "$(dirname "$(realpath "$0")")"

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
# DATASET_PATH="/llm/models/sharegpt4v_instruct_gpt4-vision_cap100k.json"
PORT=8006
TP=4

# Setup logging
mkdir -p LOG
mkdir -p "$RESULT_DIR"
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
LOG_FILE="LOG/sharegpt_benchmark_${SERVER_MODEL_NAME}_tp${TP}_${GPU_TYPE}_${CURRENT_TIME}.log"
SERVER_LOG="LOG/sharegpt_server_${SERVER_MODEL_NAME}_tp${TP}_${GPU_TYPE}_${CURRENT_TIME}.log"

echo "Test results will be saved to: $LOG_FILE"
echo "Server log will be saved to:   $SERVER_LOG"

echo "---------------------------------------------------"
echo "Starting ShareGPT Benchmark with the following parameters:"
echo "Server Model Path: $SERVER_MODEL"
echo "Server Model Name: $SERVER_MODEL_NAME"
# echo "Dataset:           $DATASET_PATH"
echo "Port:              $PORT"
echo "TP:                $TP"
echo "GPU Type:          $GPU_TYPE"
echo "---------------------------------------------------"

# Kill any existing vllm python processes
echo "Killing any existing vllm python processes..."
VLLM_PIDS=$(pgrep -f "vllm" 2>/dev/null)
if [ -n "$VLLM_PIDS" ]; then
    echo "Killing vllm processes: $VLLM_PIDS"
    echo "$VLLM_PIDS" | xargs -r kill -9
    sleep 3
else
    echo "No existing vllm processes found."
fi

# Kill any process running on the server port
echo "Checking and killing any process on port $PORT..."
PID=$(ss -lptn "sport = :${PORT}" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)
if [ -n "$PID" ]; then
    echo "Killing process $PID on port $PORT"
    echo "$PID" | xargs -r kill -9
fi
sleep 2

# Export environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Start vllm server
echo "Starting vllm server..."

MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.8
MAX_BATCHED_TOKENS=8192
if [ "$GPU_TYPE" = "XPU" ]; then
    nohup vllm serve \
    --model "$SERVER_MODEL" \
    --served-model-name "$SERVER_MODEL_NAME" \
    --allowed-local-media-path /llm/models \
    --dtype=float16 \
    --enforce-eager \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --gpu-memory-util=$GPU_MEM_UTIL \
    --max-num-batched-tokens=$MAX_BATCHED_TOKENS \
    --limit-mm-per-prompt '{"image": 1}' \
    --disable-log-requests \
    --max-model-len=$MAX_MODEL_LEN  \
    --no-enable-prefix-caching \
    --block-size 64 \
    --quantization fp8 \
    -tp=$TP > "$SERVER_LOG" 2>&1 &
else
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    nohup vllm serve \
    --model "$SERVER_MODEL" \
    --served-model-name "$SERVER_MODEL_NAME" \
    --allowed-local-media-path /llm/models \
    --dtype=float16 \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --disable-sliding-window \
    --no-enable-prefix-caching \
    --gpu-memory-util=$GPU_MEM_UTIL \
    --max-num-batched-tokens=$MAX_BATCHED_TOKENS \
    --disable-log-requests \
    --max-model-len=$MAX_MODEL_LEN \
    --block-size 64 \
    --quantization fp8 \
    -tp=$TP > "$SERVER_LOG" 2>&1 &
fi


SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
MAX_RETRIES=120
COUNT=0
SERVER_READY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
        # curl 成功（HTTP 200）→ 服务器就绪，退出循环
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

# Run ShareGPT benchmarks
echo "VLM benchmark runs..." | tee -a "$LOG_FILE"

# vllm bench serve \
#         --backend openai-chat \
#         --model "$SERVER_MODEL" \
#         --served-model-name "$SERVER_MODEL_NAME" \
#         --dataset-name sharegpt \
#         --dataset-path "$DATASET_PATH" \
#         --num-prompts 2   \
#         --endpoint /v1/chat/completions \
#         --port=$PORT
if [ "$GPU_TYPE" = "XPU" ]; then
    MAX_BSIZE=30
else
    MAX_BSIZE=50
fi     
for (( i=1; i<=MAX_BSIZE; i+=2 )); do
    # echo ">>> Running vllm bench serve with --num-prompts=$i" | tee -a "$LOG_FILE"
    vllm bench serve \
            --backend openai-chat \
            --model "$SERVER_MODEL" \
            --served-model-name "$SERVER_MODEL_NAME" \
            --endpoint /v1/chat/completions \
            --dataset-name random-mm \
            --num-prompts 2 \
            --max-concurrency 2 \
            --random-input-len 128 \
            --random-output-len 128 \
            --random-mm-base-items-per-request 1 \
            --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
            --random-mm-bucket-config '{(1920, 1080, 1): 1.0}' \
            --request-rate inf \
            --ignore-eos \
            --port=$PORT \
            --seed 42
done

run_benchmark() {
    local bsize=$1
    echo ">>> Running vllm bench serve with --num-prompts=$bsize" | tee -a "$LOG_FILE"
    # vllm bench serve \
    #     --backend openai-chat \
    #     --model "$SERVER_MODEL" \
    #     --served-model-name "$SERVER_MODEL_NAME" \
    #     --dataset-name sharegpt \
    #     --dataset-path "$DATASET_PATH" \
    #     --num-prompts $bsize \
    #     --endpoint /v1/chat/completions \
    #     --port=$PORT 2>&1 | tee -a "$LOG_FILE"
    vllm bench serve \
        --backend openai-chat \
        --model "$SERVER_MODEL" \
        --served-model-name "$SERVER_MODEL_NAME" \
        --endpoint /v1/chat/completions \
        --dataset-name random-mm \
        --num-prompts $bsize \
        --max-concurrency $bsize \
        --random-input-len 128 \
        --random-output-len 128 \
        --random-mm-base-items-per-request 1 \
        --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
        --random-mm-bucket-config '{(1920, 1080, 1): 1.0}' \
        --request-rate inf \
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
