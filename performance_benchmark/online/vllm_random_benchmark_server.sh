#!/bin/bash

# ----------------------------------------------------------------
# Bare-metal guard: if not inside Docker, re-exec inside container
# ----------------------------------------------------------------
if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    if nvidia-smi &>/dev/null; then
        CONTAINER_NAME="${VLLM_NV_CONTAINER:-vllm-nv-container}"
    else
        CONTAINER_NAME="${VLLM_XPU_CONTAINER:-lsv-container-b8}"
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
# Select model via first argument: "4b" | "q35-4b" | "30b" (default)
# 4th argument: images per request (default 1; set 10 to simulate NarratoAI multi-image)
MODEL_SELECT="${1:-30b}"
MM_ITEMS="${4:-1}"
MTP="${5:-off}"   # on=enable speculative decoding, off=disable
QUANT="${6:-fp8}"  # fp8=enable fp8 quantization, none=disable
DEVICE="${7:-}"     # GPU device ID, e.g. 4; empty=use all
OUTPUT_LEN="${8:-1024}"  # output token length; 128=realtime, 512=near-realtime, 1024=batch
INPUT_LEN="${9:-1024}"   # input token length; 512=short prompt, 1024=standard

if [ "$MODEL_SELECT" = "4b" ]; then
    SERVER_MODEL="/llm/models/Qwen3-VL-4B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-4B-Instruct"
    TP=1
elif [ "$MODEL_SELECT" = "q35-4b" ]; then
    SERVER_MODEL="/llm/models/Qwen3.5-4B"
    SERVER_MODEL_NAME="Qwen3.5-4B"
    TP=1
elif [ "$MODEL_SELECT" = "32b" ]; then
    SERVER_MODEL="/llm/models/Qwen3-VL-32B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-32B-Instruct"
    TP=4
else
    SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
    TP=4
fi

PORT=8006
# Scale token limits with images-per-request (720P ≈576 visual tokens/image)
if [ "$MM_ITEMS" -gt 1 ]; then
    MAX_BATCHED_TOKENS=32768
    MAX_MODEL_LEN=32768
else
    MAX_BATCHED_TOKENS=8192
    MAX_MODEL_LEN=16384
fi
GPU_MEM_UTIL=0.8
MM_W="${2:-1280}"
MM_H="${3:-720}"

# Setup logging
mkdir -p "$SERVER_MODEL_NAME"
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
MTP_TAG=$([ "$MTP" = "on" ] && echo "mtp_" || echo "nomtp_")
QUANT_TAG=$([ "$QUANT" = "none" ] && echo "fp16_" || echo "${QUANT}_")
DEV_TAG=$([ -n "$DEVICE" ] && echo "dev${DEVICE}_" || echo "")
LOG_FILE="${SERVER_MODEL_NAME}/${CURRENT_TIME}_client_${DEV_TAG}${QUANT_TAG}${MTP_TAG}${MM_ITEMS}_${MM_W}x${MM_H}_tp${TP}_mbt${MAX_BATCHED_TOKENS}_in${INPUT_LEN}_out${OUTPUT_LEN}_${GPU_TYPE}.log"
SERVER_LOG="${SERVER_MODEL_NAME}/${CURRENT_TIME}_server_${DEV_TAG}${QUANT_TAG}${MTP_TAG}${MM_ITEMS}_${MM_W}x${MM_H}_tp${TP}_mbt${MAX_BATCHED_TOKENS}_in${INPUT_LEN}_out${OUTPUT_LEN}_${GPU_TYPE}.log"

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
echo "Images per request: $MM_ITEMS"
echo "Quantization:       $QUANT"
echo "GPU Device:         ${DEVICE:-all}"
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
    --limit-mm-per-prompt '{"image": '"${MM_ITEMS}"'}'
    --max-model-len=$MAX_MODEL_LEN
    --block-size 64
    --mm-processor-cache-gb 0
    --async-scheduling 
    -tp=$TP
)

# Quantization (6th arg: fp8/none, default fp8)
if [ "$QUANT" != "none" ]; then
    VLLM_SERVER_ARGS+=(--quantization "$QUANT")
fi

# MTP speculative decoding (5th arg: on/off)
if [ "$MTP" = "on" ]; then
    VLLM_SERVER_ARGS+=(--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
fi

# Remove the extra closing paren — restore block

if [ "$GPU_TYPE" = "XPU" ]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1   
    if [ "$QUANT" != "none" ]; then
        VLLM_SERVER_ARGS+=(--enforce-eager)
    fi
else
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export NCCL_P2P_LEVEL=SYS
fi

if [ -n "$DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES=$DEVICE
    echo "Using GPU device: $DEVICE"
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
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "$SERVER_MODEL_NAME"; then
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

# Start GPU monitor on XPU
MONITOR_PID=""
if [ "$GPU_TYPE" = "XPU" ]; then
    MONITOR_LOG="${SERVER_MODEL_NAME}/${CURRENT_TIME}_monitor_${DEV_TAG}${QUANT_TAG}${MTP_TAG}${MM_ITEMS}_${MM_W}x${MM_H}_tp${TP}_mbt${MAX_BATCHED_TOKENS}_in${INPUT_LEN}_out${OUTPUT_LEN}_${GPU_TYPE}.log"
    echo "Starting GPU monitor, log: $MONITOR_LOG"
    bash "$(dirname "$0")/monitor_gpu.sh" > "$MONITOR_LOG" 2>&1 &
    MONITOR_PID=$!
    echo "GPU monitor PID: $MONITOR_PID"
fi

# Run benchmarks
# Multi-image: narrow range (1~20); single-image: full sweep (1~200)
if [ "$MM_ITEMS" -gt 1 ]; then
    MAX_BSIZE=20
else
    MAX_BSIZE=200
fi

MM_BUCKET_CONFIG="{(${MM_H},${MM_W}, 1): 1.0}"

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
        --random-mm-base-items-per-request $MM_ITEMS \
        --random-mm-limit-mm-per-prompt '{"image": '"${MM_ITEMS}"', "video": 0}' \
        --random-mm-bucket-config "$MM_BUCKET_CONFIG" \
        --request-rate inf \
        --backend openai-chat \
        --ignore-eos \
        --port=$PORT \
        --seed 42 2>&1 | tee -a "$LOG_FILE"
}

check_stop() {
    # 用 Benchmark Duration (s) 作为 E2E Latency（最准确）
    local e2e_s e2e_ms e2e_limit
    e2e_s=$(grep 'Benchmark duration (s):' "$LOG_FILE" | tail -1 | awk '{print $NF}')
    [ -z "$e2e_s" ] && return
    e2e_ms=$(awk -v v="$e2e_s" 'BEGIN { printf "%.0f", v * 1000 }')
    # 停止阈值：单图 60s，多图 120s
    e2e_limit=$([ "$MM_ITEMS" -gt 1 ] && echo 120000 || echo 60000)
    echo "  E2E: ${e2e_s}s (limit: $(( e2e_limit / 1000 ))s)"
    if awk "BEGIN { exit !(${e2e_ms} > ${e2e_limit}) }"; then
        echo "E2E ${e2e_s}s exceeds $(( e2e_limit / 1000 ))s threshold. Stopping server..."
        [ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null
        kill $SERVER_PID
        echo "Done."
        echo "Parsing log and generating CSV..."
        python3 "$(dirname "$0")/parse_log.py" "$LOG_FILE"
        echo "CSV saved."
        exit 0
    fi
}


run_benchmark 1
check_stop
i=1
# STEP=$([ "$MM_ITEMS" -gt 1 ] && echo 1 || echo 2)
STEP=10
while [ $i -le $MAX_BSIZE ]; do
    run_benchmark $i
    check_stop
    i=$((i + STEP))
done

echo "All benchmark runs finished. Stopping server..."
[ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null
kill $SERVER_PID
echo "Done."

# Parse log and generate CSV
echo "Parsing log and generating CSV..."
python3 "$(dirname "$0")/parse_log.py" "$LOG_FILE"
echo "CSV saved."
