#!/bin/bash

# ----------------------------------------------------------------
# Bare-metal guard: if not inside Docker, re-exec inside container
# Path translation: host WEIGHTS_DIR -> /llm/models inside container
# ----------------------------------------------------------------
if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    CONTAINER_NAME="lsv-container"
    SCRIPT_IN_CONTAINER="/llm/performance_benchmark/online/$(basename "$0")"

    # Resolve host WEIGHTS_DIR (env var > default ../weights beside intel_benchmark_vlm/)
    _SELF_DIR="$(dirname "$(realpath "$0")")"
    _WEIGHTS_DIR="${WEIGHTS_DIR:-$(dirname "$(dirname "$(dirname "$_SELF_DIR")")")/weights}"
    _WEIGHTS_DIR="$(realpath "$_WEIGHTS_DIR" 2>/dev/null || echo "${_WEIGHTS_DIR%/}")"

    # Translate each argument: replace host WEIGHTS_DIR prefix with /llm/models
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
    echo "  Host WEIGHTS_DIR : $_WEIGHTS_DIR  ->  /llm/models (in container)"
    for i in "${!TRANSLATED_ARGS[@]}"; do
        [ "${TRANSLATED_ARGS[$i]}" != "${*:$((i+1)):1}" ] && \
            echo "  arg$((i+1)) translated: ${*:$((i+1)):1}  ->  ${TRANSLATED_ARGS[$i]}"
    done
    exec sudo docker exec -it "$CONTAINER_NAME" bash "$SCRIPT_IN_CONTAINER" "${TRANSLATED_ARGS[@]}"
fi

# Ensure we run from the script's own directory so LOG/ lands in the right place
cd "$(dirname "$(realpath "$0")")"

# Check input parameters
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <modelpath> [tp] [image_dir]"
    echo "  modelpath: host or container path to the model"
    echo "Example: $0 /home/intel/llm_test/weights/Qwen3-VL-4B-Instruct 1"
    echo "         $0 /home/intel/llm_test/weights/Qwen3-VL-8B-Instruct 4 ../dataset/images"
    exit 1
fi

MODEL_PATH=$1
# If the 2nd arg is purely numeric, treat it as TP and infer model name from path
if [[ "${2}" =~ ^[0-9]+$ ]]; then
    MODEL_NAME=$(basename "${MODEL_PATH%/}")
    TP=${2:-4}
    IMAGE_DIR=${3:-$(dirname "$0")/../dataset/images}
else
    MODEL_NAME=${2:-$(basename "${MODEL_PATH%/}")}
    TP=${3:-4}
    IMAGE_DIR=${4:-$(dirname "$0")/../dataset/images}
fi
PROMPT_DIR=$(dirname "$0")/..

# Setup logging
mkdir -p LOG
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
GPU_TYPE=$(xpu-smi discovery 2>/dev/null | grep -oP 'GPU [0-9]+' | head -1 | tr ' ' '_')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
LOG_FILE="LOG/benchmark_${MODEL_NAME}_tp${TP}_${GPU_TYPE}_${CURRENT_TIME}.log"
SERVER_LOG="LOG/server_${MODEL_NAME}_tp${TP}_${GPU_TYPE}_${CURRENT_TIME}.log"
echo "Test results will be saved to: $LOG_FILE"
echo "Server log will be saved to:   $SERVER_LOG"

echo "---------------------------------------------------"
echo "Starting Auto Test with the following parameters:"
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "TP:         $TP"
echo "Image Dir:  $IMAGE_DIR"
echo "GPU Type:   $GPU_TYPE"
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

# Kill any process running on port 8000
echo "Checking and killing any process on port 8000..."
PID=$(ss -lptn 'sport = :8000' 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)
if [ -n "$PID" ]; then
    echo "Killing process $PID on port 8000"
    echo "$PID" | xargs -r kill -9
fi
sleep 2

# Export environment variables
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Start vllm server
echo "Starting vllm server..."
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.9 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len 32768 \
    --block-size 64 \
    --quantization fp8 \
    -tp=$TP > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
MAX_RETRIES=120
COUNT=0
SERVER_READY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "Server is ready!"
        SERVER_READY=1
        break
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

# Function to run benchmark
run_benchmark() {
    local bsize=$1
    local INPUT_LEN=$2
    local OUTPUT_LEN=$3

    # Select prompt file based on input length
    if [ "$INPUT_LEN" -le 256 ]; then
        PROMPT_FILE="${PROMPT_DIR}/prompt_128.txt"
    elif [ "$INPUT_LEN" -le 1500 ]; then
        PROMPT_FILE="${PROMPT_DIR}/prompt_1k.txt"
    else
        PROMPT_FILE="${PROMPT_DIR}/prompt_2k.txt"
    fi

    echo ">>> Running vlm_benchmark: bsize=$bsize input=$INPUT_LEN output=$OUTPUT_LEN prompt=$(basename $PROMPT_FILE)" | tee -a "$LOG_FILE"
    python3 vlm_benchmark.py \
        --prompt "$(cat "$PROMPT_FILE")" \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --batch_size "$bsize" \
        --output_len "$OUTPUT_LEN" \
        --image_dir "$IMAGE_DIR" \
        --port 8000 \
        --host 127.0.0.1 \
        --ignore-eos \
        --warmup "$WARMUP" \
        --num_rounds "$NUM_ROUNDS" | tee -a "$LOG_FILE"
}

# Run tests
MAX_BSIZE=60
WARMUP=3        # warmup rounds before measurement (suppressed)
NUM_ROUNDS=1    # use the result of this single run after warmup
for input in 128
do
for output in 128
do
run_benchmark 1 $input $output
for (( i=2; i<=MAX_BSIZE; i+=2 )); do
run_benchmark $i $input $output
done
done
done

echo "Test finished. Stopping server..."
kill $SERVER_PID
echo "Done."

# Parse log and generate CSV
echo "Parsing log and generating CSV..."
python3 "$(dirname "$0")/parse_log.py" "$LOG_FILE"
echo "CSV saved."
