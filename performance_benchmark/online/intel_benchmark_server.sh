#!/bin/bash

CONTAINER_NAME="lsv-container"
# Host-to-container path mapping (from setup_env.sh volume mounts)
HOST_WEIGHTS_PREFIX="/home/intel/llm_test/weights"
CONTAINER_WEIGHTS_PREFIX="/llm/models"
CONTAINER_LOG_DIR="/llm/intel_benchmark_vlm/performance_benchmark/online/LOG"

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

# Ensure container exists then always restart it for a clean state
echo "Checking container '$CONTAINER_NAME'..."
CONTAINER_STATE=$(sudo docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)
if [ -z "$CONTAINER_STATE" ]; then
    echo "ERROR: Container '$CONTAINER_NAME' does not exist. Run setup_env.sh first."
    exit 1
fi
echo "Restarting container '$CONTAINER_NAME' (current state: $CONTAINER_STATE)..."
sudo docker restart "$CONTAINER_NAME"
sleep 5
CONTAINER_STATE=$(sudo docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)
if [ "$CONTAINER_STATE" != "running" ]; then
    echo "ERROR: Failed to restart container '$CONTAINER_NAME'."
    exit 1
fi
echo "Container '$CONTAINER_NAME' restarted successfully."

# Convert host MODEL_PATH to container MODEL_PATH if needed
if [[ "$MODEL_PATH" == ${HOST_WEIGHTS_PREFIX}* ]]; then
    CONTAINER_MODEL_PATH="${CONTAINER_WEIGHTS_PREFIX}${MODEL_PATH#${HOST_WEIGHTS_PREFIX}}"
else
    CONTAINER_MODEL_PATH="$MODEL_PATH"
fi

# Setup logging
mkdir -p LOG
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
# Try to detect Intel GPU type inside container
GPU_TYPE=$(sudo docker exec "$CONTAINER_NAME" bash -c "xpu-smi discovery 2>/dev/null | grep -oP 'GPU [0-9]+' | head -1 | tr ' ' '_'" 2>/dev/null)
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
LOG_FILE="LOG/benchmark_${MODEL_NAME}_tp${TP}_${GPU_TYPE}_${CURRENT_TIME}.log"
SERVER_LOG="LOG/server_${MODEL_NAME}_tp${TP}_${GPU_TYPE}_${CURRENT_TIME}.log"
CONTAINER_SERVER_LOG="${CONTAINER_LOG_DIR}/server_${MODEL_NAME}_tp${TP}_${GPU_TYPE}_${CURRENT_TIME}.log"

echo "Test results will be saved to: $LOG_FILE"
echo "Server log will be saved to:   $SERVER_LOG"

echo "---------------------------------------------------"
echo "Starting Auto Test with the following parameters:"
echo "Container:              $CONTAINER_NAME"
echo "Model Path (host):      $MODEL_PATH"
echo "Model Path (container): $CONTAINER_MODEL_PATH"
echo "Model Name:             $MODEL_NAME"
echo "TP:                     $TP"
echo "Image Dir:              $IMAGE_DIR"
echo "GPU Type:               $GPU_TYPE"
echo "---------------------------------------------------"

# Kill any existing vllm processes inside container
echo "Killing any existing vllm processes inside container '$CONTAINER_NAME'..."
VLLM_PIDS=$(sudo docker exec "$CONTAINER_NAME" bash -c "pgrep -f 'vllm' 2>/dev/null" 2>/dev/null)
if [ -n "$VLLM_PIDS" ]; then
    echo "Killing vllm processes: $VLLM_PIDS"
    sudo docker exec "$CONTAINER_NAME" bash -c "pgrep -f 'vllm' | xargs -r kill -9" 2>/dev/null
    sleep 3
else
    echo "No existing vllm processes found in container."
fi

# Kill any process on port 8000 inside container
echo "Checking and killing any process on port 8000 inside container..."
CPID=$(sudo docker exec "$CONTAINER_NAME" bash -c "ss -lptn 'sport = :8000' 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2" 2>/dev/null)
if [ -n "$CPID" ]; then
    echo "Killing process $CPID on port 8000 inside container"
    sudo docker exec "$CONTAINER_NAME" bash -c "echo '$CPID' | xargs -r kill -9" 2>/dev/null
fi
sleep 2

# Start vllm server inside container
echo "Starting vllm server inside container '$CONTAINER_NAME'..."
sudo docker exec -d "$CONTAINER_NAME" bash -c "
    export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    mkdir -p '${CONTAINER_LOG_DIR}'
    nohup python3 -m vllm.entrypoints.openai.api_server \\
        --model '${CONTAINER_MODEL_PATH}' \\
        --served-model-name '${MODEL_NAME}' \\
        --dtype=float16 \\
        --enforce-eager \\
        --port 8000 \\
        --host 0.0.0.0 \\
        --trust-remote-code \\
        --gpu-memory-util=0.9 \\
        --no-enable-prefix-caching \\
        --max-num-batched-tokens=8192 \\
        --disable-log-requests \\
        --max-model-len 32768 \\
        --block-size 64 \\
        --quantization fp8 \\
        -tp=${TP} > '${CONTAINER_SERVER_LOG}' 2>&1
"
echo "Server started inside container (log: $SERVER_LOG)"

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
    sudo docker exec "$CONTAINER_NAME" bash -c "pgrep -f 'api_server' | xargs -r kill -9" 2>/dev/null
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
        --ignore-eos | tee -a "$LOG_FILE"
}

# Run tests
MAX_BSIZE=80
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
sudo docker exec "$CONTAINER_NAME" bash -c "pgrep -f 'api_server' | xargs -r kill -9" 2>/dev/null
echo "Done."

# Parse log and generate CSV
echo "Parsing log and generating CSV..."
python3 "$(dirname "$0")/parse_log.py" "$LOG_FILE"
echo "CSV saved."
