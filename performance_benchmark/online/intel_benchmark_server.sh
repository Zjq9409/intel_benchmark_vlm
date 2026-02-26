#!/bin/bash

# Check input parameters
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <modelpath> <modelname> [tp] [image_dir]"
    echo "Example: $0 /llm/models/DeepSeek-R1-Distill-Qwen-32B DeepSeek-R1-Distill-Qwen-32B 4 ../dataset/images"
    exit 1
fi

MODEL_PATH=$1
MODEL_NAME=$2
TP=${3:-4}
IMAGE_DIR=${4:-$(dirname "$0")/../dataset/images}
PROMPT_DIR=$(dirname "$0")/..

# Setup logging
mkdir -p LOG
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="LOG/${MODEL_NAME}_${CURRENT_TIME}.log"
SERVER_LOG="./LOG/${MODEL_NAME}_tp${TP}_server_${CURRENT_TIME}.log"
echo "Test results will be saved to: $LOG_FILE"
echo "Server log will be saved to: $SERVER_LOG"

echo "---------------------------------------------------"
echo "Starting Auto Test with the following parameters:"
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "TP:         $TP"
echo "Image Dir:  $IMAGE_DIR"
echo "---------------------------------------------------"

# Kill any process running on port 8000
echo "Checking and killing any process on port 8000..."
# Try using netstat first
PID=$(netstat -nlp 2>/dev/null | grep :8000 | awk '{print $7}' | cut -d'/' -f1)
# If netstat didn't find it (or command missing), try ss
if [ -z "$PID" ]; then
    PID=$(ss -lptn 'sport = :8000' 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)
fi

if [ -n "$PID" ]; then
    echo "Killing process $PID on port 8000"
    # Use xargs to handle potential multiple PIDs (e.g. ipv4/ipv6)
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

    echo ">>> Running benchmark: bsize=$bsize input=$INPUT_LEN output=$OUTPUT_LEN" | tee -a "$LOG_FILE"
    vllm bench serve \
        --model "$MODEL_PATH" \
        --dataset-name random \
        --served-model-name "$MODEL_NAME" \
        --random-input-len=$INPUT_LEN \
        --random-output-len=$OUTPUT_LEN \
        --ignore-eos \
        --num-prompt $bsize \
        --trust-remote-code \
        --request-rate inf \
        --backend vllm \
        --port=8000 | tee -a "$LOG_FILE"
}

# Run tests
MAX_BSIZE=128
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
