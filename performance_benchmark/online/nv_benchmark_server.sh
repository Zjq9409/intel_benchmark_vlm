
# python vlm_benchmark.py \
# --prompt "$(cat /home/intel/llm_test/intel_benchmark_vlm/performance_benchmark/prompt_128.txt)" \
# --model  /home/intel/llm_test/weights/Qwen3-VL-4B-Instruct/ \
# --served-model-name Qwen3-VL-4B-Instruct \
# --batch_size 3 \
# --port 8000 \
# --host 127.0.0.1 \
# --ignore-eos \
# --output_len 128 

#!/bin/bash

# Check input parameters
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <modelpath> <modelname> [tp] [image_dir]"
    echo "Example: $0 /home/sdf/jane/weights/Qwen3-VL-30B-A3B-Instruct Qwen3-VL-30B-A3B-Instruct 4 ../dataset/images"
    exit 1
fi

MODEL_PATH=$1
MODEL_NAME=$2
TP=${3:-4}
IMAGE_DIR=${4:-$(dirname "$0")/../dataset/images}
PROMPT_DIR=$(dirname "$0")/..
#INPUT_LEN=$5
#OUTPUT_LEN=$5
#MAX_BSIZE=$6
unset http_proxy
unset https_proxy
export NCCL_P2P_LEVEL=SYS

# Auto-set CUDA_VISIBLE_DEVICES based on TP value
CUDA_DEVICES=""
for (( g=0; g<TP; g++ )); do
    [ -n "$CUDA_DEVICES" ] && CUDA_DEVICES="${CUDA_DEVICES},"
    CUDA_DEVICES="${CUDA_DEVICES}${g}"
done
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES (tp=$TP)"

# Setup logging
mkdir -p LOG
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
#LOG_FILE="LOG/${MODEL_NAME}_in${INPUT_LEN}_out${OUTPUT_LEN}_${CURRENT_TIME}.log"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="unknown_gpu"
LOG_FILE="LOG/${MODEL_NAME}_${GPU_TYPE}_${CURRENT_TIME}.log"
echo "Test results will be saved to: $LOG_FILE"

echo "---------------------------------------------------"
echo "Starting Auto Test with the following parameters:"
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Input Len:  $INPUT_LEN"
echo "Output Len: $OUTPUT_LEN"
echo "Max Bsize:  $MAX_BSIZE"
echo "---------------------------------------------------"

# Kill any existing vllm / python vllm processes
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
# Try using netstat first
PID=$(netstat -nlp 2>/dev/null | grep :8000 | awk '{print $7}' | cut -d'/' -f1)
# If netstat didn't find it (or command missing), try ss
if [ -z "$PID" ]; then
    PID=$(ss -lptn 'sport = :8000' 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)
fi

if [ -n "$PID" ]; then
    echo "Killing process $PID on port 8000"
    echo "$PID" | xargs -r kill -9
fi
sleep 2

# Export environment variables
# Start vllm server
echo "Starting vllm server..."
SERVER_LOG="./LOG/${MODEL_NAME}_tp${TP}_${GPU_TYPE}_server_${CURRENT_TIME}.log"
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
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
MAX_RETRIES=1200
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
    python vlm_benchmark.py \
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
