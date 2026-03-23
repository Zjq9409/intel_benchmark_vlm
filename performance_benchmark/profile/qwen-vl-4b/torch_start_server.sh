GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"

export TP=1
export MODEL_PATH="/llm/models/Qwen3-VL-4B-Instruct"
export MODEL_NAME="Qwen3-VL-4B-Instruct"
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Usage: bash torch_start_server.sh [--profile|--no-profile]
# Default: profiling disabled
ENABLE_PROFILE=0
for arg in "$@"; do
    case "$arg" in
        --profile) ENABLE_PROFILE=1 ;;
        --no-profile) ENABLE_PROFILE=0 ;;
    esac
done

PROFILE_BASE_DIR="/llm/performance_benchmark/profile/qwen-vl-4b"
PROFILE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROFILE_DIR="${PROFILE_BASE_DIR}/${PROFILE_TIMESTAMP}"

if [ "$ENABLE_PROFILE" = "1" ]; then
    mkdir -p "$PROFILE_DIR"
    export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
fi

if [ "$GPU_TYPE" = "XPU" ]; then
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.8 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len 12768 \
    --enforce-eager \
    --block-size 64 \
    -tp=$TP   
else
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.8 \
    --no-enable-prefix-caching \
    --enforce-eager \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len 12768 \
    --block-size 64 \
    -tp=$TP   
fi