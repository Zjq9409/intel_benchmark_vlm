GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"

# Usage: bash torch_start_server.sh [--fp8]
#   --fp8 : enable fp8 quantization (GPU only, ignored on XPU)
ENABLE_FP8=0
for arg in "$@"; do
    case "$arg" in
        --fp8) ENABLE_FP8=1 ;;
    esac
done
FP8_FLAG=""
if [ "$ENABLE_FP8" = "1" ]; then
    FP8_FLAG="--quantization fp8"
fi

export TP=2
export MODEL_PATH="/llm/models/Qwen3.5-35B-A3B/"
export MODEL_NAME="Qwen3.5-35B-A3B"

if [ "$GPU_TYPE" = "XPU" ]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1  
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.9 \
    --max-num-batched-tokens=8192 \
    --max-model-len 12768 \
    --async-scheduling \
    --block-size 64 \
    $FP8_FLAG \
    -tp=$TP   
else
    export PYTORCH_ALLOC_CONF="expandable_segments:True"
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export NCCL_P2P_LEVEL=SYS
    python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
    --port 8006 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.9 \
    --max-num-batched-tokens=8192 \
    --max-num-seqs 32 \
    --max-model-len 12768 \
    --block-size 64 \
    --quantization fp8 \
    -tp=2   \
    # --enforce-eager 
fi
