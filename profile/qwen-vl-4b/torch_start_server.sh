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

export TP=1
export MODEL_PATH="/llm/models/Qwen3-VL-4B-Instruct"
export MODEL_NAME="Qwen3-VL-4B-Instruct"
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

if [ "$GPU_TYPE" = "XPU" ]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1  
    CUDA_VISIBLE_DEVICES=4 VLLM_TORCH_PROFILER_DIR=./profile python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.8 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --max-model-len 12768 \
    --async-scheduling \
    --block-size 64 \
    $FP8_FLAG \
    -tp=$TP   
else
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export NCCL_P2P_LEVEL=SYS
    CUDA_VISIBLE_DEVICES=4 VLLM_TORCH_PROFILER_DIR=./profile  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.8 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --max-model-len 12768 \
    --block-size 64 \
    $FP8_FLAG \
    -tp=$TP   \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./profile"}'
    # --enforce-eager 
fi
