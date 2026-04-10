GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"

# Usage: bash start_vlm_server.sh [--fp8]
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

# DP=1 TP=4 (default, 11.7GB KV cache)
# DP=2 TP=2 (enables All-to-All MoE, but only 4.1GB KV cache — use shorter max-model-len)
export DP=${DP:-1}
export TP=$((4 / DP))
export MODEL_PATH="/llm/models/Qwen3-VL-30B-A3B-Instruct"
export MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Adjust max-model-len based on DP (less KV cache available when DP>1)
if [ "$DP" -gt 1 ]; then
    MAX_MODEL_LEN=4096
else
    MAX_MODEL_LEN=12768
fi

if [ "$GPU_TYPE" = "XPU" ]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1  
    VLLM_TORCH_PROFILER_DIR=./profile python3 -m vllm.entrypoints.openai.api_server \
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
    --max-model-len $MAX_MODEL_LEN \
    --enforce-eager \
    --block-size 64 \
    -tp=$TP   
else
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export NCCL_P2P_LEVEL=SYS
    QUANT_OPT="$FP8_FLAG"
    [ "$DP" -gt 1 ] && DP_OPT="--data-parallel-size $DP" || DP_OPT=""
    VLLM_TORCH_PROFILER_DIR=./profile  python3 -m vllm.entrypoints.openai.api_server \
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
    --max-model-len $MAX_MODEL_LEN \
    --block-size 64 \
    -tp=$TP   \
    $QUANT_OPT \
    $DP_OPT \
    --enable-expert-parallel 
    #--enforce-eager 
fi
