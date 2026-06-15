GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"

# Usage: bash start_vlm_server.sh [--fp8] [--ep]
#   --fp8 : fp8 quantization (GPU only)
#   --ep  : Expert Parallel mode — DP=2 TP=2, AllToAll MoE (~4.1GB KV, max-model-len=2048)
#   default: TP=4 AllReduce MoE (~11.7GB KV, max-model-len=12768)
ENABLE_FP8=1
ENABLE_EP=0
for arg in "$@"; do
    case "$arg" in
        --fp8) ENABLE_FP8=1 ;;
        --ep)  ENABLE_EP=1 ;;
    esac
done

FP8_FLAG=""
[ "$ENABLE_FP8" = "1" ] && FP8_FLAG="--quantization fp8"

# if [ "$ENABLE_EP" = "1" ]; then
#     export DP=2; export TP=2; MAX_MODEL_LEN=2048
#     echo "[mode] EP (AllToAll): DP=$DP TP=$TP max-model-len=$MAX_MODEL_LEN"
# else
#     export DP=1; export TP=4; MAX_MODEL_LEN=32768
#     echo "[mode] TP-only (AllReduce): TP=$TP max-model-len=$MAX_MODEL_LEN"
# fi

TP=2
MAX_MODEL_LEN=12768

export MODEL_PATH="/llm/models/Qwen3-VL-30B-A3B-Instruct"
export MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"

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
    # DP_OPT=""
    # EP_OPT=""
    # if [ "$ENABLE_EP" = "1" ]; then
    #     [ "$DP" -gt 1 ] && DP_OPT="--data-parallel-size $DP"
    #     EP_OPT="--enable-expert-parallel"
    # fi
    export VLLM_TORCH_PROFILER_DIR=/llm/profile/qwen-vl-30b-a3b/profile_tp2

    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --dtype=float16 \
        --port 8007 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --gpu-memory-util=0.85 \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --no-enable-prefix-caching \
        --mm-processor-cache-gb 0 \
        --async-scheduling \
        --max-model-len $MAX_MODEL_LEN \
        --block-size 64 \
        --max-num-seqs 32 \
        -tp=$TP \
        $FP8_FLAG \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/llm/profile/qwen-vl-30b-a3b/profile_tp2"}' \
        --enforce-eager 

        
fi
