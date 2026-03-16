MAX_MODEL_LEN=16384
MAX_BATCHED_TOKENS=8192
GPU_MEM_UTIL=0.8
SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
PORT=8006
TP=4
VLLM_TORCH_PROFILER_DIR=/llm/accuracy_test vllm serve \
    --model "$SERVER_MODEL" \
    --served-model-name "$SERVER_MODEL_NAME" \
    --allowed-local-media-path /llm/models \
    --dtype=float16 \
    --port $PORT \
    --host 0.0.0.0 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-util=$GPU_MEM_UTIL \
    --max-num-batched-tokens=$MAX_BATCHED_TOKENS \
    --disable-log-requests \
    --max-model-len=$MAX_MODEL_LEN \
    --block-size 64 \
    --quantization fp8 \
    -tp=$TP
