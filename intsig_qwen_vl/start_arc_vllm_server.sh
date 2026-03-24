export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1  
vllm serve \
    --model /llm/models/Qwen3-VL-4B-Instruct/ \
    --served-model-name Qwen3-VL-4B-Instruct \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enforce-eager    \
    --max-model-len=32768 \
    --max-num-batched-tokens=8192 \
    --block-size 64 \
    --no-enable-prefix-caching \
    --gpu-memory-util=0.9 
