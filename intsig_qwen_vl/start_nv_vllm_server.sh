vllm serve \
    --model /llm/models/Qwen3-VL-8B-Instruct/ \
    --served-model-name Qwen3-VL-8B-Instruct \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len=32768 \
    --no-enable-prefix-caching \
    --gpu-memory-util=0.9 
