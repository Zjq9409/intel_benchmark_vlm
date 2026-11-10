export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export NCCL_P2P_LEVEL=SYS
vllm serve \
    --model /llm/models/Qwen3-VL-4B-Instruct/ \
    --served-model-name Qwen3-VL-4B-Instruct \
    --dtype=float16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len=32768 \
    --no-enable-prefix-caching \
    --gpu-memory-util=0.9 
