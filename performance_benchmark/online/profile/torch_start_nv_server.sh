export TP=1
export MODEL_PATH="/llm/models/Qwen3-VL-4B-Instruct"
export MODEL_NAME="Qwen3-VL-4B-Instruct"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
VLLM_TORCH_PROFILER_DIR=/llm/performance_benchmark/online/profile python3 -m vllm.entrypoints.openai.api_server \
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
    --block-size 64 \
    --quantization fp8 \
    -tp=$TP   \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
