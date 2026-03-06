export TP=1
export MODEL_PATH="/llm/models/Qwen3-VL-4B-Instruct"
export MODEL_NAME="Qwen3-VL-4B-Instruct"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 使用绝对路径，避免因工作目录不同导致报告找不到
OUTPUT_DIR="/llm/performance_benchmark/online/profile"
OUTPUT_NAME="${OUTPUT_DIR}/my_vllm_report"

nsys profile \
    -o "${OUTPUT_NAME}" \
    --force-overwrite=true \
    --stats=true  \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
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
        --block-size 64 \
        --quantization fp8 \
        --profiler-config.profiler cuda \
        -tp=$TP
