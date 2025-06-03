export VLLM_USE_V1=1
python3 -m vllm.entrypoints.openai.api_server \
    --model /weights/Kimi-VL-A3B-Thinking \
    --chat-template ./template_chatml.jinja \
    --tokenizer /weights/Kimi-VL-A3B-Thinking \
    --tensor-parallel-size 4 \
    --distributed-executor-backend ray \
    --dtype bfloat16 \
    --max_num_seqs 64 \
    --max-model-len 16384 \
    --port 8008 \
    --gpu_memory_utilization 0.95 \
    --host 127.0.0.1 \
    --trust_remote_code \
    > server_log_kimi_vl.txt 2>&1 &

