vllm serve \
    --model /DISK0/MiniCPM-V-4.6/ \
    --served-model-name MiniCPM-V-4.6 \
    --dtype=float16 \
    --port 8009 \
    --mm-processor-cache-gb 0 \
    --no-enable-prefix-caching \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-util=0.85 \
    --max-model-len=16384 \
    -tp=1 \
    2>&1 | tee ./vllm.log
