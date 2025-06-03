#!/bin/bash

MODEL="/root/jane/Qwen2-VL-7B-Instruct/"
#MODEL="/root/jane/models--RedHatAI--Qwen2-VL-72B-Instruct-FP8-dynamic/"

export VLLM_USE_V1=1

NUM_CARD=1
# 设置开关变量
enable_prefix_caching=True  # 改为 False 则禁用 prefix caching

PREFIX_CACHING_FLAG=""
if [ "$enable_prefix_caching" = "False" ]; then
    PREFIX_CACHING_FLAG="--no-enable-prefix-caching"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME=$(basename "$(realpath -m "$MODEL")")
LOGFILE="${MODEL_NAME}_${NUM_CARD}_${enable_prefix_caching}_server_${TIMESTAMP}.log"

python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --dtype bfloat16 \
    --chat-template ../../template_chatml.jinja \
    --max_num_seqs 256 \
    --max-model-len 16384 \
    --port 8000 \
    --tensor-parallel-size ${NUM_CARD} \
    --trust-remote-code \
    --host 127.0.0.1 \
    ${PREFIX_CACHING_FLAG} | tee ${LOGFILE}
