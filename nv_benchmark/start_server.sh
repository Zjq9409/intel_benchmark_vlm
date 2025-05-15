export VLLM_USE_V1=1
python -m vllm.entrypoints.openai.api_server \
    --model /root/jane/Qwen2-VL-7B-Instruct \
    --tokenizer /root/jane/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --chat-template /root/jane/vllm/examples/template_chatml.jinja \
    --max_num_seqs 256 \
    --max-model-len 16384 \
    --port 8001 \
    --host 127.0.0.1