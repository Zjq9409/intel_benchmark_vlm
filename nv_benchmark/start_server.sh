#export VLLM_USE_V1=1
#python -m vllm.entrypoints.openai.api_server \
#    --model /root/jane/Qwen2-VL-7B-Instruct/ \
#    --tokenizer /root/jane/Qwen2-VL-7B-Instruct/ \
#    --dtype bfloat16 \
#    --chat-template /root/jane/vllm/examples/template_chatml.jinja \
#    --max_num_seqs 256 \
#    --max-model-len 16384 \
#    --no-enable-prefix-caching \
#    --port 8000 \
#    --host 127.0.0.1
#
MODEL="/root/jane/models--RedHatAI--Qwen2-VL-72B-Instruct-FP8-dynamic/"
export VLLM_USE_V1=1
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --tokenizer ${MODEL} \
    --dtype bfloat16 \
    --chat-template /root/jane/vllm/examples/template_chatml.jinja \
    --max_num_seqs 256 \
    --max-model-len 16384 \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 127.0.0.1

