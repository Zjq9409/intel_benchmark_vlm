SERVER_MODEL="/llm/models/Qwen3-30B-A3B/"
SERVER_MODEL_NAME="Qwen3-30B-A3B"
bsize=28
PORT=8006
vllm bench serve \
            --backend openai-chat \
            --model "$SERVER_MODEL" \
            --served-model-name "$SERVER_MODEL_NAME" \
            --endpoint /v1/chat/completions \
            --dataset-name random \
            --num-prompts $bsize \
            --max-concurrency $bsize \
            --random-input-len 2048 \
            --random-output-len 128 \
            --request-rate inf \
            --ignore-eos \
            --port=$PORT \
            --profile \
            --seed 42
