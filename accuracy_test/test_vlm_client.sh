SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
bsize=28
PORT=8006
vllm bench serve \
            --backend openai-chat \
            --model "$SERVER_MODEL" \
            --served-model-name "$SERVER_MODEL_NAME" \
            --endpoint /v1/chat/completions \
            --dataset-name random-mm \
            --num-prompts $bsize \
            --max-concurrency $bsize \
            --random-input-len 128 \
            --random-output-len 128 \
            --random-mm-base-items-per-request 1 \
            --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
            --random-mm-bucket-config '{(1920, 1080, 1): 1.0}' \
            --request-rate inf \
            --ignore-eos \
            --port=$PORT \
            --profile \
            --seed 42
