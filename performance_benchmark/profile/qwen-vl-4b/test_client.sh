
export SERVER_MODEL="/llm/models/Qwen3-VL-4B-Instruct"
export SERVER_MODEL_NAME="Qwen3-VL-4B-Instruct"
export bsize=4
export OUTPUT_LEN=24
export PORT=8000
# Usage: bash test_client.sh [--profile|--no-profile]
# Default: profiling disabled
ENABLE_PROFILE=0
for arg in "$@"; do
    case "$arg" in
        --profile) ENABLE_PROFILE=1 ;;
        --no-profile) ENABLE_PROFILE=0 ;;
    esac
done

PROFILE_FLAG=""
[ "$ENABLE_PROFILE" = "1" ] && PROFILE_FLAG="--profile"

vllm bench serve \
            --backend openai-chat \
            --model "$SERVER_MODEL" \
            --served-model-name "$SERVER_MODEL_NAME" \
            --endpoint /v1/chat/completions \
            --dataset-name random-mm \
            --num-prompts $bsize \
            --max-concurrency $bsize \
            --random-input-len 128 \
            --random-output-len $OUTPUT_LEN \
            --random-mm-base-items-per-request 1 \
            --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
            --random-mm-bucket-config '{ (720, 1280, 1): 1 }' \
            --request-rate inf \
            --ignore-eos \
            --port=$PORT \
            --seed 42  \
            $PROFILE_FLAG \

# python3 ../vlm_benchmark.py \
#         --prompt "$(cat "$PROMPT_FILE")" \
#         --model "$MODEL_PATH" \
#         --served-model-name "$MODEL_NAME" \
#         --batch_size "$bsize" \
#         --output_len "$OUTPUT_LEN" \
#         --port 8000 \
#         --host 127.0.0.1 \
#         --profile \
#         --ignore-eos
