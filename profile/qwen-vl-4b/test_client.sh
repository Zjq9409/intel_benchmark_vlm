export SERVER_MODEL="/llm/models/Qwen3-VL-4B-Instruct"
export SERVER_MODEL_NAME="Qwen3-VL-4B-Instruct"
export bsize=4
export OUTPUT_LEN=1024
export INPUT_LEN=512
export PORT=8000
# Usage: bash test_client.sh [--profile|--no-profile] [--bsize N]
# Default: profiling disabled, bsize=4
ENABLE_PROFILE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) ENABLE_PROFILE=1; shift ;;
        --no-profile) ENABLE_PROFILE=0; shift ;;
        --bsize) bsize="$2"; shift 2 ;;
        *) shift ;;
    esac
done

PROFILE_FLAG=""
if [ "$ENABLE_PROFILE" = "1" ]; then
    PROFILE_FLAG="--profile"
    OUTPUT_LEN=20
fi

vllm bench serve \
            --backend openai-chat \
            --model "$SERVER_MODEL" \
            --served-model-name "$SERVER_MODEL_NAME" \
            --endpoint /v1/chat/completions \
            --dataset-name random-mm \
            --num-prompts $bsize \
            --max-concurrency $bsize \
            --random-input-len  $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --random-mm-base-items-per-request 1 \
            --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
             --random-mm-bucket-config '{ (720, 1280, 1): 1}' \
            --request-rate inf \
            --ignore-eos \
            --port=$PORT \
            --seed 42  \
            $PROFILE_FLAG
