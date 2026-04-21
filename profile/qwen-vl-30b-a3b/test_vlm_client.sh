export SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
export SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
export bsize=1
export NUM_PROMPTS=1
export OUTPUT_LEN=1024
export INPUT_LEN=200
export PORT=8000

# Usage: bash test_vlm_client.sh [--profile|--bench]
#   --profile : torch profiler, bsize=1, output=20 tokens  (compare kernel breakdown)
#   --bench   : throughput test, bsize=4, 20 requests       (compare tokens/s & latency)
#   default   : single latency test, bsize=1, 1 request
ENABLE_PROFILE=0
ENABLE_BENCH=0
for arg in "$@"; do
    case "$arg" in
        --profile) ENABLE_PROFILE=1 ;;
        --bench)   ENABLE_BENCH=1 ;;
    esac
done

PROFILE_FLAG=""
if [ "$ENABLE_PROFILE" = "1" ]; then
    PROFILE_FLAG="--profile"
    OUTPUT_LEN=20
elif [ "$ENABLE_BENCH" = "1" ]; then
    bsize=4
    NUM_PROMPTS=20
fi

vllm bench serve \
            --backend openai-chat \
            --model "$SERVER_MODEL" \
            --served-model-name "$SERVER_MODEL_NAME" \
            --endpoint /v1/chat/completions \
            --dataset-name random-mm \
            --num-prompts $NUM_PROMPTS \
            --max-concurrency $bsize \
            --random-input-len  $INPUT_LEN \
            --random-output-len $OUTPUT_LEN \
            --random-mm-base-items-per-request 1 \
            --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
            --random-mm-bucket-config '{ (448, 448, 1): 1}' \
            --request-rate inf \
            --ignore-eos \
            --port=$PORT \
            --seed 42 \
            $PROFILE_FLAG
