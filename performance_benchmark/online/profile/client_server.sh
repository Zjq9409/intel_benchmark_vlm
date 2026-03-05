
export PROMPT_FILE="../../prompt_128.txt"
export MODEL_PATH="/llm/models/Qwen3-VL-4B-Instruct"
export MODEL_NAME="Qwen3-VL-4B-Instruct"
export bsize=1
export OUTPUT_LEN=128
python3 ../vlm_benchmark.py \
        --prompt "$(cat "$PROMPT_FILE")" \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --batch_size "$bsize" \
        --output_len "$OUTPUT_LEN" \
        --port 8000 \
        --host 127.0.0.1 \
        --profile \
        --ignore-eos