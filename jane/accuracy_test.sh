#!/bin/bash

ACCURACY_DIR="../accuracy"
SCRIPT="python ../personal_benchmark/vlm_benchmark.py"
MODEL_PATH="/llm/models/Qwen2-VL-7B-Instruct/"
BATCH_SIZE=1
PORT=8000
HOST="127.0.0.1"

# 遍历所有目录
find "$ACCURACY_DIR" -type d | while read -r DIR; do
    # 优先找当前目录的 prompt.txt，如果没有就找 *_prompt.txt
    PROMPT_FILE=""
    if [ -f "$DIR/prompt.txt" ]; then
        PROMPT_FILE="$DIR/prompt.txt"
    else
        PROMPT_CANDIDATE=$(find "$DIR" -maxdepth 1 -type f -name "*_prompt.txt" | head -n 1)
        if [ -n "$PROMPT_CANDIDATE" ]; then
            PROMPT_FILE="$PROMPT_CANDIDATE"
        fi
    fi

    # 如果找不到 prompt，跳过这个目录
    if [ -z "$PROMPT_FILE" ]; then
        continue
    fi

    # 读取 prompt 内容并转义双引号
    #PROMPT_TEXT=$(cat "$PROMPT_FILE" | tr '\n' ' ' | sed 's/"/\\"/g')
    PROMPT_TEXT=$(paste -sd' ' "$PROMPT_FILE" | sed 's/"/\\"/g')

    # 查找当前目录下的 .jpg 和 .png 图片
    find "$DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" \) | while read -r IMAGE_PATH; do
        echo "Processing:"
        echo "  Image : $IMAGE_PATH"
        echo "  Prompt: $PROMPT_FILE"
        echo ""

        $SCRIPT \
            --image_path "$IMAGE_PATH" \
            --prompt "$PROMPT_TEXT" \
            --model "$MODEL_PATH" \
            --batch_size "$BATCH_SIZE" \
            --port "$PORT" \
            --host "$HOST"
    done
done

#python /root/jane/intel_benchmark_vlm/personal_benchmark/vlm_benchmark.py \
#--image_path /root/jane/intel_benchmark_vlm/accuracy/zto/shangmen/40_20240903093021_78830772282673.jpg \
#--prompt "Please utilize the visual large model to analyze the image and determine the presence of the following elements: 1. Entrance door (Y/N), 2. Package (Y/N). Output the results concatenated by a comma, for example: Y,N."      \
#--model  /root/jane/Qwen2-VL-7B-Instruct/ \
#--batch_size 1 \
#--port 8000 \
#--host 127.0.0.1
