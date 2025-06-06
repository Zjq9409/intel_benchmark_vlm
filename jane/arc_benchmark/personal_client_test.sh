#!/bin/bash

#MODEL="/llm/models/Qwen2-VL-7B-Instruct/"
MODEL="/llm/models/Qwen2-VL-72B-Instruct/"
PROMPT_DIR="../../performance_benchmark"
IMAGE_PATH="../../accuracy/zto/40_20240903093021_78830772282673.jpg"

# 定义 batch_size 列表
BATCH_SIZES=(1)

# 循环 prompt 文件
for PROMPT_FILE in prompt_128.txt prompt_1k.txt prompt_2k.txt; do
    FULL_PROMPT_PATH="${PROMPT_DIR}/${PROMPT_FILE}"
    PROMPT_CONTENT=$(cat "${FULL_PROMPT_PATH}")

    # 根据 prompt 文件名设置 output_len 范围
    case "$PROMPT_FILE" in
        prompt_128.txt)
            OUT_LENS=(128 512 1024)
            ;;
        prompt_1k.txt)
            OUT_LENS=(512 1024)
            ;;
        prompt_2k.txt)
            OUT_LENS=(512 1024 2048)
            ;;
        *)
            echo "Unknown prompt file: $PROMPT_FILE"
            continue
            ;;
    esac

    # 执行测试
    for BS in "${BATCH_SIZES[@]}"; do
        for OUT_LEN in "${OUT_LENS[@]}"; do
            python ../../personal_benchmark/vlm_benchmark.py \
                --image_path "${IMAGE_PATH}" \
                --prompt "${PROMPT_CONTENT}" \
                --model "${MODEL}" \
                --batch_size "${BS}" \
                --port 8000 \
                --host 127.0.0.1 \
                --ignore-eos \
                --output_len "${OUT_LEN}"
	    sleep 10
        done
    done
done

