#!/bin/bash
set -e

# 提前请求 sudo 权限，保持 token 活跃
sudo -v
( while true; do sudo -v; sleep 60; done ) &
_SUDO_KEEPALIVE_PID=$!
trap "kill $_SUDO_KEEPALIVE_PID 2>/dev/null" EXIT

RUN_START=$(date "+%Y-%m-%d %H:%M:%S")
RUN_START_TS=$(date +%s)
echo "========================================"
echo "Run started at: $RUN_START"
echo "========================================"

# ================================================================
# 准实时场景测试配置
# 场景定义：E2E < 60s 下的最大 TPS
# 指标：Max TPS @ E2E Latency < 60s
#
# 测试矩阵:
#   模型: 30B-A3B (MoE), 32B (Dense)
#   分辨率: 720p, 1080p
#   精度: fp8, fp16（32B 只测 fp8）
#   每请求图片数: 1
#   Output length: 512 tokens（准实时场景典型值）
#
# 用法: bash run_near_realtime.sh [device_id]
# ================================================================
QUANT="fp8"
DEVICE="${1:-}"
export VLLM_NV_CONTAINER="${VLLM_NV_CONTAINER:-vllm-nv-container}"
export VLLM_XPU_CONTAINER="${VLLM_XPU_CONTAINER:-lsv-container-0428}"

echo "Quantization:  $QUANT"
echo "GPU Device:    ${DEVICE:-all}"
echo "Output Length: $OUTPUT_LEN tokens"
echo "NV Container:  $VLLM_NV_CONTAINER"
echo "XPU Container: $VLLM_XPU_CONTAINER"

# 30B-A3B MoE 模型
# imgs=1: 单帧基线；imgs=4: 30s片段4帧；imgs=8: 30s片段8帧（时序理解）
# for input_len in 512 1024; do
#     for output_len in 128 1024; do
#         for res in "1280 720" "1920 1080"; do
#             w=${res% *}; h=${res#* }
#             for imgs in 1 4 8 10; do
#                 echo "--- 30b ${w}x${h} imgs=${imgs} quant=${QUANT} in=${input_len} out=${output_len} ---"
#                 bash vllm_random_benchmark_server.sh 30b $w $h $imgs off $QUANT "$DEVICE" $output_len $input_len
#             done
#         done
#     done
# done

# 4B 模型（单卡 TP=1，FP8）
for input_len in 512 1024; do
    for output_len in 128 1024; do
        for res in "1280 720" "1920 1080"; do
            w=${res% *}; h=${res#* }
            for imgs in 1 4 8 10; do
                echo "--- 4b ${w}x${h} imgs=${imgs} quant=${QUANT} in=${input_len} out=${output_len} ---"
                bash vllm_random_benchmark_server.sh 4b $w $h $imgs off $QUANT "$DEVICE" $output_len $input_len
            done
        done
    done
done

# 32B Dense 模型（仅 fp8）
# for input_len in 512 1024; do
#     for output_len in 128 1024; do
#         for res in "1280 720" "1920 1080"; do
#             w=${res% *}; h=${res#* }
#             for imgs in 1 4 8 10; do
#                 echo "--- 32b ${w}x${h} imgs=${imgs} quant=fp8 in=${input_len} out=${output_len} ---"
#                 bash vllm_random_benchmark_server.sh 32b $w $h $imgs off fp8 "$DEVICE" $output_len $input_len
#             done
#         done
#     done
# done

# ================================================================
echo "========================================"
RUN_END=$(date "+%Y-%m-%d %H:%M:%S")
RUN_END_TS=$(date +%s)
ELAPSED=$(( RUN_END_TS - RUN_START_TS ))
printf "Run started at:  %s\n" "$RUN_START"
printf "Run finished at: %s\n" "$RUN_END"
printf "Total elapsed:   %02dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "========================================"
