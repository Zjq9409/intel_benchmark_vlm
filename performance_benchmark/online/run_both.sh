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
# 参数: vllm_random_benchmark_server.sh <model> <w> <h> <mm_items> <mtp>
#   model    : 4b | q35-4b | 30b
#   w/h      : 图片分辨率
#   mm_items : 每请求图片数（1=单图, 10=多图）
#   mtp      : on | off（仅 Qwen3.5 系列支持）
# ================================================================

for res in "1280 720" "1920 1080"; do
    w=${res% *}; h=${res#* }
    for imgs in 1 10; do
        echo "--- q35-4b ${w}x${h} imgs=${imgs} ---"
        bash vllm_random_benchmark_server.sh q35-4b $w $h $imgs on
        bash vllm_random_benchmark_server.sh q35-4b $w $h $imgs off

        echo "--- 4b ${w}x${h} imgs=${imgs} ---"
        bash vllm_random_benchmark_server.sh 4b $w $h $imgs off
    done
done

# ================================================================
echo "========================================"
RUN_END=$(date "+%Y-%m-%d %H:%M:%S")
RUN_END_TS=$(date +%s)
ELAPSED=$(( RUN_END_TS - RUN_START_TS ))
printf "Run started at:  %s\n" "$RUN_START"
printf "Run finished at: %s\n" "$RUN_END"
printf "Total elapsed:   %02dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "========================================"
