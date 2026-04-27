#!/bin/bash
set -e

# 提前请求 sudo 权限，并在后台保持 token 活跃，避免长时间运行后第二条命令需要重新输入密码
sudo -v
( while true; do sudo -v; sleep 60; done ) &
_SUDO_KEEPALIVE_PID=$!
trap "kill $_SUDO_KEEPALIVE_PID 2>/dev/null" EXIT

# ================================================================
# 参数说明: vllm_random_benchmark_server.sh <model> <w> <h> <mm_items> <mtp>
#   model    : 4b | q35-4b | 30b
#   w/h      : 图片分辨率
#   mm_items : 每请求图片数（1=单图, 10=多图/图片理解场景）
#   mtp      : on | off（仅 Qwen3.5 系列支持 MTP）
# ================================================================

# ----------------------------------------------------------------
# Qwen3.5-4B — 720P — 单图（1张/请求）
# ----------------------------------------------------------------
# MTP 开启
bash vllm_random_benchmark_server.sh q35-4b 1280 720 1 on

# MTP 关闭（对比）
bash vllm_random_benchmark_server.sh q35-4b 1280 720 1 off

# ----------------------------------------------------------------
# Qwen3.5-4B — 720P — 多图（10张/请求，模拟 NarratoAI）
# ----------------------------------------------------------------
# MTP 开启
bash vllm_random_benchmark_server.sh q35-4b 1280 720 10 on

# MTP 关闭（对比）
bash vllm_random_benchmark_server.sh q35-4b 1280 720 10 off

# ----------------------------------------------------------------
# 其他模型参考（按需取消注释）
# ----------------------------------------------------------------
# bash vllm_random_benchmark_server.sh 30b 1280 720 1 off
# bash vllm_random_benchmark_server.sh 4b  1280 720 1 off
