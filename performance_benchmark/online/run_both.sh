#!/bin/bash
set -e  # 出错立即退出（可选）

# 提前请求 sudo 权限，并在后台保持 token 活跃，避免长时间运行后第二条命令需要重新输入密码
sudo -v
( while true; do sudo -v; sleep 60; done ) &
_SUDO_KEEPALIVE_PID=$!
trap "kill $_SUDO_KEEPALIVE_PID 2>/dev/null" EXIT

# 30b + 512x512（图片分辨率 512x512）
bash vllm_random_benchmark_server.sh 30b 512 512

# 30b + 1280x720（图片分辨率 1280x720，720P）
bash vllm_random_benchmark_server.sh 30b 1280 720

# 30b + 224x224（图片分辨率 224x224）
bash vllm_random_benchmark_server.sh 30b 224 224


# 4b + 1280x720（图片分辨率 1280x720，720P）
bash vllm_random_benchmark_server.sh 4b 1280 720

# 4b + 512x512（图片分辨率 512x512）
bash vllm_random_benchmark_server.sh 4b 512 512

# 4b + 224x224（图片分辨率 224x224）
bash vllm_random_benchmark_server.sh 4b 224 224
