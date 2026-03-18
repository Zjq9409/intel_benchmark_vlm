#!/bin/bash
set -e  # 出错立即退出（可选）

# 默认 30b + 1280x720
bash vllm_random_benchmark_server.sh

# 30b + 512x512
bash vllm_random_benchmark_server.sh 30b 512 512

# 4b + 1280x720（默认尺寸）
bash vllm_random_benchmark_server.sh 4b

# 4b + 512x512
bash vllm_random_benchmark_server.sh 4b 512 512
