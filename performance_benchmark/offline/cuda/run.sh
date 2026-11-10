#!/bin/bash

# CUDA环境变量配置
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行vLLM推理测试
# 单批次测试（默认）：只测试 --batch 指定的批次
# 多批次测试：添加 --multi-batch 参数测试多个批次 (1,2,4,8,16,32,64,128)

python3 vllm_demo.py \
  --model-path /llm/models/Qwen3-VL-4B-Instruct/ \
  --imgs-path /llm/work/intel_benchmark_vlm/performance_benchmark/dataset/images \
  --batch 32
  # --multi-batch  # 取消注释以测试多个批次
