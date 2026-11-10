# vLLM 离线性能测试工具

用于测试视觉语言模型（VLM）在批量图像推理场景下的性能，支持 Intel XPU 和 NVIDIA CUDA 两种硬件平台。

## 目录结构

```
offline/
├── README.md           # 本文档
├── xpu/                # Intel XPU 设备测试
│   ├── run.sh          # XPU 启动脚本
│   ├── vllm_demo.py    # XPU 测试主程序
│   └── common_args.py  # 命令行参数定义
└── cuda/               # NVIDIA CUDA 设备测试
    ├── run.sh          # CUDA 启动脚本
    ├── vllm_demo.py    # CUDA 测试主程序
    └── common_args.py  # 命令行参数定义
```

## 功能特点

### 共同特性
- ✅ 自动加载文件夹中所有图片
- ✅ 循环批量处理所有图片
- ✅ 详细的批次级和整体性能统计
- ✅ 支持自定义批次大小
- ✅ 显示吞吐量（images/s 和 batches/s）
- ✅ 计算平均每张图片处理时间

### XPU 专有特性
- 支持丰富的性能调优参数
- FP8/FP16 量化支持
- 多 GPU 张量并行
- 可调的内存利用率和 KV cache 配置

### CUDA 专有特性
- 单批次模式：循环处理所有图片
- 多批次模式：快速测试多个批次大小性能对比

## 快速开始

### XPU 版本

#### 1. 基础用法
```bash
cd xpu/
./run.sh
```

#### 2. 自定义参数
编辑 `run.sh` 或直接运行：
```bash
python3 vllm_demo.py \
  --model-path /llm/models/Qwen3-VL-4B-Instruct/ \
  --imgs-path /llm/work/intel_benchmark_vlm/performance_benchmark/dataset/images \
  --batch 32 \
  --dtype float16 \
  --max-model-len 14000 \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --block-size 64 \
  --quantization fp8
```

### CUDA 版本

#### 1. 单批次模式（默认）
循环处理所有图片：
```bash
cd cuda/
./run.sh
```

#### 2. 多批次性能对比模式
测试多个批次大小（1,2,4,8,16,32,64,128）：
```bash
python3 vllm_demo.py \
  --model-path /llm/models/Qwen3-VL-4B-Instruct/ \
  --imgs-path /llm/work/intel_benchmark_vlm/performance_benchmark/dataset/images \
  --batch 32 \
  --multi-batch
```

## 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | - | 模型路径 |
| `--imgs-path` | str | - | 图片文件夹路径 |
| `--batch` | int | 2 | 批次大小 |
| `--system-prompt` | flag | False | 是否使用系统提示词 |

### XPU 专有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dtype` | str | float16 | 数据类型 (float16/bfloat16/float32) |
| `--max-model-len` | int | 14000 | 最大序列长度 |
| `--tensor-parallel-size` | int | 4 | 张量并行 GPU 数量 |
| `--max-num-batched-tokens` | int | 8192 | 批处理最大 token 数 |
| `--gpu-memory-utilization` | float | 0.9 | GPU 内存利用率 (0.0-1.0) |
| `--block-size` | int | 64 | KV cache 块大小 |
| `--quantization` | str | fp8 | 量化方法 (fp8/fp16/none) |

### CUDA 专有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--multi-batch` | flag | False | 启用多批次测试模式 |

## 输出示例

### XPU 单批次输出
```
Image path is: /llm/work/intel_benchmark_vlm/performance_benchmark/dataset/images
Total images loaded: 150
Initialized LLM successfully

Processing 150 images with batch size 32
Total batches to process: 5

Batch 1/5: Processing images 0 to 31 (32 images)
32
Batch time: 5.20s, Batch throughput: 6.15 images/s

Batch 2/5: Processing images 32 to 63 (32 images)
32
Batch time: 5.18s, Batch throughput: 6.18 images/s

Batch 3/5: Processing images 64 to 95 (32 images)
32
Batch time: 5.16s, Batch throughput: 6.20 images/s

Batch 4/5: Processing images 96 to 127 (32 images)
32
Batch time: 5.19s, Batch throughput: 6.17 images/s

Batch 5/5: Processing images 128 to 149 (22 images)
22
Batch time: 3.55s, Batch throughput: 6.20 images/s

============================================================
Overall Results:
Total images processed: 150
Total time: 24.28s
Overall throughput: 0.2059 batches/s, 6.18 images/s
Average time per image: 0.1619s
============================================================
```

### CUDA 多批次输出
```
Total images loaded: 150

Testing batch size: 1, using first 1 images
batch 1 throughput: 2.5000 it/s, 2.50 images/s

Testing batch size: 2, using first 2 images
batch 2 throughput: 1.8000 it/s, 3.60 images/s

Testing batch size: 4, using first 4 images
batch 4 throughput: 1.2500 it/s, 5.00 images/s

Testing batch size: 8, using first 8 images
batch 8 throughput: 0.8000 it/s, 6.40 images/s
...
```

## 性能调优建议

### XPU 平台

1. **提高吞吐量**
   - 增大 `--batch` (如 64, 128)
   - 增大 `--max-num-batched-tokens`
   - 提高 `--gpu-memory-utilization` 到 0.95

2. **降低延迟**
   - 减小 `--batch` (如 1, 2, 4)
   - 减小 `--max-model-len`

3. **内存优化**
   - 使用 `--quantization fp8` (最少内存)
   - 降低 `--max-model-len`
   - 调整 `--gpu-memory-utilization`

4. **精度 vs 速度**
   - `fp8`: 最快，精度略降
   - `fp16`: 平衡
   - `float32`: 最高精度，最慢

5. **多卡配置**
   - 调整 `--tensor-parallel-size` 匹配 GPU 数量
   - 确保环境变量正确配置

### CUDA 平台

1. **快速性能测试**
   ```bash
   python3 vllm_demo.py --multi-batch --imgs-path <path>
   ```
   查看不同批次的最佳性能点

2. **生产环境**
   根据多批次测试结果选择最佳批次大小，使用单批次模式处理完整数据集

## 环境要求

### XPU
- Intel XPU 驱动
- vLLM with XPU support
- PyTorch with XPU support
- transformers
- PIL

### CUDA
- NVIDIA GPU (CUDA 11.x+)
- vLLM
- PyTorch with CUDA
- transformers
- PIL
- qwen_vl_utils

## 常见问题

### 1. 图片加载失败
确保 `--imgs-path` 路径正确，且文件夹包含有效图片文件。

### 2. OOM (内存不足)
- 减小 `--batch`
- 降低 `--gpu-memory-utilization`
- 减小 `--max-model-len`
- 使用更激进的量化 (`fp8`)

### 3. 速度慢
- 增大 `--batch`（如果内存允许）
- 使用 `fp8` 量化
- 检查是否使用了全部 GPU (`--tensor-parallel-size`)

### 4. XPU 环境变量
确保 `run.sh` 中的环境变量正确：
```bash
export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### 5. CUDA 设备选择
编辑 `run.sh` 中的 CUDA_VISIBLE_DEVICES：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用 GPU 0,1,2,3
```

## 贡献与反馈

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

请参考项目根目录的 LICENSE 文件。
