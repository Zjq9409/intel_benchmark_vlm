# Performance测试


## 脚本说明

| 脚本 | 说明 |
|------|------|
| `setup_env.sh` | 环境初始化脚本，自动检测 GPU 类型：检测到 **NVIDIA GPU** 时，使用 `uv` 创建 Python 3.12 虚拟环境并安装 vllm；检测到 **Intel GPU** 时，拉取 `intel/llm-scaler-vllm` Docker 镜像并启动容器（需传入镜像版本号，如 `0.11.1-b7`） |
| `performance_benchmark/online/intel_benchmark_server.sh` | **Intel GPU** 性能测试脚本。设置 Intel vllm 所需环境变量（`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT`、`VLLM_WORKER_MULTIPROC_METHOD` 等），启动 vllm OpenAI 兼容服务端，循环运行不同 batch_size 的 `vllm bench serve` 纯文本 benchmark，完成后自动调用 `parse_log.py` 生成 CSV 结果 |
| `performance_benchmark/online/nv_benchmark_server.sh` | **NVIDIA GPU** 性能测试脚本。根据 `tp` 参数自动设置 `CUDA_VISIBLE_DEVICES`，启动 vllm 服务端（支持 fp8 量化），循环运行不同 batch_size 的 `vlm_benchmark.py` 图文 benchmark（从 `dataset/images` 读取图片），完成后自动调用 `parse_log.py` 生成 CSV 结果 |

### 使用方式

```bash
# 初始化环境（NVIDIA，无需参数）
bash setup_env.sh

# 初始化环境（Intel，需指定镜像版本）
bash setup_env.sh 0.11.1-b7

# 运行 Intel GPU benchmark
bash performance_benchmark/online/intel_benchmark_server.sh <model_path> <model_name> [tp] [image_dir]

# 运行 NVIDIA GPU benchmark
bash performance_benchmark/online/nv_benchmark_server.sh <model_path> <model_name> [tp] [image_dir]
```

---

## 快速开始

性能测试在 `performance_benchmark` 文件夹下进行：

```bash
cd performance_benchmark
```

## 1. 下载测试数据集

首先需要下载COCO图像数据集用于测试：

```bash

cd performance_benchmark

#安装依赖
pip install datasets tqdm pillow


# 下载400张图像（默认）
python3 download_dataset.py

# 自定义下载数量
python3 download_dataset.py --num-images 320

# 下载完成后会询问是否调整图片到1080P (1920x1080)
# 选择 y 会将所有图片统一调整为1920x1080，使用黑边填充保持内容完整
```

详细说明参考 [DATASET_README.md](DATASET_README.md)

## 2. 测试输入的文本长度配置

### 测试场景
1. 128/128, 128/512, 128/1024
2. 1024/512， 1024/1024
3. 2048/512， 2048/1024， 2048/2048

### 提示词文件
- 128输入的prompt参看文件 `prompt_128.txt`
- 1024输入的prompt参看文件 `prompt_1k.txt`
- 2048输入的prompt参看文件 `prompt_2k.txt`

## 3. 手动测试代码示例

如需手动运行测试（不使用 benchmark 脚本）：

```bash
cd online

python vlm_benchmark.py \
--image_path ../dataset/images/coco_000000000009.jpg  \
--prompt "$(cat ../prompt_128.txt)" \
--model /llm_test/weights/Qwen3-VL-4B-Instruct/ \
--served-model-name Qwen3-VL-4B-Instruct \
--batch_size 1 \
--port 8000 \
--host 127.0.0.1 \
--ignore-eos \
--output_len 128
```

### 参数说明
- `--model`: 模型路径，用于加载tokenizer计算输入token数量，不需要与服务端模型一致，确保能访问到
- `--batch_size`: 并发请求数量，>1时会自动从dataset/images读取不同图片
- `--output_len`: 最大输出token长度
- `--prompt`: 输入提示词
- `--image_path`: 单张图片路径（batch_size=1时使用，>1时自动忽略）
- `--ignore-eos`: 忽略EOS token，强制输出到max_tokens

## 4. 测试结果

测试脚本会输出以下性能指标：
- **Request throughput**: 请求吞吐量（req/s）
- **Output token throughput**: 输出token吞吐量（tok/s）
- **TTFT** (Time to First Token): 首token延迟
- **TPOT** (Time per Output Token): 每个输出token的平均时间
- **ITL** (Inter-token Latency): token间延迟
- **Total Token throughput**: 总token吞吐量（包含输入+输出）

## 5. 解析测试日志

使用 `parse_log.py` 脚本可以从测试日志中提取性能指标并导出为CSV格式：

```bash
# 解析性能测试日志
python parse_log.py test_output.log

# 会生成 test_output.log.csv 文件，包含以下指标：
# - Total input tokens
# - Total image and input tokens
# - Total generated tokens
# - Successful requests
# - Mean TTFT (ms)
# - Mean TPOT (ms)
# - Mean ITL (ms)
# - Request throughput (req/s)
# - Output token throughput (tok/s)
# - Benchmark duration (s)
```

### 使用示例

```bash
# 解析性能数据（默认）
python parse_log.py benchmark_result.log

```