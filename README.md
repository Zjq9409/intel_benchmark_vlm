# Performance测试

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

## 2. 启动VLM服务（B60）

在B60服务器上启动VLLM服务：

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve \
    --model /llm/models/Qwen3-VL-4B-Instruct/ \
    --served-model-name Qwen3-VL-4B-Instruct \
    --dtype=float16 \
    --enforce-eager \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-util=0.9 \
    --max-num-batched-tokens=8192 \
    --disable-log-requests \
    --max-model-len=8192 \
    --block-size 64 \
    -tp=1
```

## 3. 运行性能测试

使用 `online/benchmark_server.sh` 脚本测试VLM模型性能：

```bash
cd online
bash benchmark_server.sh
```

测试脚本会：
- 自动从 `dataset/images` 目录读取图片
- 支持batch_size > 1时使用不同的图片
- 显示每张图片的尺寸信息
- 输出详细的性能指标（TTFT, TPOT, ITL等）

## 4. 测试输入的文本长度配置

### 测试场景
1. 128/128, 128/512, 128/1024
2. 1024/512， 1024/1024
3. 2048/512， 2048/1024， 2048/2048

### 提示词文件
- 128输入的prompt参看文件 `prompt_128.txt`
- 1024输入的prompt参看文件 `prompt_1k.txt`
- 2048输入的prompt参看文件 `prompt_2k.txt`

## 5. 手动测试代码示例

如需手动运行测试（不使用benchmark_server.sh）：

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

## 6. 测试结果

测试脚本会输出以下性能指标：
- **Request throughput**: 请求吞吐量（req/s）
- **Output token throughput**: 输出token吞吐量（tok/s）
- **TTFT** (Time to First Token): 首token延迟
- **TPOT** (Time per Output Token): 每个输出token的平均时间
- **ITL** (Inter-token Latency): token间延迟
- **Total Token throughput**: 总token吞吐量（包含输入+输出）

## 7. 解析测试日志

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

### 参数说明
- `log_file`: 必需，日志文件路径
- `--performance`: 可选，是否输出性能数据（默认为True）
  - `True/1/yes`: 解析性能指标并输出CSV
  - `False/0/no`: 解析准确性测试结果（图片路径和输出文本）

### 使用示例

```bash
# 解析性能数据（默认）
python parse_log.py benchmark_result.log

```

**注意**：脚本会自动过滤掉 Mean TTFT > 6000ms 的异常数据
