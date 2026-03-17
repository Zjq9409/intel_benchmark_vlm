# Performance测试


## 脚本说明

| 脚本 | 说明 |
|------|------|
| `setup_env.sh` | 环境初始化脚本，自动检测 GPU 类型：检测到 **NVIDIA GPU** 时，拉取 `vllm/vllm-openai` Docker 镜像并启动容器；检测到 **Intel GPU** 时，拉取 `intel/llm-scaler-vllm` Docker 镜像并启动容器。支持命名参数：`--weights-dir`（模型目录，默认 `../weights`）、`--script-dir`（脚本根目录，默认脚本所在目录）、`--image-version`（Intel 镜像版本，默认 `0.11.1-b7`） |
| `performance_benchmark/online/vllm_random_benchmark_server.sh` | **NVIDIA / Intel GPU 通用**性能测试脚本。自动检测 GPU 类型，启动 vllm OpenAI 兼容服务端（支持 fp8 量化、tp 并行），使用 `vllm bench serve --dataset-name random-mm` 进行图文 benchmark（随机生成 224×224 图片，无需预先下载数据集），循环运行 num-prompts 从 1 到 128（NVIDIA）或 100（Intel XPU）的测试，完成后自动调用 `parse_log.py` 生成 CSV 结果 |

### 参数与容器路径映射

| 参数 | 宿主机路径 | 容器内路径 | 说明 |
|------|-----------|-----------|------|
| `--script-dir` | 指定路径（默认：脚本所在目录） | `/llm` | benchmark 脚本、dataset 等均从此路径访问 |
| `--weights-dir` | 指定路径（默认：`../weights`） | `/llm/models` | 模型权重目录，容器内通过 `/llm/models/<model_name>` 引用 |

### 使用方式

```bash
# 初始化环境（NVIDIA，自动检测 ../weights 作为模型目录）
bash setup_env.sh

# 初始化环境（NVIDIA，指定 weights 目录）
bash setup_env.sh --weights-dir /data/models

# 初始化环境（NVIDIA，同时指定 weights 目录和脚本根目录）
bash setup_env.sh --weights-dir /data/models --script-dir /custom/path

# 初始化环境（Intel，使用默认镜像版本 0.11.1-b7）
bash setup_env.sh

# 初始化环境（Intel，指定镜像版本）
bash setup_env.sh --image-version 0.12.0

# 初始化环境（Intel，指定镜像版本和 weights 目录）
bash setup_env.sh --image-version 0.12.0 --weights-dir /data/models

# 初始化环境（Intel，同时指定镜像版本、weights 目录和脚本根目录）
bash setup_env.sh --image-version 0.12.0 --weights-dir /data/models --script-dir /custom/path
```

---

## 快速开始

### 1. 初始化环境

```bash
# 在宿主机上运行，自动检测 GPU 类型并启动对应容器
bash setup_env.sh
```

### 2. 运行性能测试

无需下载数据集，直接运行 benchmark 脚本即可。脚本会自动启动 vllm 服务端并依次测试各并发量：

```bash
cd performance_benchmark/online

# 直接在宿主机运行（脚本会自动进入容器执行）
bash vllm_random_benchmark_server.sh
```

脚本内置配置（可直接编辑脚本顶部变量修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SERVER_MODEL` | `/llm/models/Qwen3-VL-30B-A3B-Instruct` | 模型路径（容器内） |
| `PORT` | `8006` | 服务端口 |
| `TP` | `4` | tensor parallel 并行度 |
| `MAX_BATCHED_TOKENS` | `8192` | 最大批处理 token 数 |
| `MAX_MODEL_LEN` | `16384` | 最大模型上下文长度 |
| `GPU_MEM_UTIL` | `0.8` | GPU 显存利用率 |
| `--random-input-len` | `128` | 随机输入 token 长度 |
| `--random-output-len` | `128` | 随机输出 token 长度 |
| `--random-mm-bucket-config` | `(224,224,1): 1.0` | 随机图片尺寸为 224×224 |

测试循环范围：num-prompts = 1, 2, 4, 6, …（步长 2），NVIDIA 最大 128，Intel XPU 最大 100。

### 3. 测试结果

测试脚本会输出以下性能指标：
- **Request throughput**: 请求吞吐量（req/s）
- **Output token throughput**: 输出token吞吐量（tok/s）
- **TTFT** (Time to First Token): 首token延迟
- **TPOT** (Time per Output Token): 每个输出token的平均时间
- **ITL** (Inter-token Latency): token间延迟
- **Total Token throughput**: 总token吞吐量（包含输入+输出）

### 4. 解析测试日志

测试完成后脚本会自动调用 `parse_log.py` 生成 CSV，也可手动解析：

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
