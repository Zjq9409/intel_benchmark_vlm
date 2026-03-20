# Performance测试


## 脚本说明

| 脚本 | 说明 |
|------|------|
| `setup_env.sh` | 环境初始化脚本，自动检测 GPU 类型：检测到 **NVIDIA GPU** 时，拉取 `vllm/vllm-openai` Docker 镜像并启动容器；检测到 **Intel GPU** 时，拉取 `intel/llm-scaler-vllm` Docker 镜像并启动容器。支持命名参数：`--weights-dir`（模型目录，默认 `../weights`）、`--script-dir`（脚本根目录，默认脚本所在目录）、`--image-version`（Intel 镜像版本，默认 `0.11.1-b7`） |
| `performance_benchmark/online/vllm_random_benchmark_server.sh` | **NVIDIA / Intel GPU 通用**性能测试脚本。自动检测 GPU 类型，启动 vllm OpenAI 兼容服务端（支持 fp8 量化、tp 并行），使用 `vllm bench serve --dataset-name random-mm` 进行图文 benchmark（随机生成指定尺寸图片，无需预先下载数据集），循环运行 num-prompts 从 1 到 200（步长 2），当 Mean TTFT 超过 6000ms 时自动停止，完成后自动调用 `parse_log.py` 生成 CSV 结果 |
| `performance_benchmark/online/run_both.sh` | 批量运行脚本，依次以不同模型/图片尺寸组合执行 `vllm_random_benchmark_server.sh` |

### 参数与容器路径映射

| 参数 | 宿主机路径 | 容器内路径 | 说明 |
|------|-----------|-----------|------|
| `--script-dir` | 指定路径（默认：脚本所在目录） | `/llm` | benchmark 脚本、dataset 等均从此路径访问 |
| `--weights-dir` | 指定路径（默认：`../weights`） | `/llm/models` | 模型权重目录，容器内通过 `/llm/models/<model_name>` 引用 |

### 使用方式

脚本自动检测 GPU 类型，无需区分 NVIDIA / Intel，统一调用方式如下：

```bash
# 默认启动（自动检测 GPU，使用 ../weights 作为模型目录）
bash setup_env.sh

# 指定模型目录
bash setup_env.sh --weights-dir /data/models

# 指定模型目录和脚本根目录
bash setup_env.sh --weights-dir /data/models --script-dir /custom/path

# 仅 Intel GPU：指定镜像版本（NVIDIA 忽略此参数）
bash setup_env.sh --image-version 0.12.0 --weights-dir /data/models
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

# 测试 30B 模型（默认），图片 1280×720（默认）
bash vllm_random_benchmark_server.sh

# 测试 4B 模型，图片 1280×720（默认）
bash vllm_random_benchmark_server.sh 4b

# 测试 30B 模型，图片 512×512
bash vllm_random_benchmark_server.sh 30b 512 512

# 测试 4B 模型，图片 512×512
bash vllm_random_benchmark_server.sh 4b 512 512

# 批量运行多个组合（见 run_both.sh）
bash run_both.sh
```

#### 脚本参数

| 位置参数 | 可选值 | 默认值 | 说明 |
|---------|--------|--------|------|
| `$1` 模型规格 | `4b` / `30b` | `30b` | 选择测试模型；`4b` → `Qwen3-VL-4B-Instruct`（TP=1），`30b` → `Qwen3-VL-30B-A3B-Instruct`（TP=4） |
| `$2` 图片宽度 | 任意整数 | `1280` | 随机生成图片的宽度（像素） |
| `$3` 图片高度 | 任意整数 | `720` | 随机生成图片的高度（像素） |

#### 服务端固定配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8006` | 服务端口 |
| `MAX_BATCHED_TOKENS` | `8192` | 最大批处理 token 数 |
| `MAX_MODEL_LEN` | `16384` | 最大模型上下文长度 |
| `GPU_MEM_UTIL` | `0.8` | GPU 显存利用率 |
| `INPUT_LEN` | `1024` | 随机输入 token 长度 |
| `OUTPUT_LEN` | `1024` | 随机输出 token 长度 |
| `MAX_BSIZE` | `200` | num-prompts 最大值 |
| TTFT 阈值 | `6000 ms` | 超过此值时自动停止测试并保存结果 |

测试循环范围：num-prompts = 1, 2, 4, 6, …（步长 2），最大 200，或 Mean TTFT > 6000ms 时提前终止。

#### 日志文件命名规则

日志保存在以模型名命名的子目录下，文件名格式为：

```
<MODEL_NAME>/<YYYYMMDD_HHMMSS>_client_tp<TP>_mbt<MBT>_<W>x<H>_in<IN>_out<OUT>_<GPU_TYPE>.log
<MODEL_NAME>/<YYYYMMDD_HHMMSS>_server_tp<TP>_mbt<MBT>_<W>x<H>_in<IN>_out<OUT>_<GPU_TYPE>.log
```

例如：`Qwen3-VL-30B-A3B-Instruct/20260320_153000_client_tp4_mbt8192_1280x720_in1024_out1024_H100.log`

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
python3 parse_log.py <MODEL_NAME>/xxx_client_xxx.log

# 会生成同名 .csv 文件，包含以下指标：
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

---

## 测试其他模型

如需测试非内置的模型，修改 `vllm_random_benchmark_server.sh` 脚本顶部的模型选择逻辑：

```bash
# 脚本第 ~48 行，在 if/else 分支中添加新模型或修改已有配置
if [ "$MODEL_SELECT" = "4b" ]; then
    SERVER_MODEL="/llm/models/Qwen3-VL-4B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-4B-Instruct"
    TP=1
elif [ "$MODEL_SELECT" = "8b" ]; then          # ← 新增示例
    SERVER_MODEL="/llm/models/Qwen3-VL-8B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-8B-Instruct"
    TP=2
else
    SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
    TP=4
fi
```

同时确认：
1. **模型权重**已放置在宿主机 `weights/` 目录下（容器内映射为 `/llm/models/`）  
2. **TP**（tensor parallel）值与可用 GPU 卡数匹配  
3. 若需要不同的量化策略，修改 `VLLM_SERVER_ARGS` 中的 `--quantization` 参数  
