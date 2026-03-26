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
# 指定模型目录和脚本根目录
bash setup_env.sh --weights-dir /data/weights/ --script-dir /data/intel_benchmark_vlm/
```

---

## 快速开始

### 1. 初始化环境

```bash
# 在宿主机上运行，自动检测 GPU 类型并启动对应容器
bash setup_env.sh
```

### 2. 运行在线性能测试

详见 [performance_benchmark/online/online.md](performance_benchmark/online/online.md)  
