# Performance测试

## 参数与容器路径映射

| 参数 | 宿主机路径 | 容器内路径 | 说明 |
|------|-----------|-----------|------|
| `--script-dir` | 指定路径（默认：脚本所在目录） | `/llm` | benchmark 脚本、dataset 等均从此路径访问 |
| `--weights-dir` | 指定路径（默认：`../weights`） | `/llm/models` | 模型权重目录，容器内通过 `/llm/models/<model_name>` 引用 |
| `--container-name` | — | — | 自定义容器名（NV 默认 `vllm-nv-container`，XPU 默认 `lsv-container-<suffix>`） |
| `--image-version` | — | — | Intel Docker 镜像版本（默认 `0.17.0-xpu`） |

## 快速开始

### 1. 初始化环境

```bash
# 在宿主机上运行，自动检测 GPU 类型并启动对应容器
bash setup_env.sh --weights-dir /data/weights/ --script-dir /data/intel_benchmark_vlm/ --image-version 0.14.0-b8

# 可通过 --container-name 自定义容器名（默认：NV=vllm-nv-container，XPU=lsv-container-<suffix>）
bash setup_env.sh --weights-dir /data/weights/ --script-dir /data/intel_benchmark_vlm/ --container-name my-container
```

### 2. 运行在线性能测试

详见 [performance_benchmark/online/online.md](performance_benchmark/online/online.md)  
