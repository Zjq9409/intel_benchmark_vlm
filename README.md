# Performance测试

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
