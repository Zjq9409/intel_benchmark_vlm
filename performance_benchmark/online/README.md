# vLLM 在线性能测试

三步搞定：**起容器 → 跑脚本 → 看结果**。

---

## 1. 起容器（首次执行一次即可）

容器名建议带用户名后缀，避免和别人冲突。

**NVIDIA GPU**
```bash
bash ../../setup_env.sh \
    --container-name vllm-${USER} \
    --weights-dir /DISK0 \
    --script-dir $(realpath ../..) \
    --nv-image vllm/vllm-openai:v0.19.1-cu130
```

**Intel XPU**
```bash
bash ../../setup_env.sh \
    --container-name vllm-${USER}-xpu \
    --weights-dir /DISK0 \
    --script-dir $(realpath ../..) \
    --intel-image intel/llm-scaler-vllm:0.17.0-xpu
```

> `setup_env.sh` 自动检测 GPU 类型：有 `nvidia-smi` → NV 路径，否则走 Intel XPU 路径。

参数说明：
- `--container-name`：容器名，例如 `vllm-jane`
- `--weights-dir`：模型权重所在的 host 目录，会挂到容器内 `/llm/models`
- `--script-dir`：脚本仓库根目录（`intel_benchmark_vlm/`），挂到容器内 `/llm`
- `--intel-image`：Intel XPU 镜像（可选，默认 `intel/llm-scaler-vllm:0.17.0-xpu`）
- `--nv-image`：NV 镜像（可选，默认 `vllm/vllm-openai:v0.19.1-cu130`）

查看容器是否起来：
```bash
sudo docker ps | grep vllm-${USER}
```

---

## 2. 跑脚本

```bash
# 默认参数（4B 模型 + FP8 量化）
VLLM_NV_CONTAINER=vllm-nv-container bash run_nearrt_sweep.sh 4b

# XPU
VLLM_XPU_CONTAINER=vllm-${USER}-xpu bash run_nearrt_sweep.sh 4b
```

### 参数

| 位置 | 默认 | 可选 | 说明 |
|---|---|---|---|
| `$1` 模型 | `4b` | `4b` / `q35-4b` / `32b` / `q36-35b` / `30b` / `q36-27b` | 见 `model_config.sh` |
| `$2` 量化 | `fp8` | `fp8` / `none` | FP8 仅 Ada Lovelace 及以上 |
| `$3` MTP  | `off` | `on` / `off`  | Speculative Decoding（仅 Qwen3.5 系列） |
| `$4` Batch | `1`   | 任意正整数 | 固定 batch 大小 |

测试矩阵（脚本自动遍历）：
- 分辨率：`1280x720`、`1920x1080`
- 输入长度：`512`、`1024`
- 输出长度：`128`、`512`、`1024`
- 每请求帧数：`4`、`6`、`8`、`10`
- E2E 阈值：`30s`（超时立即停止该配置）

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_NV_CONTAINER` | `vllm-${USER}` | NV 容器名 |
| `VLLM_XPU_CONTAINER` | `vllm-${USER}-xpu` | XPU 容器名 |
| `WEIGHTS_DIR` | `/DISK0` | 模型权重宿主机路径 |
| `MODEL_PATH` | 无 | 强制指定模型绝对路径，覆盖 `model_config.sh` |

---


## 添加新模型

编辑 `model_config.sh`，在 `case` 里加一条：

```bash
8b)
    KEY_MATCHED=1
    MODEL_DIR="Qwen3-VL-8B-Instruct"
    MODEL_PATH_OVERRIDE="/llm/models/Qwen3-VL-8B-Instruct"
    TP=2
    ;;
```

要求：
- `MODEL_PATH_OVERRIDE` 是**容器内**绝对路径
- `TP` 不超过可用 GPU 数
- 模型实际放在 host `$WEIGHTS_DIR` 下，容器内通过 `/llm/models/...` 访问
