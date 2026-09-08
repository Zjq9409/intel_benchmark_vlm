# vLLM 在线性能测试

三步搞定：**起容器 → 跑脚本 → 看结果**。

---

## 1. 起容器（首次执行一次即可）

容器名建议带用户名后缀，避免和别人冲突。

**NVIDIA GPU**
```bash
bash ../../setup_env.sh \
    --container-name vllm-{USER}-nv \
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

---

## 2. 跑脚本

```bash
# NVIDIA GPU：默认参数（4B 模型 + FP8 量化）
VLLM_NV_CONTAINER=vllm-{USER}-nv bash run_nearrt_sweep.sh 4b

# 指定 GPU，例如只用 GPU 4
VLLM_NV_CONTAINER=vllm-{USER}-nv bash run_nearrt_sweep.sh 4b 1 4 fp8

# XPU
VLLM_XPU_CONTAINER=vllm-${USER}-xpu bash run_nearrt_sweep.sh 4b
```

### 参数

`run_nearrt_sweep.sh` 的位置参数如下：

| 位置 | 默认 | 可选 | 说明 |
|---|---|---|---|
| `$1` 模型 | `4b` | `4b` / `q35-4b` / `32b` / `q36-35b` / `30b` / `q36-27b` | 见 `model_config.sh` |
| `$2` TP | `4` | 正整数，例如 `1`、`2`、`4` | Tensor Parallelism；需要不超过可见 GPU 数量 |
| `$3` GPU 设备 | 空 | 例如 `4`、`0,1,2,3` | 传给 `CUDA_VISIBLE_DEVICES`；空表示使用全部可见设备 |
| `$4` 量化 | `fp8` | `fp8` / `none` | `fp8` 开启 FP8 量化，`none` 表示不启用量化 |

测试矩阵（脚本自动遍历）：
- 分辨率：`1280x720`、`1920x1080`
- 输入长度：`512`、`1024`
- 输出长度：`128`、`1024`
- 每请求帧数：`1`
- 主停止阈值：`E2E_LIMIT=30s`，某个 batch 的 E2E 超过该阈值后停止当前配置
- 绘图阈值：默认同时为 `E2E<=5s / 10s / 20s / 30s` 细扫边界，方便多硬件 TPS 对比

### Batch sweep 策略

默认先按一组对齐的 batch 候选点做粗扫：

```text
1 2 4 8 16 24 32 48 64 96 128 160 192 224 256 320 384 448 512
```

脚本不会因为跑到 `64` 就停；默认会继续向上测试，直到：

```text
E2E > E2E_LIMIT（默认 30s）
```

然后会对 `E2E_THRESHOLDS` 中的每个阈值分别做逐 batch 细扫。例如粗扫发现：

```text
batch=32  E2E=3.9s
batch=48  E2E=6.1s
batch=96  E2E=12.5s
batch=160 E2E=24s
batch=192 E2E=31s
```

脚本会按 `REFINE_BATCH_STEP` 跳步补测这些区间（默认 `+4`，尽量少跑点）：

```text
5s  边界：36,40,44
10s 边界：68,72,76,...,92
20s 边界：100,104,108,...,156
30s 边界：164,168,172,...,188
```

这样同一份日志/CSV 可以用于绘制 `E2E<=5s`、`E2E<=10s`、`E2E<=20s`、`E2E<=30s` 下的最大 batch 与 TPS。若一直跑到 `MAX_SWEEP_BATCH` 仍未超过 `E2E_LIMIT`，日志会提示 reached safety cap，此时结果代表被安全上限截断。

### 常用环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VLLM_NV_CONTAINER` | 自动检测，失败时为 `36` | NV 容器名 |
| `VLLM_XPU_CONTAINER` | 自动检测，失败时为 `36` | XPU 容器名 |
| `WEIGHTS_DIR` | 宿主机仓库同级 `weights` | 模型权重宿主机路径；bare-metal re-exec 时用于路径映射 |
| `MODEL_PATH` | 无 | 强制指定模型绝对路径，覆盖 `model_config.sh` |
| `E2E_LIMIT` | `30` | 主停止阈值；超过该秒数后停止当前配置 |
| `E2E_THRESHOLDS` | `5 10 20 30` | 绘图用 E2E 阈值；脚本会为这些阈值补测边界 |
| `MAX_SWEEP_BATCH` | `512` | 安全上限；防止一直达不到 `E2E_LIMIT` 时无限跑 |
| `BATCH_SWEEP_LIST` | `1 2 4 8 16 24 32 48 64 96 128 160 192 224 256 320 384 448 512` | 多硬件对齐的 batch 粗扫候选点 |
| `BATCH_SWEEP_STEP` | `4` | 未设置 `BATCH_SWEEP_LIST` 时的 fallback 粗扫步长 |
| `REFINE_BATCH_STEP` | `4` | 跨越 E2E 绘图阈值后的边界补测步长；默认每 `+4` 个 batch 测一次 |
| `FIX_PERMISSIONS` | `1` | 是否在解析前修复日志目录权限；设为 `0` 可跳过 |
| `CHOWN_USER` | 当前用户（优先 `SUDO_USER`） | 权限修复时使用的用户名 |
| `CHOWN_GROUP` | `CHOWN_USER` 对应主组 | 权限修复时使用的用户组 |

### 绘图建议

如果要画多硬件 TPS 对比，建议所有硬件使用同一组 `BATCH_SWEEP_LIST` 和 `E2E_THRESHOLDS`。默认配置已经覆盖：

```text
E2E<=5s, E2E<=10s, E2E<=20s, E2E<=30s
```

如果只想快速验证脚本，可以临时减少阈值或降低安全上限：

```bash
E2E_THRESHOLDS="30" MAX_SWEEP_BATCH=128 VLLM_NV_CONTAINER=vllm-{USER}-nv bash run_nearrt_sweep.sh 4b
```

如果正式出图，建议保留默认 `E2E_THRESHOLDS="5 10 20 30"`。默认 `REFINE_BATCH_STEP=4` 会在边界区间跳步补测，避免逐个 batch 跑；如果要更准，可临时设为 `REFINE_BATCH_STEP=1`。

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
