# vLLM 在线性能测试

## 运行性能测试

无需下载数据集，直接运行 benchmark 脚本即可。脚本会自动启动 vllm 服务端并依次测试各并发量：

```bash
cd performance_benchmark/online

# 准实时场景扫描（自动 batch sweep）
bash run_nearrt_sweep.sh 4b 4 fp8      # 4B模型，GPU 4，FP8量化

# 批量运行多个组合（见 run_both.sh）
bash run_both.sh

# 批量运行，关闭量化，指定 device 4
bash run_both.sh none 4
```

### run_nearrt_sweep.sh 参数

| 位置参数 | 可选值 | 默认值 | 说明 |
|---------|--------|--------|------|
| `$1` 模型规格 | `4b` / `q35-4b` / `30b` / `32b` / `q36-35b` | `4b` | 选择测试模型（详见下表） |
| `$2` GPU Device | 设备 ID，如 `4` | 空（全部） | 指定 `CUDA_VISIBLE_DEVICES`，空则使用全部可见 GPU |
| `$3` 量化 | `fp8` / `none` | `fp8` | FP8 量化（RTX 4090/Ada Lovelace 有效）；`none` 为纯 FP16 |
| `$4` MTP | `on` / `off` | `off` | 是否启用 Speculative Decoding（仅 Qwen3.5 系列支持） |

### vllm_random_benchmark_server.sh 参数

内层脚本可独立调用，参数如下：

| 位置参数 | 可选值 | 默认值 | 说明 |
|---------|--------|--------|------|
| `$1` 模型规格 | `4b` / `q35-4b` / `30b` 等 | `30b` | 选择测试模型 |
| `$2` 图片宽度 | 任意整数 | `1280` | 随机生成图片的宽度（像素） |
| `$3` 图片高度 | 任意整数 | `720` | 随机生成图片的高度（像素） |
| `$4` 每请求图片数 | 任意正整数 | `1` | 单图测试填 `1`；模拟 NarratoAI 真实负载填 `10` |
| `$5` MTP | `on` / `off` | `off` | 是否启用 Speculative Decoding |
| `$6` 量化 | `fp8` / `none` | `fp8` | FP8 量化 |
| `$7` GPU Device | 设备 ID，如 `4` | 空（全部） | 指定 `CUDA_VISIBLE_DEVICES` |
| `$8` 输出长度 | 整数 | `1024` | 随机输出 token 长度 |
| `$9` 输入长度 | 整数 | `1024` | 随机输入 token 长度 |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_NV_CONTAINER` | `vllm-nv-container` | NVIDIA 路径使用的 Docker 容器名 |
| `VLLM_XPU_CONTAINER` | `lsv-container-b8` | Intel XPU 路径使用的 Docker 容器名 |

可在 `run_both.sh` 开头修改，或在 shell 中 `export` 后执行脚本。

### 可调参数（run_nearrt_sweep.sh 顶部）

以下参数集中在 `run_nearrt_sweep.sh` 脚本顶部，修改一处即可同步到内层脚本：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `E2E_LIMIT` | `40` | E2E 阈值（秒），超过时停止 batch sweep |
| `PORT` | `8008` | vllm 服务端口 |
| `MAX_BATCHED_TOKENS` | `32768` | 最大批处理 token 数 |
| `MAX_MODEL_LEN` | `32768` | 最大模型上下文长度 |
| `GPU_MEM_UTIL` | `0.9` | GPU 显存利用率 |

### 支持的模型

| `$1` 值 | 模型名称 | TP | 权重路径（容器内） |
|---------|---------|-----|-------------------|
| `30b`（默认） | Qwen3-VL-30B-A3B-Instruct | 4 | `/llm/models/Qwen3-VL-30B-A3B-Instruct` |
| `4b` | Qwen3-VL-4B-Instruct | 1 | `/llm/models/Qwen3-VL-4B-Instruct` |
| `q35-4b` | Qwen3.5-4B | 1 | `/llm/models/Qwen3.5-4B` |
| `32b` | Qwen3-VL-32B-Instruct | 4 | `/llm/models/Qwen3-VL-32B-Instruct` |
| `q36-35b` | Qwen3.6-35B-A3B | 4 | `/DISK0/Qwen3.6-35B-A3B` |

### 服务端配置

| 参数 | 单图模式（`$4=1`） | 多图模式（`$4>1`） | 说明 |
|------|-----------------|-----------------|------|
| `PORT` | `8008` | `8008` | 服务端口（可在脚本顶部调） |
| `MAX_BATCHED_TOKENS` | `32768` | `32768` | 最大批处理 token 数（可在脚本顶部调） |
| `MAX_MODEL_LEN` | `32768` | `32768` | 最大模型上下文长度（可在脚本顶部调） |
| `GPU_MEM_UTIL` | `0.9` | `0.9` | GPU 显存利用率（可在脚本顶部调） |
| `INPUT_LEN` | `1024` | `1024` | 随机输入 token 长度 |
| `OUTPUT_LEN` | `1024` | `1024` | 随机输出 token 长度 |
| `MAX_BSIZE` | `200` | `20` | batch sweep 上限 |
| E2E 阈值 | `40s` | `40s` | 超过此值时停止 batch sweep（`E2E_LIMIT` 变量） |
| 步长 | 动态计算 | 动态计算 | 根据 batch=1 的 E2E 自动调整（1~5） |

MTP（Speculative Decoding）配置：`{"method":"qwen3_next_mtp","num_speculative_tokens":2}`，通过 `$5=on` 追加到服务端启动参数，仅 Qwen3.5 系列支持。

### 日志文件命名规则

所有日志统一保存在 `LOG/<MODEL_NAME>/` 子目录下，文件名格式为：

```
<MODEL_NAME>/<YYYYMMDD_HHMMSS>_[dev<ID>_][fp8_|fp16_][mtp_|nomtp_]<N>_<W>x<H>_tp<TP>_mbt<MBT>_in<IN>_out<OUT>_<GPU>.log
```

| 前缀 | 出现条件 | 示例 |
|------|---------|------|
| `dev4_` | `$7` 指定了 device ID | `dev4_fp8_nomtp_...log` |
| `fp8_` | `$6 = fp8`（默认） | `fp8_nomtp_...log` |
| `fp16_` | `$6 = none` | `fp16_nomtp_...log` |
| `mtp_` | `$5 = on` | `mtp_...log` |
| `nomtp_` | `$5 = off` | `nomtp_...log` |

示例：
```
Qwen3.5-4B/20260507_153000_dev4_fp8_nomtp_1_1280x720_tp1_mbt8192_in1024_out1024_RTX4090D.log
```

XPU 运行时还会生成 GPU 监控日志（`monitor_gpu.sh` 采集，5 秒间隔）：
```
LOG/<MODEL_NAME>/<YYYYMMDD_HHMMSS>_[mm<N>_][mtp_|nomtp_]monitor_...log
```

## 测试结果

测试脚本会输出以下性能指标：
- **Request throughput**: 请求吞吐量（req/s）
- **Output token throughput**: 输出 token 吞吐量（tok/s）
- **TTFT** (Time to First Token): 首 token 延迟
- **TPOT** (Time per Output Token): 每个输出 token 的平均时间
- **ITL** (Inter-token Latency): token 间延迟
- **Total Token throughput**: 总 token 吞吐量（输入+输出）
- **Acceptance rate**: MTP 投机解码接受率（仅 `$5=on` 时输出）


## Batch Sweep 自动测试

使用 `run_nearrt_sweep.sh` 进行准实时场景的批量测试，会自动扫描多种配置组合：

```bash
bash run_nearrt_sweep.sh [model] [device] [quant] [mtp]

# 示例
bash run_nearrt_sweep.sh 4b 4 fp8 off    # 4B模型，GPU 4，FP8量化，MTP关闭
bash run_nearrt_sweep.sh 4b "" fp8       # 使用所有GPU
```

**测试矩阵**：
- 分辨率：`720p (1280x720)`、`1080p (1920x1080)`
- 输入长度：`1024 tokens`
- 输出长度：`1024 tokens`
- 每请求图片数：`1, 4, 6, 8, 10, 14, 16` 张
- Batch 大小：动态扫描至 E2E 超过 40 秒

**自动化特性**：
1. 单次启动服务器，复用测试所有组合
2. 每个图片数量自动寻找最大 batch（E2E < 40s）
3. 自动修复 docker 创建的文件权限
4. 统一输出到单个日志文件，便于对比

## 解析测试日志

测试完成后脚本会自动调用 `parse_log.py` 生成 CSV，也可手动解析：

```bash
python3 parse_log.py LOG/<MODEL_NAME>/xxx_client_xxx.log
# 生成同名 .csv 文件，包含各并发点的 TTFT/TPOT/ITL/throughput 等指标
```

**CSV 字段说明**：
- `batch_size`: 并发批次大小（同时处理的请求数）
- `Images per request`: 每个请求包含的图片数量
- `TTFT (ms)`: Time to First Token（首 token 延迟）
- `TPOT (ms)`: Time per Output Token（每个输出 token 平均时间）
- `TPS (tokens/s)`: 输出 token 吞吐量
- `QPS (req/s)`: 请求吞吐量
- `E2E Latency (s)`: 端到端延迟（Benchmark duration）

## NarratoAI 真实负载场景

NarratoAI 生产配置（`config.toml`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `vision_max_concurrency` | `2` | 同时在途请求数 |
| `vision_batch_size` | `10` | 每请求携带图片数 |
| `frame_interval_input` | `3` | 每 3 秒取一帧（源视频 720P） |
| 图片预处理 | `thumbnail(1024,1024)` | 超出 1024px 时缩放 |

对应 benchmark 命令：

```bash
# 精准模拟 NarratoAI 负载（10张图/请求，并发扫描 1~20）
bash vllm_random_benchmark_server.sh q35-4b 1280 720 10 on
```

## 测试其他模型

所有模型配置集中在 `model_config.sh`，只需在 `case` 中新增一条：

```bash
8b)
    MODEL_DIR="Qwen3-VL-8B-Instruct"
    SERVER_MODEL="/llm/models/Qwen3-VL-8B-Instruct"
    TP=2
    ;;
```

确认：
1. 模型权重已放置在容器可访问的路径，`SERVER_MODEL` 填容器内绝对路径
2. `TP` 值与可用 GPU 卡数匹配
3. MTP 仅适用于 Qwen3.5 系列，其他模型请使用 `$5=off`（默认）
