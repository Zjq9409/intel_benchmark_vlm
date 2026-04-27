# vLLM 在线性能测试

## 运行性能测试

无需下载数据集，直接运行 benchmark 脚本即可。脚本会自动启动 vllm 服务端并依次测试各并发量：

```bash
cd performance_benchmark/online

# 测试 30B 模型（默认），图片 1280×720（默认）
bash vllm_random_benchmark_server.sh

# 测试 4B 模型，图片 1280×720
bash vllm_random_benchmark_server.sh 4b

# 测试 Qwen3.5-4B，单图，MTP 开启
bash vllm_random_benchmark_server.sh q35-4b 1280 720 1 on

# 测试 Qwen3.5-4B，多图（10张/请求，模拟 NarratoAI），MTP 开启
bash vllm_random_benchmark_server.sh q35-4b 1280 720 10 on

# 批量运行多个组合（见 run_both.sh）
bash run_both.sh
```

### 脚本参数

| 位置参数 | 可选值 | 默认值 | 说明 |
|---------|--------|--------|------|
| `$1` 模型规格 | `4b` / `q35-4b` / `30b` | `30b` | 选择测试模型（详见下表） |
| `$2` 图片宽度 | 任意整数 | `1280` | 随机生成图片的宽度（像素） |
| `$3` 图片高度 | 任意整数 | `720` | 随机生成图片的高度（像素） |
| `$4` 每请求图片数 | 任意正整数 | `1` | 单图测试填 `1`；模拟 NarratoAI 真实负载填 `10` |
| `$5` MTP | `on` / `off` | `off` | 是否启用 Speculative Decoding（仅 Qwen3.5 系列支持） |

### 支持的模型

| `$1` 值 | 模型名称 | TP |
|---------|---------|-----|
| `30b`（默认） | Qwen3-VL-30B-A3B-Instruct | 4 |
| `4b` | Qwen3-VL-4B-Instruct | 1 |
| `q35-4b` | Qwen3.5-4B | 1 |

### 服务端配置

| 参数 | 单图模式（`$4=1`） | 多图模式（`$4>1`） | 说明 |
|------|-----------------|-----------------|------|
| `PORT` | `8006` | `8006` | 服务端口 |
| `MAX_BATCHED_TOKENS` | `8192` | `32768` | 最大批处理 token 数 |
| `MAX_MODEL_LEN` | `16384` | `32768` | 最大模型上下文长度 |
| `GPU_MEM_UTIL` | `0.8` | `0.8` | GPU 显存利用率 |
| `INPUT_LEN` | `1024` | `1024` | 随机输入 token 长度 |
| `OUTPUT_LEN` | `1024` | `1024` | 随机输出 token 长度 |
| `MAX_BSIZE` | `200` | `20` | num-prompts 最大值 |
| TTFT 阈值 | `5000 ms` | `10000 ms` | 超过此值时自动停止 |
| 步长 | `2` | `1` | num-prompts 递增步长 |

MTP（Speculative Decoding）配置：`{"method":"qwen3_next_mtp","num_speculative_tokens":2}`，通过 `$5=on` 追加到服务端启动参数，仅 Qwen3.5 系列支持。

### 日志文件命名规则

所有日志统一保存在 `LOG/<MODEL_NAME>/` 子目录下，文件名格式为：

```
LOG/<MODEL_NAME>/<YYYYMMDD_HHMMSS>_[mm<N>_][mtp_|nomtp_]client_tp<TP>_mbt<MBT>_<W>x<H>_in<IN>_out<OUT>_<GPU>.log
LOG/<MODEL_NAME>/<YYYYMMDD_HHMMSS>_[mm<N>_][mtp_|nomtp_]server_tp<TP>_mbt<MBT>_<W>x<H>_in<IN>_out<OUT>_<GPU>.log
```

| 前缀 | 出现条件 | 示例 |
|------|---------|------|
| `mm10_` | `$4 > 1`（多图） | `mm10_mtp_client_...log` |
| `mtp_` | `$5 = on` | `mtp_client_...log` |
| `nomtp_` | `$5 = off` | `nomtp_client_...log` |

示例：
```
LOG/Qwen3.5-4B/20260427_153000_mm10_mtp_client_tp1_mbt32768_1280x720_in1024_out1024_RTX4090D.log
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

## 解析测试日志

测试完成后脚本会自动调用 `parse_log.py` 生成 CSV，也可手动解析：

```bash
python3 parse_log.py LOG/<MODEL_NAME>/xxx_client_xxx.log
# 生成同名 .csv 文件，包含各并发点的 TTFT/TPOT/ITL/throughput 等指标
```

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

在 `vllm_random_benchmark_server.sh` 的模型选择分支中添加新条目：

```bash
elif [ "$MODEL_SELECT" = "8b" ]; then
    SERVER_MODEL="/llm/models/Qwen3-VL-8B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-8B-Instruct"
    TP=2
```

确认：
1. 模型权重已放置在宿主机 `weights/` 目录（容器内映射为 `/llm/models/`）
2. `TP` 值与可用 GPU 卡数匹配
3. MTP 仅适用于 Qwen3.5 系列，其他模型请使用 `$5=off`（默认）
