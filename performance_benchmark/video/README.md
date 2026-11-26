# 视频推理性能测试

本目录提供两类测试入口：

- `run_blog_benchmark.sh`：主基准入口，支持单配置测试和完整 GPU/CPU 矩阵；负责启动/停止 vLLM 服务并汇总性能数据。
- `run_video_bench_serve.sh`：对已运行的 OpenAI 兼容视频服务执行一次 `vllm bench serve` 压测。

性能矩阵视频位于 `videos_captioning/`，容器内路径为 `/llm/performance_benchmark/video/videos_captioning/`。仓库根目录挂载为 `/llm`。

## 初始化容器

若尚无可用容器，从仓库根目录运行：

```bash
bash performance_benchmark/video/setup_container.sh
```

可指定容器名和权重目录：

```bash
bash performance_benchmark/video/setup_container.sh vllm-video-nv /DISK0
```

矩阵测试默认使用模型 `Qwen3.5-9B`，权重源路径 `/DISK0/weights/Qwen3.5-9B`。可通过 `MODEL_NAME`、`MODEL_SRC`、`VLLM_CONTAINER` 覆盖。

## 完整 GPU/CPU 矩阵

```bash
bash performance_benchmark/video/run_blog_benchmark.sh --matrix
```

脚本自动检测 NVIDIA GPU 数量，并运行可用的 1、2、4、8 GPU 档位。每档分别测试两种 CPU profile：

| Profile | 含义 |
|---|---|
| `host_all` | 使用容器当前 CPU 集合；通常是主机的全部逻辑 CPU。 |
| `gold6530` | 默认模拟双路 Gold 6530：每路选择 32 个物理核及 SMT 线程，共 64 核/128 逻辑 CPU。 |

模拟单路 Gold 6530（32 核/64 线程）：

```bash
GOLD6530_SOCKETS=1 bash performance_benchmark/video/run_blog_benchmark.sh --matrix
```

只预览 GPU/CPU 测试组合而不启动服务：

```bash
DRY_RUN=1 bash performance_benchmark/video/run_blog_benchmark.sh --matrix
```

每个 GPU/CPU 配置会逐个运行以下负载：

- 编码：H.264、H.265
- 分辨率：720p、1080p、1440p（2K）
- 时长：10 秒、30 秒
- 解码后端：OpenCV、PyNvVideoCodec
- 默认每副本并发 32、每副本 100 个请求、抽取 16 帧、输出 100 token
- 默认 FP8、`Qwen3.5-9B`；每组视频单独运行，不把编码、分辨率或时长混在一起

每个 GPU/CPU 配置共 12 种视频条件 × 2 个后端。若主机有 8 张 GPU，矩阵最多包含 8 种 GPU/CPU 配置、96 条视频配置结果（每条各有两个后端指标），耗时较长。

### 视频矩阵说明

| 编码 | 目标码率 | 文件覆盖 |
|---|---:|---|
| H.264 | 约 8 Mb/s | 720p、1080p、1440p；每种 10s、30s |
| H.265 | 约 4 Mb/s | 720p、1080p、1440p；每种 10s、30s |

目标码率来自文件名，实际码率会随视频内容略有偏差。跨编码比较同时包含码率差异，不是严格控制码率后的纯编码对比。1440p 视频由 1080p 源放大生成，只适合测处理负载，不代表原生 2K 画质。素材是性能测试视频，不用于评估驾驶标签准确率。

### 矩阵结果和监控

默认结果目录：

```text
performance_benchmark/video/blog_repro/<时间戳>_gpu_cpu_matrix/
├── summary.json
├── host_all_1gpu/
│   ├── h264_720p_10s/
│   ├── h264_720p_30s/
│   └── ...
└── gold6530_8gpu/
    └── ...
```

根目录的 `summary.json` 是完整汇总，包含 CPU/GPU 配置、视频编码/分辨率/时长及后端性能指标。各测试子目录保留 `benchmark.log`、`startup.log`、`utilization.csv`，便于排障。

- OpenCV 主要查看 `host_cpu_util_pct`：测试期间的**整台主机** CPU 利用率。
- PyNvVideoCodec 主要查看 `nvdec_mean_util_pct` / `nvdec_max_util_pct`：NVDEC 解码器利用率。

CPU 利用率通过主机 `/proc/stat` 采样，不是容器独占统计。CPU 利用率与 NVDEC 利用率属于不同硬件单元，不应直接比较数值大小。Gold 6530 profile 只设置当前主机的 CPU affinity，不会模拟目标 CPU 的微架构、频率或内存带宽。

## 单配置测试

参数格式：

```text
run_blog_benchmark.sh [replicas] [cores|auto] [backend] [codec] [resolution]
```

示例：

```bash
# 2 GPU，容器当前全部 CPU，两个后端，H.265，720p
bash performance_benchmark/video/run_blog_benchmark.sh 2 auto both h265 720p

# 8 GPU，限制容器到逻辑 CPU 0-127，仅 OpenCV、H.264、全部分辨率
bash performance_benchmark/video/run_blog_benchmark.sh 8 128 opencv h264 all

# 指定非连续 CPU 集合并给结果加标签
CPUSET=0-15,26-41,104-119,130-145 CPU_PROFILE=custom \
  bash performance_benchmark/video/run_blog_benchmark.sh 1 auto both h265 1080p
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `replicas` | `8` | GPU 副本数；默认使用 GPU `0..replicas-1`。 |
| `cores` | `auto` | `auto` 保留容器原 cpuset；正整数 `N` 使用逻辑 CPU `0..N-1`。 |
| `backend` | `both` | `opencv`、`pynvvideocodec` 或 `both`。 |
| `codec` | `both` | `h264`、`h265` 或 `both`。 |
| `resolution` | `all` | `720p`、`1080p`、`2k`/`1440p` 或 `all`。 |

常用环境变量：`GPU_LIST`、`CPUSET`、`CPU_PROFILE`、`MATRIX_ROOT`、`MODEL_NAME`、`MODEL_SRC`、`CONCURRENCY`、`NUM_PROMPTS`、`FRAMES`、`OUTPUT_LEN`、`QUANT`、`MAX_MODEL_LEN`、`MAX_NUM_SEQS`、`GPU_MEM_UTIL`、`API_SERVER_COUNT`、`HW_DECODERS`、`MM_IPC_GB`、`USE_MPS`。

## 对已运行服务执行一次压测

`run_video_bench_serve.sh` 需要一个已启动、允许读取本地媒体的 OpenAI 兼容视频服务：

```bash
# 参数：模型名、端口、并发列表、每档请求数
bash performance_benchmark/video/run_video_bench_serve.sh "" 8012 "1 4 8" 16

# VIDEO_DIRS 可传一个或多个容器内的视频目录/视频文件路径
VIDEO_DIRS=/llm/performance_benchmark/video/videos_captioning \
  bash performance_benchmark/video/run_video_bench_serve.sh "" 8012 "1 4 8" 16
```

默认并发为 `1 4 8`，每档 16 个请求，输出长度 128 token。常用环境变量：`VIDEO_DIRS`、`PROMPT`、`OUTPUT_LEN`、`TOKENIZER`、`HOST`、`EXTRA_ARGS`、`RESULT_DIR`、`BENCH_LABEL`。脚本会生成 ShareGPT 视频数据集，调用官方 `vllm bench serve` 并保存基准 JSON 和日志。

## 注意事项

- PyNvVideoCodec/NVDEC 需要 NVIDIA GPU 和匹配的 NVIDIA 容器；H.265 还需要容器支持 HEVC 解码。
- 运行矩阵前确认权重路径、视频文件和 GPU 可用；矩阵入口会自动启动和停止每轮服务。
- 需要单独对比编码或分辨率时，选择对应的 `codec`/`resolution` 参数，避免把多个条件混在一组结果里。
