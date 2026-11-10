# vLLM 视频（Video）在线性能测试

三步：**起容器 → 起服务 → 跑测试**。

输入数据是真实视频文件（不是合成图片）：

| 宿主机目录 | 容器内路径 | 内容 |
|---|---|---|
| `intel_benchmark_vlm/videos_h264/` | `/llm/videos_h264` | H.264，720p / 1080p |
| `intel_benchmark_vlm/videos_h265/` | `/llm/videos_h265` | H.265(HEVC)，720p / 1080p |

> 仓库根目录被挂载到容器内 `/llm`，所以视频在容器里就是 `/llm/videos_h26x/*.mp4`。

---

## 1. 起容器

```bash
bash setup_container.sh                      # 自动检测 NV / Intel XPU，容器名 vllm-video-{nv,xpu}
bash setup_container.sh vllm-jane-nv /DISK0  # 指定容器名和权重目录
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `$1` 容器名 | `vllm-video-nv` / `vllm-video-xpu` | 也可用 `VLLM_CONTAINER` |
| `$2` 权重目录 | `/home/intel/LLM/weights/` | 挂到容器内 `/llm/models` |
| `NV_IMAGE` | `vllm/vllm-openai:v0.19.1-cu130` | NV 镜像 |
| `INTEL_IMAGE` | `vllm/vllm-openai-xpu:v0.32.0` | XPU 镜像 |

---

## 2. 起服务

```bash
bash start_video_server.sh Qwen3.5-2B              # 最常用：直接写模型目录名
bash start_video_server.sh 4b 8012 2               # 预置 key / 端口 / TP
VLLM_CONTAINER=vllm-jane-nv bash start_video_server.sh 4b
```

裸机执行会自动 `docker exec` 进容器。

**换模型不用改任何文件**，`$1` 支持四种写法：

```bash
bash start_video_server.sh list                     # 列出 /llm/models 下所有模型
bash start_video_server.sh Qwen3.5-2B     # 目录名
bash start_video_server.sh /llm/models/my-model     # 容器内绝对路径
bash start_video_server.sh 4b                       # 预置短 key
```

预置 key 复用 `../online/model_config.sh`
（`4b` / `q35-4b` / `32b` / `q36-35b` / `q30-a3b` / `q36-27b`）。模型找不到时会自动列出可用模型。

| 位置 | 默认 | 说明 |
|---|---|---|
| `$1` 模型 | `4b` | 预置 key / `/llm/models` 下的目录名 / 绝对路径 / `list` |
| `$2` 端口 | `8012` | |
| `$3` TP | 模型默认 | tensor parallel |
| `$4` device | 全部 | `CUDA_VISIBLE_DEVICES`，如 `0,1` |

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `VIDEO_LIMIT` | `1` | 每请求最多几个视频 → `--limit-mm-per-prompt` |
| `VIDEO_NUM_FRAMES` | `16` | 解码采样帧数 |
| `VIDEO_BACKEND` | `opencv` | 解码后端，见下表 |
| `MAX_MODEL_LEN` | `32768` | |
| `GPU_MEM_UTIL` | `0.9` | |
| `MAX_NUM_SEQS` | 空 | vllm 默认 |
| `QUANT` | `none` | `fp8` / `none` |
| `MEDIA_PATH` | `/llm` | `--allowed-local-media-path`，必须包含视频目录 |
| `VLLM_CONTAINER` | 自动探测 | 容器名。不指定时按 `VLLM_NV_CONTAINER`/`VLLM_XPU_CONTAINER` → `docker ps` 里第一个名字含 `vllm` 的容器 → `vllm-video-{nv,xpu}` 依次兜底 |

### 解码后端

视频解码默认走 CPU（OpenCV + FFMPEG），兼容性最好，但高并发 / 长视频时 CPU 容易成为瓶颈。
NVIDIA 平台可切到 NVDEC 硬解，帧直接留在显存里。

| `VIDEO_BACKEND` | 生成的 `--media-io-kwargs` | 适用场景 |
|---|---|---|
| `opencv`（默认） | `{"video":{"backend":"opencv","num_frames":16}}` | 小规模 / 偶尔推理，开箱即用 |
| `pynvvideocodec` | `{"video":{"backend":"pynvvideocodec","min_frames":16,"max_frames":16,"hw_decoders":2}}` | 高吞吐 / 多 GPU 生产环境，官方主推 |
| `torchcodec` | `{"video":{"backend":"torchcodec","device":"cuda","num_frames":16}}` | 解码帧直接输出为 GPU tensor |
| `deepstream` | `{"video":{"backend":"deepstream","fps":1}}` | 流式视频源，需 `vllm[deepstream]` |

```bash
VIDEO_BACKEND=pynvvideocodec MM_IPC_GPU_MEM_GB=4 bash start_video_server.sh 4b
VIDEO_IO_KWARGS='{"video":{"backend":"opencv","num_frames":32}}' bash start_video_server.sh 4b
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `HW_DECODERS` | `2` | `pynvvideocodec` 用的硬件解码器数量 |
| `DEEPSTREAM_FPS` | `1` | `deepstream` 抽帧 fps |
| `MM_IPC_GPU_MEM_GB` | 空 | 设了才加 `--mm-ipc-gpu-memory-gb`，NVDEC 路径建议配 CUDA MPS 一起用 |
| `VIDEO_IO_KWARGS` | 空 | 直接给原始 JSON，设了就忽略上面的自动拼装 |

> NVDEC 相关后端（`pynvvideocodec` / `torchcodec device=cuda`）只能在 NV 容器上用，XPU 上请保持 `opencv`。

---

## 3. 跑测试

### 推荐：官方 `vllm bench serve`（`run_video_bench_serve.sh`）

```bash
bash run_video_bench_serve.sh                      # 模型自动探测，并发 1 4 8，16 条
bash run_video_bench_serve.sh "" 8012 "1 8 16" 32
VIDEO_DIRS=/llm/videos_h265 bash run_video_bench_serve.sh
```

原理：vLLM 的 ShareGPT dataset loader 支持条目里的 `"video"` 字段，`process_video()`
会把本地路径转成 `file://`。脚本把 `VIDEO_DIRS` 下的 mp4 自动包成 ShareGPT 格式的
JSON，再交给官方 `vllm bench serve`，所以输出就是原生的
`============ Serving Benchmark Result ============` 块。

| 位置 | 默认 | 说明 |
|---|---|---|
| `$1` 模型名 | 自动从 `/v1/models` 读 | 留空即可 |
| `$2` 端口 | `8012` | |
| `$3` 并发列表 | `1 4 8` | 每个并发度跑一轮 |
| `$4` 请求数 | `16` | `--num-prompts` |

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `VIDEO_DIRS` | `/llm/videos_h264 /llm/videos_h265` | 视频目录，空格分隔 |
| `PROMPT` | `Describe what happens in this video in detail.` | |
| `OUTPUT_LEN` | `128` | `--sharegpt-output-len` |
| `TOKENIZER` | 自动 `/llm/models/<model>` | served name 不是路径时需要 |
| `HOST` | `localhost` | |
| `EXTRA_ARGS` | 空 | 透传给 `vllm bench serve` |

```
============ Serving Benchmark Result ============
Successful requests:                     16
Benchmark duration (s):                  74.13
Total input tokens:                      193300
Total generated tokens:                  2048
Request throughput (req/s):              0.22
Output token throughput (tok/s):         27.63
Total token throughput (tok/s):          2635.28
---------------Time to First Token----------------
Mean TTFT (ms):                          3286.61
...
```

### 轻量版：纯 bash + curl（`run_video_benchmark.sh`）

无 Python 依赖，适合容器里没装 `vllm` CLI 的场景。指标是自己用
`curl -w '%{time_starttransfer} %{time_total}'` 算的，TTFT 只是近似值。

```bash
bash run_video_benchmark.sh                        # 模型自动探测
bash run_video_benchmark.sh "" 8012 "1 4 8" 128
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `VIDEO_DIRS` | `/llm/videos_h264 /llm/videos_h265` | |
| `NUM_REQUESTS` | = 并发度 | 每轮请求总数 |
| `IGNORE_EOS` | `true` | 固定输出长度，吞吐数据才可比 |
| `WARMUP` | `1` | `0` 关闭预热 |
| `TIMEOUT` | `600` | 单请求超时（秒） |

---

## 日志

服务端和客户端日志都落在同一个模型目录下（宿主机 / 容器内路径相同）：

```
video/Qwen3.5-2B/
├── 20261102_125759_video_XPU_server.log       # 服务端 (start_video_server.sh)
├── 20261102_132847_bench/                      # 一次 bench serve 运行
│   ├── local_videos_sharegpt.json              # 自动生成的数据集
│   ├── bench_serve.log                         # 完整控制台输出
│   └── openai-chat-infqps-concurrency{1,4,8}-*.json   # vLLM --save-result
└── 20261102_210547_video_client.log            # curl 版客户端日志
```

服务端日志名里只有量化时才带 quant 标记（`..._video_fp8_XPU_server.log`）。

server PID 存在容器内 `/tmp/vllm_video_server_<port>.pid`，停服务：

```bash
sudo docker exec <容器名> bash -c 'kill $(cat /tmp/vllm_video_server_8012.pid)'
```

---

## 常见问题

- **日志目录 Permission denied**：服务端在容器里以 root 建目录。`start_video_server.sh`
  已自动 `chmod 777`；老目录手动 `sudo chmod 777 <模型名>` 即可，客户端也会自动退回 `/tmp`。
- **`Couldn't instantiate the backend tokenizer`**：`vllm bench serve` 的 `--tokenizer`
  需要指到权重目录而不是 served name，脚本已自动处理。
- **HTTP 400 / 解码失败**：H.265 需要容器内解码器带 HEVC 支持，不支持就设
  `VIDEO_DIRS=/llm/videos_h264` 只测 H.264。
- **`local media path not allowed`**：`MEDIA_PATH` 必须是视频目录的父路径（默认 `/llm`）。
- **输入 token 比预期多很多**：Qwen3-VL 的 transformers 实现默认不做 per-frame pixel cap，
  可加 `--mm-processor-kwargs '{"cap_pixels_per_frame": true}'` 对齐 qwen-vl-utils 行为，
  或调小 `VIDEO_NUM_FRAMES`。
- **跑的是合成多图而不是真实视频**：那是 `../online/` 的 `random-mm` 路径，本目录才是真实视频文件。

---

## 准备视频数据

从任意 mp4 转出 720p / 1080p 两种分辨率：

```bash
ffmpeg -i src.mp4 -vf scale=1280:720  -c:v libx264 -preset medium -crf 23 videos_h264/src_720p_h264.mp4
ffmpeg -i src.mp4 -vf scale=1920:1080 -c:v libx265 -preset medium -crf 23 -tag:v hvc1 videos_h265/src_1080p_h265.mp4
```
