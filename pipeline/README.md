# Video VLM Analysis Pipeline

视频分析流水线。接受单个或多个视频，自动完成：

1. **解码** — ffmpeg 硬件/软件解码视频
2. **抽帧** — 按固定间隔提取关键帧（JPEG）
3. **VLM 分析** — 支持两种输入模式：
  - `single`（默认）：1 个 prompt + 1 张图（逐帧并发，对齐客户流程）
  - `batch`：1 个 prompt + 多张图（按批处理）

LLM 生成解说词和视频 Summary 为**可选步骤**。

---

## 目录结构

```
pipeline/
├── main.py            # CLI 入口
├── frame_extractor.py # 抽帧模块
├── vlm_analyzer.py    # VLM 分析模块（含可选 LLM 解说填充）
├── requirements.txt
└── README.md
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

> 系统需已安装 `ffmpeg`（含 `ffprobe`）。

---

## 快速开始

### 完整流程：单图模式（默认，和客户流程对齐，单个prompt+单图）

- CPU抽帧 + VLM推理
```bash
python main.py run \
  --videos /home/intel/jane/intel_benchmark_vlm/pipeline/video/beijing_1080p.mp4 \
  --output /home/intel/jane/intel_benchmark_vlm/pipeline/output \
  --vlm-url http://localhost:8006/v1 \
  --vlm-model Qwen3.5-35B-A3B \
  --vlm-mode single \
  --interval 5 \
  --concurrency 2
```

-  Nvidia GPU抽帧 + VLM推理
```bash
python main.py run \
  --videos /home/intel/jane/intel_benchmark_vlm/pipeline/video/beijing_1080p.mp4 \
  --output /home/intel/jane/intel_benchmark_vlm/pipeline/output \
  --vlm-url http://localhost:8006/v1 \
  --vlm-model Qwen3.5-35B-A3B \
  --vlm-mode single \
  --interval 5 \
  --hwaccel cuda \
  --hwaccel-device 0
  --concurrency 2
```

-  Intel ARC 抽帧 + VLM推理
```bash
python main.py run \
  --videos /home/intel/jane/intel_benchmark_vlm/pipeline/video/beijing_1080p.mp4 \
  --output /home/intel/jane/intel_benchmark_vlm/pipeline/output \
  --vlm-url http://localhost:8006/v1 \
  --vlm-model Qwen3.5-35B-A3B \
  --vlm-mode single \
  --interval 5 \
  --hwaccel vaapi \
  --hwaccel-device /dev/dri/renderD128
  --concurrency 2
```

说明：

- `single` 表示 **1 个 prompt + 1 张图**，逐帧并发调用 VLM。
- 该模式下 `--batch-size` **不生效**，可以不写。
- `--concurrency` 仍然生效，表示同时并发多少个单图请求。

### 完整流程：多图批处理模式（性能优先，单个prompt+多图）

```bash
python main.py run \
  --videos /data/video1.mp4 /data/video2.mp4 \
  --output /data/output \
  --vlm-url http://localhost:8000/v1 \
  --vlm-model Qwen2-VL-7B-Instruct \
  --vlm-mode batch \
  --interval 5 \
  --batch-size 8 \
  --concurrency 2
```

说明：

- `batch` 表示 **1 个 prompt + 多张图**。
- `--batch-size` 表示每次 VLM 请求打包多少张图。
- `--concurrency` 表示同时并发多少个批次请求。

输出结构：
```
/data/output/
└── video1/
    ├── frames/                    # 抽帧图片
    │   ├── keyframe_000000_000000000.jpg
    │   └── ...
    ├── analysis.json              # VLM 分析结果（含 video_summary 字段，若启用 LLM）
    └── summary.txt                # 视频整体内容总结（仅启用 LLM 时生成）
```

### 只抽帧

**Intel VAAPI 硬件解码：**

```bash
python main.py extract \
  --videos /data/video1.mp4 \
  --output /data/output \
  --interval 5 \
  --hwaccel vaapi \
  --hwaccel-device /dev/dri/renderD128
```

**NVIDIA CUDA 硬件解码：**

```bash
python main.py extract \
  --videos /data/video1.mp4 \
  --output /data/output \
  --interval 5 \
  --hwaccel cuda \
  --hwaccel-device 0
```

`--hwaccel-device` 对 CUDA 传 GPU 编号（`0`、`1`…），对 VAAPI 传 DRM 设备路径。不传 `--hwaccel` 时使用 CPU 软解。

> **注意：** 若设置了 `CUDA_VISIBLE_DEVICES` 环境变量，ffmpeg 中的 CUDA 设备编号是该变量子集内的**序号**，而非物理 GPU ID。
> 例如 `CUDA_VISIBLE_DEVICES=4`，则 ffmpeg 里 `--hwaccel-device 0` 对应物理 GPU 4，不应传 `4`。

硬件解码失败时自动降级到 CPU 软解，抽帧结束后输出 GPU 显存峰值/均值监控信息。
