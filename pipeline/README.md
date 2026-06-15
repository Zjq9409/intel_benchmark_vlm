# Video VLM Analysis Pipeline

视频分析流水线。接受单个或多个视频，自动完成：

1. **解码** — ffmpeg 硬件/软件解码视频
2. **抽帧** — 按固定间隔提取关键帧（JPEG）
3. **VLM 分析** — 批量发送帧图片给视觉语言模型，输出结构化 JSON

LLM 生成视频 Summary 为**可选步骤**。

---

## 目录结构

\`\`\`
pipeline/
├── main.py            # CLI 入口
├── frame_extractor.py # 抽帧模块
├── vlm_analyzer.py    # VLM 分析模块（含可选 LLM 解说填充）
├── requirements.txt
└── README.md
\`\`\`

---

## 安装依赖

\`\`\`bash
pip install -r requirements.txt
\`\`\`

系统需已安装 ffmpeg（含 ffprobe）。

---

## 快速开始

### 完整流程：解码 → 抽帧 → VLM 分析

\`\`\`bash
python3 main.py run \
  --videos /home/intel/jane/intel_benchmark_vlm/pipeline/video/beijing_1080p.mp4 \
  --output ./output \
  --vlm-url http://localhost:8006/v1 \
  --vlm-model Qwen3.5-35B-A3B \
  --interval 5 \
  --batch-size 3 \
  --concurrency 1
\`\`\`

输出结构：
\`\`\`
/data/output/
└── video1/
    ├── frames/                    # 抽帧图片
    │   ├── keyframe_000000_000000000.jpg
    │   └── ...
    ├── analysis.json              # VLM 分析结果（含 video_summary 字段，若启用 LLM）
    └── summary.txt                # 视频整体内容总结（仅启用 LLM 时生成）
\`\`\`

### 开启 LLM 视频 Summary（可选）

```bash
python main.py run \
  --videos /home/intel/jane/intel_benchmark_vlm/pipeline/beijing_1080p.mp4 \
  --output ./output \
  --vlm-url http://localhost:8006/v1 \
  --vlm-model Qwen3-VL-4B-Instruct \
  --interval 60 \
  --batch-size 3 \
  --concurrency 1 \
  --llm-url http://localhost:8006/v1 \
  --llm-model Qwen3-VL-4B-Instruct \
  --hwaccel cuda \
  --hwaccel-device 5
```

传入 `--llm-url` 和 `--llm-model` 后，LLM 会汇总所有 VLM 片段描述，生成连贯的全视频内容总结，保存到 `summary.txt`，并写入 `analysis.json` 的 `video_summary` 字段。

不传 `--llm-url` 时跳过，输出与原来完全一致。

### 只抽帧

**Intel VAAPI 硬件解码：**

\`\`\`bash
python main.py extract \
  --videos /data/video1.mp4 \
  --output /data/output \
  --interval 5 \
  --hwaccel vaapi \
  --hwaccel-device /dev/dri/renderD128
\`\`\`

**NVIDIA CUDA 硬件解码：**

\`\`\`bash
python main.py extract \
  --videos /home/intel/jane/intel_benchmark_vlm/pipeline/video/beijing_1080p.mp4 \
  --output ./output \
  --interval 5 \
  --hwaccel cuda \
  --hwaccel-device 4
\`\`\`

--hwaccel-device 对 CUDA 传 GPU 编号（0、1…），对 VAAPI 传 DRM 设备路径。不传 --hwaccel 时使用 CPU 软解。

若设置了 CUDA_VISIBLE_DEVICES 环境变量，ffmpeg 中的 CUDA 设备编号是该变量子集内的序号，而非物理 GPU ID。
例如 CUDA_VISIBLE_DEVICES=4，则 ffmpeg 里 --hwaccel-device 0 对应物理 GPU 4。

硬件解码失败时自动降级到 CPU 软解，抽帧结束后输出 GPU 显存峰值/均值监控信息。

---

## analysis.json 结构

\`\`\`json
{
  "artifact_version": "pipeline-vlm-v1",
  "generated_at": "2026-04-20T...",
  "video_path": "/data/video1.mp4",
  "frame_interval_seconds": 5.0,
  "vision_model": "Qwen2-VL-7B-Instruct",
  "batches": ["..."],
  "video_summary": "（启用 LLM 时由 LLM 生成的全视频内容总结）",
  "video_clip_json": [
    {
      "_id": 1,
      "timestamp": "00:00:00,000-00:00:25,000",
      "picture": "画面整体描述",
      "OST": 2
    }
  ]
}
\`\`\`

---

## VLM 接口要求

- 兼容 OpenAI /v1/chat/completions 接口
- 支持多图输入（image_url 类型，base64 编码）
- 支持 vLLM、Ollama、本地部署模型等

---

## 参数说明

| 参数 | 子命令 | 说明 | 默认值 |
|------|--------|------|--------|
| --videos | run/extract | 输入视频路径（可多个） | — |
| --output | run/extract | 输出根目录 | — |
| --vlm-url | run | VLM API base URL | — |
| --vlm-model | run | VLM 模型名称 | — |
| --interval | run/extract | 抽帧间隔（秒） | 5 |
| --batch-size | run | 每次 VLM 调用发送的帧数 | 8 |
| --concurrency | run | 并发 VLM 请求数 | 2 |
| --timeout | run | VLM 单次请求超时（秒） | 120 |
| --hwaccel | run/extract | ffmpeg 硬件解码方式（可选）：vaapi / cuda / qsv | — |
| --hwaccel-device | run/extract | VAAPI：设备节点如 /dev/dri/renderD128；CUDA：GPU 编号如 0 | — |
| --llm-model | run | （可选）LLM 模型名称 | — |
