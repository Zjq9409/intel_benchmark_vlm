# Video VLM Pipeline

基于 ffmpeg + OpenAI 兼容 VLM 接口的视频分析与编码流水线，支持 Intel VAAPI 和 NVIDIA NVENC 硬件加速。

## 目录结构

```
pipeline/
├── frame_extractor.py   # ffmpeg 抽帧模块
├── vlm_analyzer.py      # VLM 视觉分析模块（OpenAI 兼容接口）
├── video_encoder.py     # 视频裁剪 + 拼接模块（硬件加速）
├── main.py              # CLI 主入口
└── requirements.txt     # Python 依赖
```

## 依赖安装

```bash
pip install -r requirements.txt
```

系统依赖：`ffmpeg`（需支持目标硬件加速）

---

## 硬件加速

`--encoder` 为必填参数，可选值：

| 值 | 适用硬件 |
|----|---------|
| `h264_vaapi` | Intel Arc / Intel GPU（通过 `/dev/dri/renderD128`） |
| `h264_nvenc` | NVIDIA GPU |
| `libx264`    | 软件编码（无硬件要求） |

Intel VAAPI 需额外指定 `--vaapi-device`（默认 `/dev/dri/renderD128`），查看可用节点：

```bash
ls /dev/dri/render*
```

---

## 使用方法

### 完整流程：抽帧 → VLM 分析 → 视频编码

**Intel Arc / VAAPI：**

```bash
python main.py run \
  --videos  /home/intel/media_ai/video/10722664_MotionElements_athletic-fitness-man-running-in-urban-city_preview.mp4 /home/intel/media_ai/video/24303216_MotionElements_a-view-of-the-city-of-istanbul-and-bebek-neighborhood-and-rain_preview.mp4 \
  --output /home/intel/media_ai/output \
  --vlm-url http://10.112.234.173:8000/v1 \
  --vlm-key intel123 \
  --vlm-model Qwen3-VL-4B-Instruct \
  --interval 5 \
  --encoder h264_vaapi \
  --vaapi-device /dev/dri/renderD128
```

**NVIDIA GPU / NVENC：**

```bash
python main.py run \
  --videos /home/intel/jane/video/10722664_MotionElements_athletic-fitness-man-running-in-urban-city_preview.mp4 /home/intel/jane/video/24303216_MotionElements_a-view-of-the-city-of-istanbul-and-bebek-neighborhood-and-rain_preview.mp4 \
  --output /home/intel/jane/output \
  --vlm-url http://10.112.234.173:8000/v1 \
  --vlm-key intel123 \
  --vlm-model Qwen3-VL-4B-Instruct \
  --interval 5 \
  --encoder h264_nvenc
```

### 可选：LLM 生成旁白解说词

```bash
python main.py run ... \
  --llm-url http://localhost:8001/v1 \
  --llm-key EMPTY \
  --llm-model Qwen2.5-7B
```

LLM 为可选步骤。不传 `--llm-url` 时跳过，`narration` 字段保持为空。

### 只抽帧

```bash
python main.py extract \
  --videos /data/v.mp4 \
  --output /data/output \
  --interval 5
```

### 用已有 JSON 直接编码

```bash
python main.py encode \
  --video /data/v.mp4 \
  --json /data/output/v/analysis.json \
  --output /data/output/v/v_output.mp4 \
  --encoder h264_vaapi \
  --vaapi-device /dev/dri/renderD128
```

---

## 输出格式

每个视频的输出目录结构：

```
output/
└── {video_name}/
    ├── frames/                  # 抽取的关键帧（JPEG）
    │   ├── keyframe_000001_000000000.jpg
    │   └── ...
    ├── analysis.json            # VLM 分析结果 + video_clip_json
    └── {video_name}_output.mp4  # 编码后视频
```

`analysis.json` 中的 `video_clip_json` 字段格式：

```json
[
  {
    "_id": 1,
    "timestamp": "00:00:00,000-00:00:40,000",
    "picture": "VLM 对该片段的整体描述",
    "narration": "",
    "OST": 2
  }
]
```

| 字段 | 说明 |
|------|------|
| `timestamp` | `开始-结束`，格式 `HH:MM:SS,mmm` |
| `picture` | VLM 视觉描述 |
| `narration` | 旁白文案（LLM 可选填充） |
| `OST` | 2=保留原声，0=静音（配 TTS），1=原声 |

---

## 参数说明

| 参数 | 子命令 | 默认值 | 说明 |
|------|--------|--------|------|
| `--videos` | run/extract | — | 输入视频（可多个） |
| `--output` | run/extract/encode | — | 输出目录或文件路径 |
| `--vlm-url` | run | — | VLM API base URL |
| `--vlm-model` | run | — | VLM 模型名称 |
| `--interval` | run/extract | 5.0 | 抽帧间隔（秒） |
| `--batch-size` | run | 8 | 每批帧数 |
| `--concurrency` | run | 2 | VLM 并发批次数 |
| `--encoder` | run/encode | **必填** | h264_vaapi / h264_nvenc / libx264 |
| `--vaapi-device` | run/encode | /dev/dri/renderD128 | Intel VAAPI 设备节点 |
| `--no-encode` | run | — | 只分析，不编码 |
| `--llm-url` | run | — | （可选）LLM API URL |
| `--llm-model` | run | — | （可选）LLM 模型名称 |
