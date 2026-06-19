from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import requests


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "video_index"
THUMBS_ROOT = OUTPUT_ROOT / "thumbnails"
EXPORTS_ROOT = OUTPUT_ROOT / "exports"
LOGS_ROOT = OUTPUT_ROOT / "logs"
CACHE_ROOT = OUTPUT_ROOT / "cache"

VL_API_URL = os.getenv("VL_API_URL", "http://10.165.0.67:8000/v1/chat/completions")
VL_MODEL = os.getenv("VL_MODEL", "Qwen3-VL-8B-Instruct")
MODEL_FRAME_MAX_SIDE = int(os.getenv("MODEL_FRAME_MAX_SIDE", "2048"))
THUMB_MAX_SIDE = int(os.getenv("THUMB_MAX_SIDE", "320"))
VL_CONCURRENCY = int(os.getenv("VL_CONCURRENCY", "12"))
FFMPEG_FRAME_TIMEOUT_SEC = float(os.getenv("FFMPEG_FRAME_TIMEOUT_SEC", "30"))
PUBLIC_PORT = int(os.getenv("PUBLIC_PORT", "8000"))
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
SKIP_SCAN_DIR_NAMES = {"work", "航拍集锦输出"}

# 默认配置：直接运行 `python video_indexer.py` 时使用。
# 默认跑完整单视频；10 秒保底抽帧 + 3 秒最小间隔适合约 1 小时视频。
# 后续要换视频或批量跑全部视频，可以继续用命令行参数覆盖这些默认值。
DEFAULT_ROOT = "/mnt/share/INTEL项目"
DEFAULT_VIDEO = ""
DEFAULT_LIMIT_SECONDS = 0.0
DEFAULT_FIXED_INTERVAL_SEC = 10.0
DEFAULT_MIN_GAP_SEC = 6.0
DEFAULT_FEISHU_BASE_TOKEN = "XXX"
DEFAULT_FEISHU_VIDEO_TABLE = "XXX"
DEFAULT_FEISHU_FRAME_TABLE = "XXX"


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    video_name: str
    category: str
    nas_path: str
    duration_sec: float
    duration: str
    resolution: str
    fps: float
    file_size_mb: float
    status: str
    extracted_frame_count: int
    success_frame_count: int
    failed_frame_count: int
    video_summary: str
    started_at: str
    completed_at: str
    error_message: str


@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    frame_number: int
    timestamp_sec: float
    reason: str
    diff_score: float


@dataclass(frozen=True)
class PreparedFrame:
    candidate: FrameCandidate
    model_data_url: str
    thumb_path: Path


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    video_id: str
    video_name: str
    video_category: str
    source_video_path: str
    timestamp_sec: float
    timestamp: str
    recommended_clip_start_sec: float
    recommended_clip_end_sec: float
    frame_index: int
    extraction_reason: str
    frame_path: str
    thumbnail_path: str
    frame_url: str
    caption: str
    scene_type: str
    shooting_method: str
    environment_elements: str
    has_people: str
    vehicles: str
    main_objects: str
    mood: str
    ocr_text: str
    search_tags: str
    remix_value: int
    model_name: str
    understanding_status: str
    error_message: str
    created_at: str


def ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, THUMBS_ROOT, EXPORTS_ROOT, LOGS_ROOT, CACHE_ROOT):
        path.mkdir(parents=True, exist_ok=True)


RUN_LOG_PATH = LOGS_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log_event(event: str, **fields: object) -> None:
    payload = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False)
    print(line, flush=True)
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def infer_public_base_url() -> str:
    configured = os.getenv("PUBLIC_BASE_URL")
    if configured:
        return configured.rstrip("/")

    host = "127.0.0.1"
    try:
        model_host = VL_API_URL.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((model_host, 8001))
            host = sock.getsockname()[0]
    except (OSError, IndexError):
        pass
    return f"http://{host}:{PUBLIC_PORT}"


PUBLIC_BASE_URL = infer_public_base_url()


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", name).strip("_")
    return cleaned[:120] or uuid.uuid4().hex


NAS_MOUNT_ALIASES = {
    "/mnt/share": "/Volumes/share",
}


def share_mount_alias_paths(video_path: Path) -> set[str]:
    # 历史 CSV 里已混用 /mnt/share 与 /Volumes/share；跳过 checkpoint 时要同时接受这些挂载别名，但不能改写原始 stable_video_id。
    raw_path = str(video_path)
    aliases = {raw_path}
    for alias_prefix, canonical_prefix in NAS_MOUNT_ALIASES.items():
        if raw_path == alias_prefix or raw_path.startswith(f"{alias_prefix}/"):
            aliases.add(f"{canonical_prefix}{raw_path[len(alias_prefix):]}")
        if raw_path == canonical_prefix or raw_path.startswith(f"{canonical_prefix}/"):
            aliases.add(f"{alias_prefix}{raw_path[len(canonical_prefix):]}")
    return aliases


def stable_video_id(video_path: Path) -> str:
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:12]
    return f"vid_{digest}"


def stable_frame_id(video_id: str, timestamp_sec: float) -> str:
    return f"{video_id}_t{int(timestamp_sec * 1000):010d}"


def local_cache_path_for(source_video_path: Path, video_id: str) -> Path:
    return CACHE_ROOT / f"{video_id}_{safe_name(source_video_path.name)}"


def copy_video_to_local_cache(source_video_path: Path, video_id: str) -> Path:
    if not source_video_path.exists():
        raise FileNotFoundError(f"视频文件不存在或 NAS 未挂载：{source_video_path}")
    if not os.access(source_video_path, os.R_OK):
        raise PermissionError(f"视频文件不可读，请检查 NAS 只读挂载权限：{source_video_path}")

    cached_path = local_cache_path_for(source_video_path, video_id)
    temp_path = cached_path.with_suffix(cached_path.suffix + f".tmp_{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()

    source_size = source_video_path.stat().st_size
    if cached_path.exists() and cached_path.stat().st_size == source_size:
        log_event(
            "cache_reuse",
            video_name=source_video_path.name,
            video_id=video_id,
            cached_path=str(cached_path),
            size_mb=round(source_size / 1024 / 1024, 2),
        )
        return cached_path

    started = time.time()
    log_event(
        "cache_copy_start",
        video_name=source_video_path.name,
        video_id=video_id,
        source_size_mb=round(source_size / 1024 / 1024, 2),
    )
    shutil.copyfile(source_video_path, temp_path)
    temp_path.replace(cached_path)
    log_event(
        "cache_copy_done",
        video_name=source_video_path.name,
        video_id=video_id,
        cached_path=str(cached_path),
        elapsed_sec=round(time.time() - started, 2),
    )
    return cached_path


def remove_local_cache(cached_path: Path, video_name: str, video_id: str) -> None:
    try:
        if cached_path.exists():
            cached_path.unlink()
            log_event("cache_removed", video_name=video_name, video_id=video_id, cached_path=str(cached_path))
    except OSError as exc:
        log_event("cache_remove_failed", video_name=video_name, video_id=video_id, cached_path=str(cached_path), error=str(exc))


def seconds_to_timestamp(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def scan_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [dir_name for dir_name in dirs if dir_name not in SKIP_SCAN_DIR_NAMES]
        for file_name in files:
            path = Path(current_root) / file_name
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
    return sorted(videos, key=lambda item: str(item))


def video_category(video_path: Path, root: Path) -> str:
    try:
        relative = video_path.relative_to(root)
    except ValueError:
        return "未分类"
    return relative.parts[0] if len(relative.parts) > 1 else "未分类"


def resize_keep_aspect(frame, max_side: int):
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame
    scale = max_side / longest_side
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def compare_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (5, 5), 0)


def mean_abs_diff(previous, current) -> float:
    return float(cv2.absdiff(previous, current).mean())


def dynamic_scene_threshold(diff_scores: list[float]) -> float:
    if not diff_scores:
        return 12.0
    sorted_scores = sorted(diff_scores)
    median = sorted_scores[len(sorted_scores) // 2]
    mean = sum(sorted_scores) / len(sorted_scores)
    variance = sum((score - mean) ** 2 for score in sorted_scores) / len(sorted_scores)
    std = math.sqrt(variance)
    return max(10.0, min(28.0, median + 1.8 * std))


def collect_frame_candidates(
    video_path: Path,
    fixed_interval_sec: float,
    min_gap_sec: float,
    limit_seconds: float | None,
    display_video_name: str | None = None,
) -> tuple[list[FrameCandidate], float, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    effective_duration = min(duration_sec, limit_seconds) if limit_seconds else duration_sec
    scan_interval_sec = max(min_gap_sec, fixed_interval_sec, 1.0)
    similar_frame_diff_threshold = 12.0
    max_similar_gap_sec = max(30.0, scan_interval_sec * 3)

    raw_candidates: list[tuple[int, float, str, float]] = []
    last_kept_small = None
    last_kept_sec = -max_similar_gap_sec
    skipped_black_count = 0
    skipped_similar_count = 0
    log_video_name = display_video_name or video_path.name

    log_event(
        "scan_start",
        video_name=log_video_name,
        duration_sec=round(duration_sec, 2),
        effective_duration_sec=round(effective_duration, 2),
        scan_interval_sec=round(scan_interval_sec, 2),
        estimated_seek_count=int(effective_duration // scan_interval_sec) + 1 if effective_duration else 0,
        fps=round(fps, 2),
    )

    current_sec = 0.0
    last_progress_bucket = -1
    while current_sec <= effective_duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)
        ok, frame = capture.read()
        if not ok:
            current_sec += scan_interval_sec
            continue
        frame_number = int(round(current_sec * fps))
        current_small = compare_frame(frame)
        black_frame, mean_brightness, brightness_std = is_black_frame(frame)
        if black_frame:
            skipped_black_count += 1
            log_event(
                "black_candidate_skipped",
                video_name=log_video_name,
                timestamp=seconds_to_timestamp(current_sec),
                timestamp_sec=round(current_sec, 2),
                mean_brightness=round(mean_brightness, 2),
                brightness_std=round(brightness_std, 2),
            )
            current_sec += scan_interval_sec
            continue

        if last_kept_small is None:
            raw_candidates.append((frame_number, 0.0, "首帧保底", current_sec))
        else:
            score = mean_abs_diff(last_kept_small, current_small)
            if score < similar_frame_diff_threshold and current_sec - last_kept_sec < max_similar_gap_sec:
                skipped_similar_count += 1
                current_sec += scan_interval_sec
                continue
            raw_candidates.append((frame_number, score, "固定间隔保底", current_sec))
        last_kept_small = current_small
        last_kept_sec = current_sec
        progress_bucket = int(current_sec // 300)
        if progress_bucket != last_progress_bucket and current_sec > 0:
            last_progress_bucket = progress_bucket
            log_event("scan_progress", video_name=log_video_name, current_sec=round(current_sec, 1), effective_duration_sec=round(effective_duration, 1))
        current_sec += scan_interval_sec

    capture.release()

    merged: dict[int, tuple[int, float, str, float]] = {}
    for candidate_frame, score, reason, timestamp_sec in raw_candidates:
        existing = merged.get(candidate_frame)
        if existing is None:
            merged[candidate_frame] = (candidate_frame, score, reason, timestamp_sec)
        else:
            reasons = sorted(set(existing[2].split("+") + [reason]))
            merged[candidate_frame] = (candidate_frame, max(existing[1], score), "+".join(reasons), timestamp_sec)

    candidates: list[FrameCandidate] = []
    last_timestamp_sec = -min_gap_sec
    for candidate_frame, score, reason, timestamp_sec in sorted(merged.values(), key=lambda item: item[3]):
        if timestamp_sec - last_timestamp_sec < min_gap_sec:
            continue
        candidates.append(
            FrameCandidate(
                frame_index=len(candidates),
                frame_number=candidate_frame,
                timestamp_sec=timestamp_sec,
                reason=reason,
                diff_score=round(score, 2),
            )
        )
        last_timestamp_sec = timestamp_sec

    log_event(
        "scan_done",
        video_name=log_video_name,
        candidates=len(candidates),
        scan_interval_sec=round(scan_interval_sec, 2),
        skipped_black=skipped_black_count,
        skipped_similar=skipped_similar_count,
        similar_frame_diff_threshold=similar_frame_diff_threshold,
    )

    return candidates, fps, duration_sec, total_frames


def is_black_frame(frame: Any) -> tuple[bool, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    brightness_std = float(gray.std())
    return mean_brightness < 8.0 and brightness_std < 5.0, mean_brightness, brightness_std


def read_frame_with_ffmpeg(video_path: Path, candidate: FrameCandidate, timeout_sec: float) -> Any:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{candidate.timestamp_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg 读取帧超时：{video_path} @ {candidate.timestamp_sec:.2f}s timeout={timeout_sec}s") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg 不存在，无法使用带超时的读帧方式") from exc
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg 读取帧失败：{video_path} @ {candidate.timestamp_sec:.2f}s {detail}")
    encoded = np_from_buffer(completed.stdout)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"ffmpeg 帧解码失败：{video_path} @ {candidate.timestamp_sec:.2f}s")
    return frame


def np_from_buffer(data: bytes) -> Any:
    import numpy as np

    return np.frombuffer(data, dtype=np.uint8)


def read_candidate_frame_with_retry(video_path: Path, candidate: FrameCandidate, retry_count: int = 1) -> Any:
    try:
        return read_frame_with_ffmpeg(video_path, candidate, FFMPEG_FRAME_TIMEOUT_SEC)
    except RuntimeError as exc:
        log_event(
            "frame_read_ffmpeg_failed",
            video_name=video_path.name,
            timestamp=seconds_to_timestamp(candidate.timestamp_sec),
            timestamp_sec=round(candidate.timestamp_sec, 2),
            frame_number=candidate.frame_number,
            timeout_sec=FFMPEG_FRAME_TIMEOUT_SEC,
            error=str(exc),
        )

    last_error = ""
    for attempt in range(1, retry_count + 1):
        capture = cv2.VideoCapture(str(video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_number)
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_MSEC, candidate.timestamp_sec * 1000)
                ok, frame = capture.read()
        finally:
            capture.release()
        if ok:
            return frame
        last_error = f"attempt={attempt}"
        log_event(
            "frame_read_retry",
            video_name=video_path.name,
            timestamp=seconds_to_timestamp(candidate.timestamp_sec),
            timestamp_sec=round(candidate.timestamp_sec, 2),
            frame_number=candidate.frame_number,
            attempt=attempt,
            retry_count=retry_count,
        )
    raise RuntimeError(f"无法读取帧：{video_path} @ {candidate.timestamp_sec:.2f}s ({last_error})")


def prepare_candidate_frame(video_path: Path, candidate: FrameCandidate, video_id: str) -> tuple[str, Path] | None:
    thumb_dir = THUMBS_ROOT / video_id
    thumb_dir.mkdir(parents=True, exist_ok=True)

    frame = read_candidate_frame_with_retry(video_path, candidate)

    black_frame, mean_brightness, brightness_std = is_black_frame(frame)
    if black_frame:
        log_event(
            "black_frame_skipped",
            video_name=video_path.name,
            timestamp=seconds_to_timestamp(candidate.timestamp_sec),
            timestamp_sec=round(candidate.timestamp_sec, 2),
            mean_brightness=round(mean_brightness, 2),
            brightness_std=round(brightness_std, 2),
        )
        return None

    file_stem = f"frame_{candidate.frame_index:06d}_{int(candidate.timestamp_sec * 1000):010d}"
    thumb_path = thumb_dir / f"{file_stem}.jpg"

    model_frame = resize_keep_aspect(frame, MODEL_FRAME_MAX_SIDE)
    thumb_frame = resize_keep_aspect(frame, THUMB_MAX_SIDE)
    ok, encoded = cv2.imencode(".jpg", model_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError(f"无法编码关键帧：{video_path} @ {candidate.timestamp_sec:.2f}s")
    model_data_url = f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"
    if not cv2.imwrite(str(thumb_path), thumb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]):
        raise RuntimeError(f"无法保存缩略图：{thumb_path}")
    return model_data_url, thumb_path


def output_url_for(path: Path) -> str:
    relative = path.relative_to(OUTPUT_ROOT).as_posix()
    return f"{PUBLIC_BASE_URL}/video_index/{relative}"


def call_chat_completion(messages: list[dict[str, Any]], max_tokens: int | None, temperature: float = 0) -> str:
    payload = {
        "model": VL_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    response = requests.post(
        VL_API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=180,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:1200]
        raise requests.HTTPError(f"{exc}; response body: {detail}", response=response) from exc
    payload = response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return content.strip() if isinstance(content, str) else ""


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def stringify_list(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def understand_frame(frame_url: str) -> tuple[dict[str, Any], str]:
    prompt = """
请分析这张视频关键帧，只输出 JSON，不要输出 Markdown。
请按真实可见内容填写字段，不要为了贴合示例而硬套分类；不确定就写更宽泛的词。
字段要求：
caption: 用一段中文客观描述画面，包含主要场景、主体、动作、空间关系和氛围；不要识别具体姓名，不要编造画面外信息；
scene_type: 用一个短语概括场景类型，可以自由填写，例如自然风光、公路行驶、车内视角、城市街景、人物讲话、车辆展示、幕后拍摄、营地生活、其他；
shooting_method: 用短语描述明显可判断的拍摄方式或镜头视角，例如航拍、高空俯拍、车内视角、车外跟拍、前挡风视角、手持拍摄、固定机位、近景特写、中景、远景、俯拍、仰拍、第一人称视角、运动相机视角、屏幕录制/地图画面；如果不明显或无法可靠判断，必须输出空字符串，不要输出“无法判断”；
environment_elements: 数组，列出画面中的环境元素，按实际可见内容自由填写；
has_people: 是/否；
vehicles: 数组，列出可见车辆类型、颜色或状态；
main_objects: 数组，列出画面主要对象；
mood: 数组，列出画面氛围或情绪词；
ocr_text: 可见文字，没有则空字符串；
search_tags: 数组，给出适合业务搜索的具体关键词，避免过泛标签；
remix_value: 1到5的整数，越适合混剪越高；
严格只描述可见信息，不要输出无法确认的人名、品牌或事件背景。
""".strip()
    raw = call_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": frame_url}},
                ],
            }
        ],
        max_tokens=512,
        temperature=0,
    )
    return extract_json_object(raw), raw


def format_summary_frame_lines(records: list[FrameRecord]) -> list[str]:
    frame_lines = []
    for record in records:
        caption = record.caption[:180]
        frame_lines.append(
            " | ".join(
                [
                    f"时间码：{record.timestamp}",
                    f"场景：{record.scene_type or '未标注'}",
                    f"拍摄方式：{record.shooting_method or '未标注'}",
                    f"描述：{caption}",
                    f"主体：{feishu_main_objects(record) or '未标注'}",
                    f"关键词：{record.search_tags or '未标注'}",
                    f"混剪价值：{record.remix_value}",
                ]
            )
        )
    return frame_lines


def summarize_frame_chunk(video_name: str, records: list[FrameRecord], chunk_index: int, chunk_count: int) -> str:
    frame_lines = format_summary_frame_lines(records)
    prompt = f"""
你是视频素材整理助手。请根据关键帧理解结果，为视频《{video_name}》的第 {chunk_index}/{chunk_count} 段生成中文分段总结。
要求：
1. 内容要丰富，不要过短；覆盖本段出现的主要场景、人物/车辆/物体、动作、环境、氛围和镜头变化；
2. 只根据关键帧信息总结，不要编造未出现的剧情、人物身份、品牌或地点；
3. 按时间顺序概括，保留对业务搜索有价值的场景、对象、动作、氛围词和混剪亮点；
4. 如果本段内容较复杂，可以写成 2-4 句，不要只写一句泛泛概括。

关键帧信息：
{chr(10).join(frame_lines)}
""".strip()
    return call_chat_completion(
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        max_tokens=None,
        temperature=0,
    )


def merge_video_summaries(video_name: str, chunk_summaries: list[str], merge_round: int = 1) -> str:
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    max_merge_items = 20
    if len(chunk_summaries) > max_merge_items:
        merged_batches: list[str] = []
        for start in range(0, len(chunk_summaries), max_merge_items):
            batch = chunk_summaries[start : start + max_merge_items]
            merged_batches.append(merge_video_summaries(video_name, batch, merge_round=merge_round + 1))
        log_event(
            "video_summary_merge_round",
            video_name=video_name,
            merge_round=merge_round,
            input_summaries=len(chunk_summaries),
            output_summaries=len(merged_batches),
            max_merge_items=max_merge_items,
        )
        return merge_video_summaries(video_name, merged_batches, merge_round=merge_round + 1)

    summary_lines = [f"第 {index + 1} 段：{summary[:500]}" for index, summary in enumerate(chunk_summaries)]

    prompt = f"""
你是视频素材整理助手。请根据分段总结，为视频《{video_name}》生成最终中文视频总结。
要求：
1. 内容要比普通一句话摘要更丰富，适合作为素材库里的视频级说明；
2. 按时间线概括主要内容脉络，覆盖重要场景变化、人物互动、车辆/道路/自然环境、美食/露营/运动等可见素材元素；
3. 单独点出适合检索和混剪的画面亮点，例如自然风光、车内对话、道路行驶、人物互动、营地生活、运动场景等；
4. 只根据分段总结，不要编造未出现的剧情、人物身份、品牌或地点；
5. 输出 2-4 段中文，第一段讲内容脉络，第二段讲素材亮点和使用价值；如果信息足够丰富，可增加第三段补充搜索关键词方向。

分段总结：
{chr(10).join(summary_lines)}
""".strip()
    return call_chat_completion(
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        max_tokens=None,
        temperature=0,
    )


def summarize_video(video_name: str, records: list[FrameRecord]) -> str:
    successful_records = [record for record in records if record.understanding_status == "成功" and record.caption]
    if not successful_records:
        return ""

    chunk_size = 30
    chunks = [successful_records[index : index + chunk_size] for index in range(0, len(successful_records), chunk_size)]
    if len(chunks) > 1:
        log_event(
            "video_summary_chunked",
            video_name=video_name,
            source_frames=len(successful_records),
            chunk_count=len(chunks),
            chunk_size=chunk_size,
        )

    chunk_summaries = [
        summarize_frame_chunk(video_name, chunk, index + 1, len(chunks))
        for index, chunk in enumerate(chunks)
    ]
    chunk_summaries = [summary for summary in chunk_summaries if summary.strip()]
    return merge_video_summaries(video_name, chunk_summaries) if chunk_summaries else ""


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_completed_video_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    for row in read_csv_rows(path):
        if row.get("video_id") and row.get("status") == "已完成":
            completed.add(row["video_id"])
            nas_path = row.get("nas_path", "")
            if nas_path:
                for alias_path in share_mount_alias_paths(Path(nas_path)):
                    completed.add(stable_video_id(Path(alias_path)))
    return completed


def load_successful_frame_rows(path: Path, video_id: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        if row.get("video_id") != video_id:
            continue
        frame_id = row.get("frame_id", "")
        if frame_id and row.get("understanding_status") == "成功":
            rows[frame_id] = row
    return rows


def coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def frame_record_from_row(row: dict[str, str]) -> FrameRecord:
    values: dict[str, Any] = {}
    for field_name in FRAME_FIELDNAMES:
        value = row.get(field_name, "")
        if field_name in {"timestamp_sec", "recommended_clip_start_sec", "recommended_clip_end_sec"}:
            values[field_name] = coerce_float(value)
        elif field_name in {"frame_index", "remix_value"}:
            values[field_name] = coerce_int(value)
        else:
            values[field_name] = value
    return FrameRecord(**values)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])
        if existing_header != fieldnames:
            existing_rows = read_csv_rows(path)
            write_csv(path, existing_rows, fieldnames)
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def upsert_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str], key_field: str) -> None:
    existing_rows = read_csv_rows(path)
    output_fieldnames = list(fieldnames)
    for existing_row in existing_rows:
        for field_name in existing_row:
            if field_name not in output_fieldnames:
                output_fieldnames.append(field_name)
    key = str(row.get(key_field, ""))
    replaced = False
    output_rows: list[dict[str, Any]] = []
    for existing_row in existing_rows:
        if existing_row.get(key_field) == key:
            output_rows.append({**existing_row, **row})
            replaced = True
        else:
            output_rows.append(existing_row)
    if not replaced:
        output_rows.append(row)
    write_csv(path, output_rows, output_fieldnames)


def feishu_keywords(record: FrameRecord) -> list[str]:
    text = " ".join(
        [
            record.caption,
            record.scene_type,
            record.environment_elements,
            record.vehicles,
            record.main_objects,
            record.search_tags,
        ]
    )
    keyword_rules = [
        ("风景", ["风景", "自然", "沙漠", "山", "河", "雪", "岩石", "航拍", "地貌", "山谷", "瀑布"]),
        ("车辆", ["汽车", "车辆", "车门"]),
        ("人物", ["人物", "男子", "工作人员", "讲话", "自拍", "挥手"]),
        ("道路", ["道路", "公路", "城市道路"]),
    ]
    keywords: list[str] = []
    for keyword, words in keyword_rules:
        if any(word in text for word in words):
            keywords.append(keyword)
    unique_keywords: list[str] = []
    for keyword in keywords:
        if keyword not in unique_keywords:
            unique_keywords.append(keyword)
    return unique_keywords


def feishu_remix_value(record: FrameRecord) -> str:
    try:
        remix_value = int(record.remix_value)
    except (TypeError, ValueError):
        remix_value = 0
    if remix_value >= 5:
        return "高"
    if remix_value >= 3:
        return "中"
    return "低"


def feishu_main_objects(record: FrameRecord) -> str:
    return record.main_objects or record.vehicles or record.environment_elements


def is_lark_rate_limited(result: dict[str, Any], output: str) -> bool:
    error = result.get("error", {})
    code = error.get("code") if isinstance(error, dict) else result.get("code")
    message = json.dumps(result, ensure_ascii=False) + output
    return code in {800004135, 1254290, 99991400} or "OpenAPISearchRecord limited" in message or "TooManyRequest" in message


def parse_lark_cli_json(stdout: str) -> dict[str, Any] | None:
    json_start = stdout.find("{")
    if json_start < 0:
        return None
    try:
        result = json.loads(stdout[json_start:])
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def lark_cli(args: list[str], cwd: Path, dry_run: bool) -> dict[str, Any]:
    command = ["lark-cli", *args]
    if dry_run:
        log_event("dry_run_lark_cli", command=command, cwd=str(cwd))
        return {"ok": True, "data": {"record_id_list": []}}
    retry_delays = [8.0, 16.0, 32.0]
    for attempt in range(len(retry_delays) + 1):
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
        if completed.stdout:
            print(completed.stdout.strip())
        if completed.stderr:
            print(completed.stderr.strip())
        result = parse_lark_cli_json(completed.stdout)
        if completed.returncode == 0:
            if result is None:
                raise RuntimeError(f"lark-cli 返回非 JSON：{completed.stdout[:500]}")
            if result.get("ok"):
                return result
            if is_lark_rate_limited(result, completed.stdout + completed.stderr) and attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
                continue
            raise RuntimeError(f"lark-cli 返回失败：{json.dumps(result, ensure_ascii=False)[:1200]}")
        if result is not None and is_lark_rate_limited(result, completed.stdout + completed.stderr) and attempt < len(retry_delays):
            time.sleep(retry_delays[attempt])
            continue
        raise RuntimeError(f"lark-cli 执行失败：{' '.join(command)}")
    raise RuntimeError(f"lark-cli 执行失败：{' '.join(command)}")


def write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def batch_create_feishu_records(
    base_token: str,
    table_id: str,
    payload_path: Path,
    identity: str,
    dry_run: bool,
) -> list[str]:
    result = lark_cli(
        [
            "base",
            "+record-batch-create",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--json",
            f"@{payload_path.name}",
            "--as",
            identity,
        ],
        cwd=payload_path.parent,
        dry_run=dry_run,
    )
    record_ids = result.get("data", {}).get("record_id_list", [])
    return [str(record_id) for record_id in record_ids]


def cell_value_matches(value: object, expected: str) -> bool:
    if isinstance(value, str):
        return value == expected
    if isinstance(value, (int, float, bool)):
        return str(value) == expected
    if isinstance(value, dict):
        text = value.get("text") or value.get("value") or value.get("name")
        if text is not None and str(text) == expected:
            return True
        return any(cell_value_matches(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(cell_value_matches(item, expected) for item in value)
    return False


def iter_feishu_records(payload: object) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        if "record_id" in payload and isinstance(payload.get("fields"), dict):
            yield payload
        for value in payload.values():
            yield from iter_feishu_records(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from iter_feishu_records(item)


def iter_feishu_matrix_rows(result: dict[str, Any]) -> Iterable[tuple[dict[str, Any], str | None]]:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return
    fields = data.get("fields", [])
    rows = data.get("data", [])
    record_ids = data.get("record_id_list", [])
    if not isinstance(fields, list) or not isinstance(rows, list):
        return
    for index, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        row_fields = {str(field): row[field_index] for field_index, field in enumerate(fields) if field_index < len(row)}
        record_id = record_ids[index] if isinstance(record_ids, list) and index < len(record_ids) else None
        yield row_fields, str(record_id) if record_id is not None else None


def search_feishu_record_ids_by_field(
    base_token: str,
    table_id: str,
    identity: str,
    field_name: str,
    field_value: str,
    dry_run: bool,
) -> list[str]:
    if dry_run:
        return []
    limit = 200
    result = lark_cli(
        [
            "base",
            "+record-search",
            "--base-token",
            base_token,
            "--table-id",
            table_id,
            "--json",
            json.dumps(
                {
                    "keyword": field_value,
                    "search_fields": [field_name],
                    "select_fields": [field_name],
                    "offset": 0,
                    "limit": limit,
                },
                ensure_ascii=False,
            ),
            "--as",
            identity,
        ],
        cwd=BASE_DIR,
        dry_run=False,
    )
    record_ids: list[str] = []
    for fields, record_id in iter_feishu_matrix_rows(result):
        if record_id is not None and cell_value_matches(fields.get(field_name), field_value):
            record_ids.append(record_id)
    for record in iter_feishu_records(result):
        fields = record.get("fields", {})
        if not isinstance(fields, dict):
            continue
        if cell_value_matches(fields.get(field_name), field_value):
            record_id = record.get("record_id")
            if record_id is not None:
                record_ids.append(str(record_id))
    return record_ids


def field_text_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (int, float, bool)):
        yield str(value)
    elif isinstance(value, dict):
        text = value.get("text") or value.get("value") or value.get("name")
        if text is not None:
            yield str(text)
        for item in value.values():
            yield from field_text_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from field_text_values(item)


def search_feishu_field_values_by_field(
    base_token: str,
    table_id: str,
    identity: str,
    search_field_name: str,
    search_field_value: str,
    return_field_name: str,
    dry_run: bool,
) -> list[str]:
    if dry_run:
        return []
    limit = 200
    values: list[str] = []
    offset = 0
    while True:
        result = lark_cli(
            [
                "base",
                "+record-search",
                "--base-token",
                base_token,
                "--table-id",
                table_id,
                "--json",
                json.dumps(
                    {
                        "keyword": search_field_value,
                        "search_fields": [search_field_name],
                        "select_fields": [search_field_name, return_field_name],
                        "offset": offset,
                        "limit": limit,
                    },
                    ensure_ascii=False,
                ),
                "--as",
                identity,
            ],
            cwd=BASE_DIR,
            dry_run=False,
        )
        for fields, _record_id in iter_feishu_matrix_rows(result):
            if cell_value_matches(fields.get(search_field_name), search_field_value):
                values.extend(field_text_values(fields.get(return_field_name)))
        for record in iter_feishu_records(result):
            fields = record.get("fields", {})
            if not isinstance(fields, dict):
                continue
            if not cell_value_matches(fields.get(search_field_name), search_field_value):
                continue
            values.extend(field_text_values(fields.get(return_field_name)))
        data = result.get("data", {})
        has_more = isinstance(data, dict) and data.get("has_more") is True
        if not has_more:
            break
        offset += limit
    return values


def search_feishu_record_ids_and_field_values_by_field(
    base_token: str,
    table_id: str,
    identity: str,
    search_field_name: str,
    search_field_value: str,
    return_field_name: str,
    dry_run: bool,
) -> dict[str, str]:
    if dry_run:
        return {}
    limit = 200
    values: dict[str, str] = {}
    offset = 0
    while True:
        result = lark_cli(
            [
                "base",
                "+record-search",
                "--base-token",
                base_token,
                "--table-id",
                table_id,
                "--json",
                json.dumps(
                    {
                        "keyword": search_field_value,
                        "search_fields": [search_field_name],
                        "select_fields": [search_field_name, return_field_name],
                        "offset": offset,
                        "limit": limit,
                    },
                    ensure_ascii=False,
                ),
                "--as",
                identity,
            ],
            cwd=BASE_DIR,
            dry_run=False,
        )
        for fields, record_id in iter_feishu_matrix_rows(result):
            if record_id is not None and cell_value_matches(fields.get(search_field_name), search_field_value):
                for value in field_text_values(fields.get(return_field_name)):
                    values[str(value)] = record_id
        for record in iter_feishu_records(result):
            fields = record.get("fields", {})
            record_id = record.get("record_id")
            if record_id is None or not isinstance(fields, dict):
                continue
            if not cell_value_matches(fields.get(search_field_name), search_field_value):
                continue
            for value in field_text_values(fields.get(return_field_name)):
                values[str(value)] = str(record_id)
        data = result.get("data", {})
        has_more = isinstance(data, dict) and data.get("has_more") is True
        if not has_more:
            break
        offset += limit
    return values


def upload_feishu_attachments_for_existing_records(
    base_token: str,
    frame_table: str,
    identity: str,
    video_id: str,
    frame_records: list[FrameRecord],
    dry_run: bool,
) -> None:
    frame_record_ids = search_feishu_record_ids_and_field_values_by_field(
        base_token=base_token,
        table_id=frame_table,
        identity=identity,
        search_field_name="视频ID",
        search_field_value=video_id,
        return_field_name="素材ID",
        dry_run=dry_run,
    )
    for frame_record in frame_records:
        record_id = frame_record_ids.get(frame_record.frame_id)
        if not record_id:
            continue
        thumb_path = Path(frame_record.thumbnail_path)
        if not thumb_path.exists():
            log_event("missing_thumbnail", thumbnail_path=frame_record.thumbnail_path, frame_id=frame_record.frame_id)
            continue
        try:
            cli_thumb_path = thumb_path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            cli_thumb_path = str(thumb_path)
        lark_cli(
            [
                "base",
                "+record-upload-attachment",
                "--base-token",
                base_token,
                "--table-id",
                frame_table,
                "--record-id",
                record_id,
                "--field-id",
                "缩略图",
                "--file",
                cli_thumb_path,
                "--name",
                thumb_path.name,
                "--as",
                identity,
            ],
            cwd=BASE_DIR,
            dry_run=dry_run,
        )
        time.sleep(1)


def write_feishu_records(
    base_token: str,
    video_table: str,
    frame_table: str,
    identity: str,
    video_infos: list[VideoInfo],
    frame_records: list[FrameRecord],
    dry_run: bool,
    upload_attachments: bool,
    create_video_records: bool = True,
) -> None:
    payload_dir = EXPORTS_ROOT / "feishu_payloads"
    video_payload_path = payload_dir / "video_records.json"

    video_fields = [
        "视频ID",
        "视频名称",
        "NAS路径",
        "目录分类",
        "时长秒",
        "分辨率",
        "帧率",
        "处理状态",
        "关键帧数",
        "成功帧数",
        "失败帧数",
        "视频总结",
        "处理耗时秒",
    ]
    video_rows = [
        [
            info.video_id,
            info.video_name,
            info.nas_path,
            info.category,
            info.duration_sec,
            info.resolution,
            info.fps,
            info.status,
            info.extracted_frame_count,
            info.success_frame_count,
            info.failed_frame_count,
            info.video_summary,
            None,
        ]
        for info in video_infos
    ]
    if create_video_records:
        write_json_payload(video_payload_path, {"fields": video_fields, "rows": video_rows})

    frame_fields = [
        "素材ID",
        "视频ID",
        "视频名称",
        "时间码",
        # 缩略图是附件字段，必须在记录创建后用 +record-upload-attachment 单独上传；
        # 因此不放入 batch-create payload，但业务展示顺序应放在「画面描述」前。
        "拍摄方式",
        "画面描述",
        "场景类型",
        "主体对象",
        "关键词",
        "混剪价值",
        "分析状态",
    ]
    frame_rows = [
        [
            record.frame_id,
            record.video_id,
            record.video_name,
            record.timestamp,
            record.shooting_method,
            record.caption,
            record.scene_type,
            feishu_main_objects(record),
            feishu_keywords(record),
            feishu_remix_value(record),
            record.understanding_status,
        ]
        for record in frame_records
    ]
    if create_video_records:
        batch_create_feishu_records(base_token, video_table, video_payload_path, identity, dry_run)
    frame_record_ids: list[str] = []
    for batch_index, frame_row_batch in enumerate(chunks(frame_rows, 200), start=1):
        frame_payload_path = payload_dir / f"frame_records_{batch_index:03d}.json"
        write_json_payload(frame_payload_path, {"fields": frame_fields, "rows": frame_row_batch})
        frame_record_ids.extend(batch_create_feishu_records(base_token, frame_table, frame_payload_path, identity, dry_run))
        time.sleep(0.8)

    if upload_attachments:
        for record_id, frame_record in zip(frame_record_ids, frame_records):
            thumb_path = Path(frame_record.thumbnail_path)
            if not thumb_path.exists():
                log_event("missing_thumbnail", thumbnail_path=frame_record.thumbnail_path, frame_id=frame_record.frame_id)
                continue
            try:
                cli_thumb_path = thumb_path.relative_to(BASE_DIR).as_posix()
            except ValueError:
                cli_thumb_path = str(thumb_path)
            lark_cli(
                [
                    "base",
                    "+record-upload-attachment",
                    "--base-token",
                    base_token,
                    "--table-id",
                    frame_table,
                    "--record-id",
                    record_id,
                    "--field-id",
                    "缩略图",
                    "--file",
                    cli_thumb_path,
                    "--name",
                    thumb_path.name,
                    "--as",
                    identity,
                ],
                cwd=BASE_DIR,
                dry_run=dry_run,
            )
            time.sleep(1)


def write_video_to_feishu_if_ready(
    base_token: str,
    video_table: str,
    frame_table: str,
    identity: str,
    video_info: VideoInfo,
    records: list[FrameRecord],
    dry_run: bool,
    upload_attachments: bool,
) -> bool:
    if video_info.status != "已完成" or video_info.success_frame_count <= 0:
        log_event(
            "skip_feishu_write",
            video_name=video_info.video_name,
            video_id=video_info.video_id,
            reason="video_not_successful",
            status=video_info.status,
            success_frame_count=video_info.success_frame_count,
        )
        return False

    successful_records = [record for record in records if record.understanding_status == "成功"]
    if not successful_records:
        log_event(
            "skip_feishu_write",
            video_name=video_info.video_name,
            video_id=video_info.video_id,
            reason="no_successful_frames",
        )
        return False

    existing_video_record_ids = search_feishu_record_ids_by_field(
        base_token=base_token,
        table_id=video_table,
        identity=identity,
        field_name="视频ID",
        field_value=video_info.video_id,
        dry_run=dry_run,
    )
    create_video_record = not existing_video_record_ids

    existing_frame_ids = set(
        search_feishu_field_values_by_field(
            base_token=base_token,
            table_id=frame_table,
            identity=identity,
            search_field_name="视频ID",
            search_field_value=video_info.video_id,
            return_field_name="素材ID",
            dry_run=dry_run,
        )
    )
    records_to_create = [record for record in successful_records if record.frame_id not in existing_frame_ids]
    skipped_existing_frames = len(successful_records) - len(records_to_create)

    if not create_video_record and not records_to_create:
        log_event(
            "skip_feishu_write",
            video_name=video_info.video_name,
            video_id=video_info.video_id,
            reason="already_exists",
            existing_video_record_ids=existing_video_record_ids,
            existing_frame_count=skipped_existing_frames,
        )
        return False

    write_feishu_records(
        base_token=base_token,
        video_table=video_table,
        frame_table=frame_table,
        identity=identity,
        video_infos=[video_info] if create_video_record else [],
        frame_records=records_to_create,
        dry_run=dry_run,
        upload_attachments=upload_attachments,
        create_video_records=create_video_record,
    )
    log_event(
        "feishu_write_done",
        video_name=video_info.video_name,
        video_id=video_info.video_id,
        frame_count=len(records_to_create),
        skipped_existing_frames=skipped_existing_frames,
        skipped_existing_video=not create_video_record,
        existing_video_record_ids=existing_video_record_ids,
        feishu_base_token=base_token,
        feishu_video_table=video_table,
        feishu_frame_table=frame_table,
        feishu_attachments=upload_attachments,
        feishu_dry_run=dry_run,
    )
    return True


def wait_for_feishu_uploads(futures: list[Future[bool]]) -> None:
    if not futures:
        return
    log_event("feishu_upload_wait_start", pending_uploads=len(futures))
    success_count = 0
    failed_count = 0
    skipped_count = 0
    for future in futures:
        try:
            wrote = future.result()
        except Exception as exc:
            failed_count += 1
            log_event("feishu_upload_error", error=str(exc))
            continue
        if wrote:
            success_count += 1
        else:
            skipped_count += 1
    log_event(
        "feishu_upload_wait_done",
        total_uploads=len(futures),
        success_uploads=success_count,
        skipped_uploads=skipped_count,
        failed_uploads=failed_count,
    )


def build_frame_record(
    video_id: str,
    video_path: Path,
    category: str,
    duration_sec: float,
    candidate: FrameCandidate,
    thumb_path: Path | str,
    data: dict[str, Any],
    status: str,
    error: str,
) -> FrameRecord:
    timestamp_sec = round(candidate.timestamp_sec, 2)
    clip_start = round(max(0.0, timestamp_sec - 1.0), 2)
    clip_end = round(min(duration_sec, timestamp_sec + 2.0), 2) if duration_sec else round(timestamp_sec + 2.0, 2)
    remix_value_raw = data.get("remix_value", 0)
    try:
        remix_value = int(remix_value_raw)
    except (TypeError, ValueError):
        remix_value = 0
    return FrameRecord(
        frame_id=stable_frame_id(video_id, timestamp_sec),
        video_id=video_id,
        video_name=video_path.name,
        video_category=category,
        source_video_path=str(video_path),
        timestamp_sec=timestamp_sec,
        timestamp=seconds_to_timestamp(timestamp_sec),
        recommended_clip_start_sec=clip_start,
        recommended_clip_end_sec=clip_end,
        frame_index=candidate.frame_index,
        extraction_reason=candidate.reason,
        frame_path="",
        thumbnail_path=str(thumb_path),
        frame_url="",
        caption=str(data.get("caption", "")).strip(),
        scene_type=str(data.get("scene_type", "")).strip(),
        shooting_method=str(data.get("shooting_method", "")).strip(),
        environment_elements=stringify_list(data.get("environment_elements")),
        has_people=str(data.get("has_people", "")).strip(),
        vehicles=stringify_list(data.get("vehicles")),
        main_objects=stringify_list(data.get("main_objects")),
        mood=stringify_list(data.get("mood")),
        ocr_text=str(data.get("ocr_text", "")).strip(),
        search_tags=stringify_list(data.get("search_tags")),
        remix_value=remix_value,
        model_name=VL_MODEL,
        understanding_status=status,
        error_message=error,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def analyze_candidate_frame(
    video_id: str,
    working_video_path: Path,
    source_video_path: Path,
    category: str,
    duration_sec: float,
    candidate: FrameCandidate,
) -> FrameRecord | None:
    try:
        prepared = prepare_candidate_frame(working_video_path, candidate, video_id)
        if prepared is None:
            return None
        model_data_url, thumb_path = prepared
        return analyze_prepared_frame(
            video_id=video_id,
            source_video_path=source_video_path,
            category=category,
            duration_sec=duration_sec,
            prepared=PreparedFrame(candidate=candidate, model_data_url=model_data_url, thumb_path=thumb_path),
        )
    except Exception as exc:
        return build_frame_record(
            video_id=video_id,
            video_path=source_video_path,
            category=category,
            duration_sec=duration_sec,
            candidate=candidate,
            thumb_path="",
            data={"caption": ""},
            status="失败",
            error=str(exc),
        )


def analyze_prepared_frame(
    video_id: str,
    source_video_path: Path,
    category: str,
    duration_sec: float,
    prepared: PreparedFrame,
) -> FrameRecord:
    try:
        candidate = prepared.candidate
        model_data_url = prepared.model_data_url
        thumb_path = prepared.thumb_path
        data, raw = understand_frame(model_data_url)
        if not data:
            data = {"caption": raw[:500]}
        return build_frame_record(
            video_id=video_id,
            video_path=source_video_path,
            category=category,
            duration_sec=duration_sec,
            candidate=candidate,
            thumb_path=thumb_path,
            data=data,
            status="成功",
            error="",
        )
    except Exception as exc:
        return build_frame_record(
            video_id=video_id,
            video_path=source_video_path,
            category=category,
            duration_sec=duration_sec,
            candidate=prepared.candidate,
            thumb_path=prepared.thumb_path,
            data={"caption": ""},
            status="失败",
            error=str(exc),
        )


FRAME_FIELDNAMES = list(FrameRecord.__dataclass_fields__.keys())
VIDEO_FIELDNAMES = list(VideoInfo.__dataclass_fields__.keys())


def index_video(
    source_video_path: Path,
    working_video_path: Path,
    root: Path,
    fixed_interval_sec: float,
    min_gap_sec: float,
    limit_seconds: float | None,
    frames_csv: Path,
    resume: bool,
    vl_concurrency: int,
) -> tuple[VideoInfo, list[FrameRecord]]:
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_id = stable_video_id(source_video_path)
    category = video_category(source_video_path, root)
    records: list[FrameRecord] = []
    success_count = 0
    failed_count = 0
    error_message = ""
    status = "已完成"
    duration_sec = 0.0
    fps = 0.0
    resolution = ""
    video_summary = ""
    existing_frame_rows = load_successful_frame_rows(frames_csv, video_id) if resume else {}

    try:
        if not working_video_path.exists():
            raise FileNotFoundError(f"本地缓存视频不存在：{working_video_path}")
        if not os.access(working_video_path, os.R_OK):
            raise PermissionError(f"本地缓存视频不可读：{working_video_path}")
        candidates, fps, duration_sec, _ = collect_frame_candidates(
            video_path=working_video_path,
            fixed_interval_sec=fixed_interval_sec,
            min_gap_sec=min_gap_sec,
            limit_seconds=limit_seconds,
            display_video_name=source_video_path.name,
        )
        capture = cv2.VideoCapture(str(working_video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        resolution = f"{width}x{height}" if width and height else ""

        pending_candidates: list[FrameCandidate] = []
        for candidate in candidates:
            frame_id = stable_frame_id(video_id, round(candidate.timestamp_sec, 2))
            existing_row = existing_frame_rows.get(frame_id)
            if existing_row:
                record = frame_record_from_row(existing_row)
                success_count += 1
                records.append(record)
                log_event("skip_frame", frame_id=frame_id, reason="checkpoint_success")
                continue
            pending_candidates.append(candidate)

        prepared_frames: list[PreparedFrame] = []
        for candidate in pending_candidates:
            try:
                prepared = prepare_candidate_frame(working_video_path, candidate, video_id)
                if prepared is None:
                    continue
                model_data_url, thumb_path = prepared
                prepared_frames.append(PreparedFrame(candidate=candidate, model_data_url=model_data_url, thumb_path=thumb_path))
            except Exception as exc:
                record = build_frame_record(
                    video_id=video_id,
                    video_path=source_video_path,
                    category=category,
                    duration_sec=duration_sec,
                    candidate=candidate,
                    thumb_path="",
                    data={"caption": ""},
                    status="失败",
                    error=str(exc),
                )
                failed_count += 1
                records.append(record)
                append_csv(frames_csv, asdict(record), FRAME_FIELDNAMES)
                log_event("frame_done", frame_id=record.frame_id, timestamp=record.timestamp, status=record.understanding_status)

        worker_count = max(1, min(vl_concurrency, len(prepared_frames) or 1))
        if pending_candidates:
            log_event(
                "pending_frames",
                pending_frames=len(pending_candidates),
                prepared_frames=len(prepared_frames),
                vl_concurrency=worker_count,
            )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_frame = {
                executor.submit(
                    analyze_prepared_frame,
                    video_id,
                    source_video_path,
                    category,
                    duration_sec,
                    prepared_frame,
                ): prepared_frame
                for prepared_frame in prepared_frames
            }
            for future in as_completed(future_to_frame):
                record = future.result()
                if record.understanding_status == "成功":
                    success_count += 1
                else:
                    failed_count += 1
                records.append(record)
                append_csv(frames_csv, asdict(record), FRAME_FIELDNAMES)
                log_event("frame_done", frame_id=record.frame_id, timestamp=record.timestamp, status=record.understanding_status)
        records.sort(key=lambda item: item.timestamp_sec)
        if records and success_count > 0:
            try:
                video_summary = summarize_video(source_video_path.name, records)
            except Exception as exc:
                summary_error = str(exc)
                error_message = f"视频总结失败：{summary_error}"
                log_event("video_summary_error", video_name=source_video_path.name, video_id=video_id, error=summary_error)
    except Exception as exc:
        status = "失败"
        error_message = str(exc)
        log_event("video_error", video_name=source_video_path.name, video_id=video_id, error=error_message)

    completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_info = VideoInfo(
        video_id=video_id,
        video_name=source_video_path.name,
        category=category,
        nas_path=str(source_video_path),
        duration_sec=round(duration_sec, 2),
        duration=seconds_to_timestamp(duration_sec),
        resolution=resolution,
        fps=round(fps, 2),
        file_size_mb=round(source_video_path.stat().st_size / 1024 / 1024, 2) if source_video_path.exists() else 0.0,
        status=status,
        extracted_frame_count=len(records),
        success_frame_count=success_count,
        failed_frame_count=failed_count,
        video_summary=video_summary,
        started_at=started_at,
        completed_at=completed_at,
        error_message=error_message,
    )
    return video_info, records

# 由于默认 --video 写死为单个测试视频，跑全部时要把 --video "" 覆盖为空，并把 --limit-videos 0 表示不限制数量。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1 视频关键词素材库索引器：分析视频并写入本地 CSV；飞书上传请使用 upload_analyzed_to_feishu.py。")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="只读视频根目录；批量扫描时使用")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="指定单个视频路径；改为空字符串才会扫描 root")
    parser.add_argument("--limit-videos", type=int, default=0, help="测试时限制处理视频数量；0 表示不限制")
    parser.add_argument("--limit-seconds", type=float, default=DEFAULT_LIMIT_SECONDS, help="只处理每个视频前 N 秒；0 表示完整视频")
    parser.add_argument("--fixed-interval-sec", type=float, default=DEFAULT_FIXED_INTERVAL_SEC, help="固定保底抽帧间隔秒；值越小抽帧越密")
    parser.add_argument("--min-gap-sec", type=float, default=DEFAULT_MIN_GAP_SEC, help="候选帧最小间隔秒；避免相邻重复画面")
    parser.add_argument("--vl-concurrency", type=int, default=VL_CONCURRENCY, help="并发调用视觉模型的 worker 数，默认 3")
    parser.add_argument("--frames-csv", default=str(EXPORTS_ROOT / "关键帧素材库.csv"), help="关键帧表 CSV 输出路径")
    parser.add_argument("--videos-csv", default=str(EXPORTS_ROOT / "视频清单.csv"), help="视频清单 CSV 输出路径")
    parser.add_argument("--force", action="store_true", help="强制重跑，忽略已有 CSV checkpoint")
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    vl_concurrency = max(1, args.vl_concurrency)
    root = Path(args.root)
    limit_seconds = args.limit_seconds if args.limit_seconds > 0 else None
    if args.video:
        videos = [Path(args.video)]
    else:
        videos = scan_videos(root)
    if args.limit_videos > 0:
        videos = videos[: args.limit_videos]

    videos_csv = Path(args.videos_csv)
    frames_csv = Path(args.frames_csv)
    resume = not args.force
    completed_video_ids = load_completed_video_ids(videos_csv) if resume else set()
    log_event(
        "run_start",
        root=str(root),
        video_count=len(videos),
        limit_seconds=limit_seconds,
        fixed_interval_sec=args.fixed_interval_sec,
        min_gap_sec=args.min_gap_sec,
        model_frame_max_side=MODEL_FRAME_MAX_SIDE,
        vl_concurrency=vl_concurrency,
        public_base_url=PUBLIC_BASE_URL,
        log_path=str(RUN_LOG_PATH),
    )
    if not videos:
        root_exists = root.exists()
        log_event(
            "no_videos_found",
            root=str(root),
            root_exists=root_exists,
            root_readable=os.access(root, os.R_OK) if root_exists else False,
            video_extensions=sorted(VIDEO_EXTENSIONS),
            hint="DEFAULT_VIDEO 为空时会扫描 root；请确认 NAS 已挂载、root 路径正确且目录下有支持格式视频。",
        )
        return

    for video_path in videos:
        started = time.time()
        video_id = stable_video_id(video_path)
        if resume and video_id in completed_video_ids:
            log_event("skip_video", video_name=video_path.name, video_id=video_id, reason="checkpoint_completed")
            continue
        log_event("video_start", video_name=video_path.name, video_id=video_id)
        cached_video_path: Path | None = None
        try:
            cached_video_path = copy_video_to_local_cache(video_path, video_id)
            video_info, records = index_video(
                source_video_path=video_path,
                working_video_path=cached_video_path,
                root=root,
                fixed_interval_sec=args.fixed_interval_sec,
                min_gap_sec=args.min_gap_sec,
                limit_seconds=limit_seconds,
                frames_csv=frames_csv,
                resume=resume,
                vl_concurrency=vl_concurrency,
            )
        except Exception as exc:
            log_event("video_error", video_name=video_path.name, video_id=video_id, error=str(exc))
            completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            video_info = VideoInfo(
                video_id=video_id,
                video_name=video_path.name,
                category=video_category(video_path, root),
                nas_path=str(video_path),
                duration_sec=0.0,
                duration="00:00:00",
                resolution="",
                fps=0.0,
                file_size_mb=round(video_path.stat().st_size / 1024 / 1024, 2) if video_path.exists() else 0.0,
                status="失败",
                extracted_frame_count=0,
                success_frame_count=0,
                failed_frame_count=0,
                video_summary="",
                started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                completed_at=completed_at,
                error_message=str(exc),
            )
            records = []
        finally:
            if cached_video_path is not None:
                remove_local_cache(cached_video_path, video_path.name, video_id)
        log_event(
            "video_done",
            video=video_path.name,
            status=video_info.status,
            frames=len(records),
            success=video_info.success_frame_count,
            failed=video_info.failed_frame_count,
            elapsed_sec=round(time.time() - started, 2),
        )
        upsert_csv_row(videos_csv, asdict(video_info), VIDEO_FIELDNAMES, "video_id")

    log_event("csv_written", videos_csv=args.videos_csv, frames_csv=args.frames_csv)


if __name__ == "__main__":
    main()
