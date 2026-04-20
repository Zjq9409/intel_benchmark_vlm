"""
视频编码模块

根据 VLM 分析生成的 video_clip_json 对原始视频进行裁剪，
然后将所有片段拼接成最终视频。不涉及 TTS 配音。

支持硬件加速解码 + 编码：
  - Intel VAAPI  : -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -hwaccel_output_format vaapi
                   编码器 h264_vaapi，增加 -low_power on
  - NVIDIA NVENC : -hwaccel cuda -hwaccel_output_format cuda，编码器 h264_nvenc
  - 软件回退      : libx264

clip JSON 格式：
  {
    "_id": 1,
    "timestamp": "HH:MM:SS,mmm-HH:MM:SS,mmm",
    "picture": "画面描述",
    "narration": "",   # 可选，本模块不使用
    "OST": 2           # 2 = 保留原声
  }
"""

import json
import os
import subprocess
import tempfile
from typing import List, Optional

from loguru import logger
from tqdm import tqdm

# VAAPI 默认设备节点（Intel GPU，通常为 renderD128）
VAAPI_DEVICE_DEFAULT = "/dev/dri/renderD128"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str) -> float:
    """将 HH:MM:SS,mmm 或 HH:MM:SS.mmm 解析为秒数"""
    ts = ts.strip()
    try:
        if "," in ts:
            time_part, ms_part = ts.split(",", 1)
        elif "." in ts:
            time_part, ms_part = ts.split(".", 1)
        else:
            time_part, ms_part = ts, "000"
        parts = time_part.split(":")
        while len(parts) < 3:
            parts.insert(0, "0")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s + int(ms_part) / 1000
    except Exception as e:
        raise ValueError(f"无法解析时间戳 '{ts}': {e}")


def _ffmpeg_has_encoder(name: str) -> bool:
    """检查 ffmpeg 是否内置了指定编码器"""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=False,
        )
        return name.lower() in r.stdout.lower()
    except Exception:
        return False


def _test_hwenc(enc: str, extra_decode: list, extra_encode: list) -> bool:
    """用 nullsrc 测试某个硬件编码器 + 解码器组合是否可用"""
    try:
        cmd = (
            ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            + extra_decode
            + ["-f", "lavfi", "-i", "nullsrc=s=128x72:d=0.1",
               "-frames:v", "1", "-c:v", enc]
            + extra_encode
            + ["-f", "null", "-"]
        )
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        return r.returncode == 0
    except Exception:
        return False




def _build_decode_args(hwaccel_type: str, vaapi_device: str) -> list:
    """生成放在 -i 之前的硬件解码参数"""
    if hwaccel_type == "vaapi":
        return ["-hwaccel", "vaapi",
                "-hwaccel_device", vaapi_device,
                "-hwaccel_output_format", "vaapi"]
    if hwaccel_type == "cuda":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    return []


def _build_encode_args(encoder: str) -> list:
    """生成编码器专属参数（跟在 -c:v encoder 后面）"""
    if encoder == "h264_vaapi":
        return ["-qp", "23", "-low_power", "on"]
    if encoder == "h264_nvenc":
        return ["-cq", "23", "-preset", "medium"]
    # libx264：加像素格式（避免 yuv444 等不兼容格式）
    return ["-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast"]


# ---------------------------------------------------------------------------
# 裁剪单个片段
# ---------------------------------------------------------------------------

def _clip_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    encoder: str,
    hwaccel_type: str,
    keep_audio: bool,
    vaapi_device: str = VAAPI_DEVICE_DEFAULT,
) -> bool:
    """裁剪视频片段，返回是否成功"""
    duration = end_sec - start_sec
    if duration <= 0:
        logger.warning("片段时长 <= 0，跳过: %.3f-%.3f", start_sec, end_sec)
        return False

    decode_args = _build_decode_args(hwaccel_type, vaapi_device)
    encode_args = _build_encode_args(encoder)

    cmd = (
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        + decode_args
        + ["-ss", f"{start_sec:.3f}", "-i", video_path,
           "-t", f"{duration:.3f}",
           "-c:v", encoder]
        + encode_args
    )
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd.append(output_path)

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.CalledProcessError as e:
        if encoder != "libx264":
            logger.warning("硬件编码失败，回退到 libx264: %s", e.stderr[:120])
            return _clip_segment(
                video_path, start_sec, end_sec, output_path,
                "libx264", "none", keep_audio, vaapi_device,
            )
        logger.error("片段裁剪失败: %s", e.stderr[:200])
        return False
    except subprocess.TimeoutExpired:
        logger.error("片段裁剪超时: %.1f-%.1f s", start_sec, end_sec)
        return False


# ---------------------------------------------------------------------------
# 拼接片段
# ---------------------------------------------------------------------------

def _concat_segments(segment_paths: List[str], output_path: str) -> None:
    """使用 ffmpeg concat demuxer 将片段无损拼接"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        concat_file = f.name
        for p in segment_paths:
            safe = p.replace("\\", "/")
            f.write(f"file {repr(safe)}\n")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    finally:
        try:
            os.remove(concat_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def encode_video(
    video_path: str,
    clips: List[dict],
    output_path: str,
    encoder: str,
    keep_audio: bool = True,
    temp_dir: Optional[str] = None,
    vaapi_device: str = VAAPI_DEVICE_DEFAULT,
) -> str:
    """
    根据 video_clip_json 裁剪并拼接视频。

    Args:
        video_path:   原始视频文件路径
        clips:        video_clip_json 列表，每项需含 timestamp 字段
        output_path:  最终输出视频路径
        encoder:      编码器，必须显式指定；可选值：
                        h264_vaapi / h264_nvenc / libx264
        keep_audio:   是否保留原声（OST=0 的片段始终静音）
        temp_dir:     临时目录（None 时自动创建并清理）
        vaapi_device: Intel VAAPI 设备节点（默认 /dev/dri/renderD128）

    Returns:
        输出视频路径
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"原始视频不存在: {video_path}")
    if not clips:
        raise ValueError("clips 列表为空")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    _SUPPORTED = {"h264_vaapi", "hevc_vaapi", "h264_nvenc", "hevc_nvenc", "libx264", "libx265"}
    if encoder not in _SUPPORTED:
        raise ValueError(
            f"不支持的编码器: {encoder!r}\n"
            f"请从以下选项中指定: {', '.join(sorted(_SUPPORTED))}"
        )
    _hwaccel_map = {
        "h264_vaapi": "vaapi", "hevc_vaapi": "vaapi",
        "h264_nvenc": "cuda", "hevc_nvenc": "cuda",
    }
    enc = encoder
    hwaccel_type = _hwaccel_map.get(encoder, "none")
    logger.info(f"编码器: {enc}  hwaccel: {hwaccel_type}")

    use_temp = temp_dir is None
    if use_temp:
        temp_dir = tempfile.mkdtemp(prefix="pipeline_enc_")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    segment_paths: List[str] = []
    skipped = 0

    try:
        with tqdm(total=len(clips), desc="裁剪片段", unit="片段") as pbar:
            for clip in clips:
                ts = clip.get("timestamp", "")
                if not ts or "-" not in ts:
                    logger.warning("片段 %s timestamp 格式错误，跳过: %s", clip.get("_id"), ts)
                    skipped += 1
                    pbar.update(1)
                    continue

                parts = ts.split("-", 1)
                try:
                    start = _parse_timestamp(parts[0])
                    end = _parse_timestamp(parts[1])
                except ValueError as e:
                    logger.warning("timestamp 解析失败，跳过: %s", e)
                    skipped += 1
                    pbar.update(1)
                    continue

                seg_id = clip.get("_id", len(segment_paths))
                seg_out = os.path.join(temp_dir, f"seg_{seg_id:04d}.mp4")
                ost = clip.get("OST", 2)
                seg_keep_audio = keep_audio and (ost != 0)

                ok = _clip_segment(
                    video_path, start, end, seg_out,
                    enc, hwaccel_type, seg_keep_audio, vaapi_device,
                )
                if ok:
                    segment_paths.append(seg_out)
                else:
                    skipped += 1
                pbar.update(1)

        if not segment_paths:
            raise RuntimeError("所有片段均裁剪失败，无法生成输出视频")

        logger.info(f"裁剪完成: {len(segment_paths)} 成功，{skipped} 跳过")
        logger.info(f"拼接 {len(segment_paths)} 个片段 -> {output_path}")

        if len(segment_paths) == 1:
            import shutil
            shutil.copy2(segment_paths[0], output_path)
        else:
            _concat_segments(segment_paths, output_path)

    finally:
        if use_temp:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(f"视频编码完成: {output_path}")
    return output_path


def load_clips_from_json(json_path: str) -> List[dict]:
    """从 JSON 文件加载 video_clip_json，支持列表格式和 artifact dict 格式"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "video_clip_json" in data:
        return data["video_clip_json"]
    raise ValueError(f"无法识别 JSON 格式: {json_path}")
