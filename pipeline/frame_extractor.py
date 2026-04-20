"""
视频抽帧模块

按固定时间间隔从视频中提取帧，保存为 JPEG 图片，
文件名编码时间戳信息供后续模块使用。
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm


def _get_video_info(video_path: str) -> dict:
    """通过 ffprobe 获取视频基本信息"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v
        if "r_frame_rate" in info:
            try:
                num, den = map(int, info["r_frame_rate"].split("/"))
                info["fps"] = str(num / den)
            except ValueError:
                info["fps"] = info.get("r_frame_rate", "25")
        return info
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffprobe 获取视频信息失败: {e.stderr}")
        return {"fps": "25", "duration": "0", "width": "1280", "height": "720"}


def _format_timestamp_token(timestamp_sec: float) -> str:
    """将秒数格式化为 HHMMSSmmm 字符串，编码到文件名中"""
    hours = int(timestamp_sec // 3600)
    minutes = int((timestamp_sec % 3600) // 60)
    seconds = int(timestamp_sec % 60)
    milliseconds = int((timestamp_sec % 1) * 1000)
    return f"{hours:02d}{minutes:02d}{seconds:02d}{milliseconds:03d}"


def _fast_extract(video_path: str, output_dir: str, interval_seconds: float, fps: float) -> List[str]:
    """单次 ffmpeg 调用按固定间隔批量抽帧（快速路径）"""
    raw_pattern = os.path.join(output_dir, "fastframe_%06d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "2",
        "-start_number", "0",
        "-y", raw_pattern,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)

    raw_files = sorted(
        f for f in os.listdir(output_dir)
        if re.fullmatch(r"fastframe_\d{6}\.jpg", f)
    )
    if not raw_files:
        raise RuntimeError("快速路径抽帧未生成任何文件")

    renamed: List[str] = []
    for idx, filename in enumerate(raw_files):
        ts = idx * interval_seconds
        frame_no = int(ts * fps)
        token = _format_timestamp_token(ts)
        src = os.path.join(output_dir, filename)
        dst = os.path.join(output_dir, f"keyframe_{frame_no:06d}_{token}.jpg")
        os.replace(src, dst)
        renamed.append(dst)
    return renamed


def _extract_single_frame(video_path: str, timestamp_sec: float, output_path: str) -> bool:
    """提取单帧（兼容回退路径），优先 PNG 中转避免 MJPEG 问题"""
    png_path = output_path.replace(".jpg", ".png")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-vframes", "1", "-f", "image2",
        "-y", png_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            return False
        try:
            from PIL import Image
            with Image.open(png_path) as img:
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                img.save(output_path, "JPEG", quality=90)
            os.remove(png_path)
        except Exception:
            os.rename(png_path, output_path)
        return True
    except Exception as e:
        logger.debug(f"单帧提取失败 {timestamp_sec:.1f}s: {e}")
        return False


def _compat_extract(video_path: str, output_dir: str, interval_seconds: float, fps: float, duration: float) -> List[str]:
    """逐帧兼容回退路径"""
    times, t = [], 0.0
    while t < duration:
        times.append(t)
        t += interval_seconds

    results: List[str] = []
    ok = fail = 0
    with tqdm(total=len(times), desc="抽帧(兼容)", unit="帧") as pbar:
        for ts in times:
            frame_no = int(ts * fps)
            token = _format_timestamp_token(ts)
            out = os.path.join(output_dir, f"keyframe_{frame_no:06d}_{token}.jpg")
            if _extract_single_frame(video_path, ts, out):
                ok += 1
                results.append(out)
            else:
                fail += 1
            pbar.set_postfix(ok=ok, fail=fail)
            pbar.update(1)

    logger.info(f"兼容路径抽帧完成: 成功 {ok}/{len(times)}")
    return results


def _cleanup_fastframe_artifacts(output_dir: str) -> None:
    for name in os.listdir(output_dir):
        if re.fullmatch(r"fastframe_\d{6}\.jpg", name):
            try:
                os.remove(os.path.join(output_dir, name))
            except OSError:
                pass


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 5.0,
) -> List[str]:
    """
    从单个视频中按固定间隔抽帧。

    Args:
        video_path: 视频文件路径
        output_dir: 帧图片输出目录
        interval_seconds: 抽帧间隔（秒），默认 5 秒

    Returns:
        按时间顺序排列的帧图片路径列表
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds 必须大于 0")

    os.makedirs(output_dir, exist_ok=True)

    info = _get_video_info(video_path)
    fps = float(info.get("fps", 25))
    duration = float(info.get("duration", 0))
    logger.info(f"视频信息: fps={fps:.2f}, duration={duration:.1f}s, 预计抽帧 {int(duration / interval_seconds) + 1} 张")

    try:
        logger.info("使用快速路径抽帧...")
        frames = _fast_extract(video_path, output_dir, interval_seconds, fps)
        logger.info(f"快速路径完成，共 {len(frames)} 帧")
        return frames
    except Exception as e:
        logger.warning(f"快速路径失败，回退到兼容路径: {e}")
        _cleanup_fastframe_artifacts(output_dir)

    frames = _compat_extract(video_path, output_dir, interval_seconds, fps, duration)
    if not frames:
        raise RuntimeError(f"视频 {video_path} 抽帧完全失败，请检查 ffmpeg 安装")
    return frames


def extract_frames_batch(
    video_paths: List[str],
    output_root: str,
    interval_seconds: float = 5.0,
) -> dict:
    """
    批量处理多个视频的抽帧。

    Returns:
        dict，key 为视频路径，value 为对应帧路径列表
    """
    results = {}
    for vp in video_paths:
        video_name = Path(vp).stem
        out_dir = os.path.join(output_root, video_name, "frames")
        logger.info(f"[{video_name}] 开始抽帧 -> {out_dir}")
        try:
            frames = extract_frames(vp, out_dir, interval_seconds)
            results[vp] = frames
            logger.info(f"[{video_name}] 抽帧完成，共 {len(frames)} 帧")
        except Exception as e:
            logger.error(f"[{video_name}] 抽帧失败: {e}")
            results[vp] = []
    return results
