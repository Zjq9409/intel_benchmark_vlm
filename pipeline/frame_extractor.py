"""
视频抽帧模块

按固定时间间隔从视频中提取帧，保存为 JPEG 图片，
文件名编码时间戳信息供后续模块使用。
"""

import os
import re
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from loguru import logger
from tqdm import tqdm

from gpu_monitor import GpuMemSampler, _shorten_gpu_name


def _get_video_info(video_path: str) -> dict:
    """通过 ffprobe 获取视频基本信息"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,codec_name,pix_fmt",
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



# NVIDIA cuvid 硬件解码器映射表
_CUVID_DECODER_MAP = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "av1": "av1_cuvid",
    "vp9": "vp9_cuvid",
    "vp8": "vp8_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vc1": "vc1_cuvid",
}


def _detect_video_codec(video_path: str) -> str:
    """使用 ffprobe 检测视频编码格式，返回小写 codec 名称"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        return result.stdout.strip().lower()
    except Exception:
        return ""


def _fast_extract(video_path: str, output_dir: str, interval_seconds: float, fps: float,
                  hwaccel: str = None, hwaccel_device: str = None) -> List[str]:
    """单次 ffmpeg 调用按固定间隔批量抽帧（快速路径）"""
    raw_pattern = os.path.join(output_dir, "fastframe_%06d.jpg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    if hwaccel_device:
        cmd += ["-hwaccel_device", hwaccel_device]
    if hwaccel == "cuda":
        codec = _detect_video_codec(video_path)
        cuvid_decoder = _CUVID_DECODER_MAP.get(codec)
        if cuvid_decoder:
            cmd += ["-c:v", cuvid_decoder]
            logger.debug(f"NVDEC 解码器: {cuvid_decoder} (codec={codec})")
        else:
            logger.debug(f"无 cuvid 解码器对应 codec={codec}，使用软解")
    cmd += [
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


def _extract_single_frame(video_path: str, timestamp_sec: float, output_path: str,
                          hwaccel: str = None, hwaccel_device: str = None) -> bool:
    """提取单帧（兼容回退路径），优先 PNG 中转避免 MJPEG 问题"""
    png_path = output_path.replace(".jpg", ".png")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    if hwaccel_device:
        cmd += ["-hwaccel_device", hwaccel_device]
    cmd += [
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


def _compat_extract(video_path: str, output_dir: str, interval_seconds: float, fps: float, duration: float,
                    hwaccel: str = None, hwaccel_device: str = None) -> List[str]:
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
            if _extract_single_frame(video_path, ts, out, hwaccel=hwaccel, hwaccel_device=hwaccel_device):
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
    hwaccel: str = None,
    hwaccel_device: str = None,
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
    _dur_raw = info.get("duration", "0")
    try:
        duration = float(_dur_raw) if _dur_raw not in ("N/A", "", None) else 0.0
    except ValueError:
        duration = 0.0
    width = info.get("width", "?")
    height = info.get("height", "?")
    codec = info.get("codec_name", "unknown")
    logger.info(f"视频信息: {width}x{height}  fps={fps:.2f}  duration={duration:.1f}s  codec={codec}  预计抽帧 {int(duration / interval_seconds) + 1} 张")

    sampler = GpuMemSampler(hwaccel, hwaccel_device).start() if hwaccel else None
    try:
        try:
            logger.info("使用快速路径抽帧...")
            frames = _fast_extract(video_path, output_dir, interval_seconds, fps, hwaccel=hwaccel, hwaccel_device=hwaccel_device)
            logger.info(f"快速路径完成，共 {len(frames)} 帧")
        except Exception as e:
            logger.warning(f"快速路径失败，回退到兼容路径: {e}")
            _cleanup_fastframe_artifacts(output_dir)
            frames = _compat_extract(video_path, output_dir, interval_seconds, fps, duration, hwaccel=hwaccel, hwaccel_device=hwaccel_device)
            if not frames:
                raise RuntimeError(f"视频 {video_path} 抽帧完全失败，请检查 ffmpeg 安装")
    finally:
        if sampler:
            sampler.stop()
            sampler.log_result()
            # 构建带设备/分辨率/编码格式的文件名
            gpu_prefix = "nv" if hwaccel == "cuda" else "intel" if hwaccel in ("vaapi", "qsv") else "gpu"
            raw_gpu_name = sampler.stats.get("gpu_name") or ""
            gpu_short = _shorten_gpu_name(raw_gpu_name) if raw_gpu_name else gpu_prefix
            dev_id = (hwaccel_device or "0").replace("/dev/dri/renderD", "renderD")
            codec = info.get("codec_name", "unknown")
            pix = info.get("pix_fmt", "")  # e.g. yuv420p10le -> 10bit
            bitdepth = "10bit" if "10" in pix else "8bit"
            video_stem = Path(video_path).stem
            stats_name = f"{gpu_prefix}_{gpu_short}_{width}x{height}_{codec}_{bitdepth}_{fps:.0f}fps_{video_stem}"
            pipeline_dir = Path(__file__).resolve().parent
            stats_dir = pipeline_dir / "gpu_stats"
            sampler.save_samples_csv(str(stats_dir / (stats_name + "_samples.csv")))
            plot_title = f"{gpu_short} | {dev_id} | {width}x{height} {codec} {bitdepth} {fps:.0f}fps | {video_stem}"
            sampler.save_plot(str(stats_dir / (stats_name + "_plot.png")), title=plot_title)
    return frames


def extract_frames_batch(
    video_paths: List[str],
    output_root: str,
    interval_seconds: float = 5.0,
    hwaccel: str = None,
    hwaccel_device: str = None,
    max_workers: Optional[int] = None,
) -> dict:
    """
    批量处理多个视频的抽帧，并行执行（类似 ffmpeg ... & ffmpeg ... &）。

    Args:
        max_workers: 最大并发数，默认等于视频数量（全并行）。
    Returns:
        dict，key 为视频路径，value 为对应帧路径列表
    """
    results = {}

    if max_workers is None:
        max_workers = len(video_paths)

    def _extract_one(vp):
        video_name = Path(vp).stem
        out_dir = os.path.join(output_root, video_name, "frames")
        logger.info(f"[{video_name}] 开始抽帧 -> {out_dir}")
        frames = extract_frames(vp, out_dir, interval_seconds, hwaccel=hwaccel, hwaccel_device=hwaccel_device)
        logger.info(f"[{video_name}] 抽帧完成，共 {len(frames)} 帧")
        return vp, frames

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_one, vp): vp for vp in video_paths}
        for future in as_completed(futures):
            vp = futures[future]
            video_name = Path(vp).stem
            try:
                vp_result, frames = future.result()
                results[vp_result] = frames
            except Exception as e:
                logger.error(f"[{video_name}] 抽帧失败: {e}")
                results[vp] = []

    return results
