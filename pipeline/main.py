#!/usr/bin/env python3
"""
Pipeline 主入口

用法示例（命令行）：

  # 完整流程：解码 + 抽帧 + VLM 分析
  python main.py run \
    --videos /data/v1.mp4 /data/v2.mp4 \
    --output /data/output \
    --vlm-url http://localhost:8000/v1 \
      --vlm-model Qwen2-VL-7B

  # 可选：开启 LLM 生成视频 summary
  python main.py run ... --llm-url http://... --llm-model Qwen2.5-7B

  # 只抽帧
  python main.py extract --videos /data/v.mp4 --output /data/out --interval 5
"""

import argparse
import time
import os
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

from frame_extractor import extract_frames, extract_frames_batch, _get_video_info as get_video_info
from vlm_analyzer import (
    analyze_frames,
    build_analysis_artifact,
    save_analysis_artifact,
    generate_video_summary,
)


def _setup_logging(log_dir: Optional[str] = None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger.add(os.path.join(log_dir, "pipeline_{time}.log"), level="DEBUG", rotation="50 MB")


def run_pipeline(
    video_paths: List[str],
    output_root: str,
    vlm_url: str,
    vlm_model: str,
    interval_seconds: float = 5.0,
    batch_size: int = 8,
    max_concurrency: int = 2,
    custom_prompt: str = "",
    vlm_timeout: int = 120,
    hwaccel: Optional[str] = None,
    hwaccel_device: Optional[str] = None,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> dict:
    """
    完整 pipeline：解码 -> 抽帧 -> VLM 分析 -> (可选 LLM summary)

    Returns:
        dict，key 为视频路径，value 为该视频的输出结果 dict
    """
    results = {}

    for vp in video_paths:
        video_name = Path(vp).stem
        video_out_dir = os.path.join(output_root, video_name)
        frames_dir = os.path.join(video_out_dir, "frames")
        analysis_path = os.path.join(video_out_dir, "analysis.json")
        summary_path = os.path.join(video_out_dir, "summary.txt")
        video_summary = ""

        logger.info(f"========== 处理视频: {video_name} ==========")
        timing: dict = {}

        vinfo = get_video_info(vp)
        logger.info(
            f"原始视频: {vinfo.get('width', '?')}x{vinfo.get('height', '?')}  "
            f"fps={float(vinfo.get('fps', 0)):.2f}  "
            f"时长={float(vinfo.get('duration', 0)):.1f}s  "
            f"codec={vinfo.get('codec_name', 'unknown')}"
        )

        # 步骤 1：解码 + 抽帧
        logger.info("步骤 1/2: 解码 + 抽帧")
        t0 = time.perf_counter()
        try:
            # frame_paths = extract_frames(vp, frames_dir, interval_seconds)
        #     results = extract_frames_batch(
        #     video_paths=args.videos,
        #     output_root=args.output,
        #     interval_seconds=args.interval,
        #     hwaccel=args.hwaccel,
        #     hwaccel_device=args.hwaccel_device,
        # )
            frames = extract_frames(vp, out_dir, interval_seconds, hwaccel=hwaccel, hwaccel_device=hwaccel_device)
            t_extract = time.perf_counter() - t0
            timing["extract_frames_s"] = round(t_extract, 3)
            frame_w, frame_h = "?", "?"
            if frame_paths:
                try:
                    from PIL import Image
                    with Image.open(frame_paths[0]) as img:
                        frame_w, frame_h = img.size
                except Exception:
                    pass
            logger.info(
                f"抽帧完成: {len(frame_paths)} 帧  [{t_extract:.2f}s]  "
                f"VLM 输入图片尺寸: {frame_w}x{frame_h}"
            )
        except Exception as e:
            logger.error(f"[{video_name}] 抽帧失败: {e}")
            results[vp] = {"error": str(e), "stage": "extract_frames"}
            continue

        # 步骤 2：VLM 分析
        logger.info("步骤 2/2: VLM 分析")
        t0 = time.perf_counter()
        try:
            batch_results = analyze_frames(
                frame_paths=frame_paths,
                model=vlm_model,
                base_url=vlm_url,
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                custom_prompt=custom_prompt,
                timeout=vlm_timeout,
            )
            artifact = build_analysis_artifact(batch_results, vp, interval_seconds, vlm_model, vlm_url)
            save_analysis_artifact(artifact, analysis_path)
            clips = artifact["video_clip_json"]
            t_vlm = time.perf_counter() - t0
            timing["vlm_analyze_s"] = round(t_vlm, 3)
            logger.info(f"VLM 分析完成: {len(clips)} 个片段  [{t_vlm:.2f}s]")

            # 可选：LLM 生成视频整体 summary
            if llm_url and llm_model:
                logger.info("生成视频整体 summary ...")
                video_summary = generate_video_summary(
                    batch_results=batch_results,
                    model=llm_model,
                    base_url=llm_url,
                )
                artifact["video_summary"] = video_summary
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(video_summary)
                save_analysis_artifact(artifact, analysis_path)
                logger.info(f"视频 summary 已保存: {summary_path}")

        except Exception as e:
            logger.error(f"[{video_name}] VLM 分析失败: {e}")
            results[vp] = {"error": str(e), "stage": "vlm_analyze"}
            continue

        total_s = sum(timing.values())
        timing["total_s"] = round(total_s, 3)
        logger.info(
            f"[{video_name}] 全流程完成 ✓  "
            f"抽帧={timing.get('extract_frames_s', '-')}s  "
            f"VLM={timing.get('vlm_analyze_s', '-')}s  "
            f"合计={total_s:.2f}s"
        )

        results[vp] = {
            "analysis_json": analysis_path,
            "summary_txt": summary_path if video_summary else None,
            "video_summary": video_summary,
            "clips": clips,
            "timing": timing,
        }

    return results


# ---------------------------------------------------------------------------
# CLI 子命令
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    run_pipeline(
        video_paths=args.videos,
        output_root=args.output,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        interval_seconds=args.interval,
        batch_size=args.batch_size,
        max_concurrency=args.concurrency,
        custom_prompt=args.prompt or "",
        vlm_timeout=args.timeout,
        hwaccel=args.hwaccel,
        hwaccel_device=args.hwaccel_device,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
    )


def _cmd_extract(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    results = extract_frames_batch(
        video_paths=args.videos,
        output_root=args.output,
        interval_seconds=args.interval,
        hwaccel=args.hwaccel,
        hwaccel_device=args.hwaccel_device,
    )
    elapsed = time.perf_counter() - t0
    for vp, frames in results.items():
        logger.info(f"{Path(vp).name}: {len(frames)} 帧  [{elapsed:.2f}s]")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description="视频 VLM 分析 Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="完整流程：抽帧 + VLM 分析")
    run_p.add_argument("--videos", nargs="+", required=True, help="输入视频路径（可多个）")
    run_p.add_argument("--output", required=True, help="输出根目录")
    run_p.add_argument("--vlm-url", required=True, help="VLM API base URL（OpenAI 兼容）")
    run_p.add_argument("--vlm-model", required=True, help="VLM 模型名称")
    run_p.add_argument("--interval", type=float, default=5.0, help="抽帧间隔秒数（默认 5）")
    run_p.add_argument("--batch-size", type=int, default=8, help="每批帧数（默认 8）")
    run_p.add_argument("--concurrency", type=int, default=2, help="VLM 并发批次数（默认 2）")
    run_p.add_argument("--prompt", default="", help="自定义 VLM 提示词（可选）")
    run_p.add_argument("--timeout", type=int, default=120, help="VLM 请求超时秒数（默认 120）")
    run_p.add_argument("--hwaccel", default=None, help="ffmpeg 硬件解码方式（可选）：vaapi / cuda / qsv")
    run_p.add_argument("--hwaccel-device", default=None, help="硬件设备（可选），CUDA 传 GPU 编号如 0，VAAPI 传 /dev/dri/renderD128")
    run_p.add_argument("--llm-url", default=None, help="（可选）LLM API base URL，用于生成视频 summary")
    run_p.add_argument("--llm-model", default=None, help="（可选）LLM 模型名称")
    run_p.set_defaults(func=_cmd_run)

    # --- extract ---
    ext_p = sub.add_parser("extract", help="只抽帧，不做 VLM 分析")
    ext_p.add_argument("--videos", nargs="+", required=True)
    ext_p.add_argument("--output", required=True, help="输出根目录")
    ext_p.add_argument("--interval", type=float, default=5.0)
    ext_p.add_argument("--hwaccel", default=None, help="ffmpeg 硬件解码方式（可选）：vaapi / cuda / qsv")
    ext_p.add_argument("--hwaccel-device", default=None, help="ffmpeg 硬件设备（可选），如 /dev/dri/renderD128")
    ext_p.set_defaults(func=_cmd_extract)

    return parser


def main() -> None:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
