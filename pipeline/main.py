#!/usr/bin/env python3
"""
Pipeline 主入口

用法示例（命令行）：

  # 完整流程：抽帧 + VLM 分析 + 视频编码
  python main.py run \
    --videos /data/v1.mp4 /data/v2.mp4 \
    --output /data/output \
    --vlm-url http://localhost:8000/v1 \
    --vlm-key EMPTY \
    --vlm-model Qwen2-VL-7B

  # 可选：开启 LLM 生成解说词
  python main.py run ... --llm-url http://... --llm-key KEY --llm-model Qwen2.5-7B

  # 只抽帧
  python main.py extract --videos /data/v.mp4 --output /data/out --interval 5

  # 用已有 JSON 编码视频
  python main.py encode --video /data/v.mp4 --json /data/out/v/analysis.json --output /data/final.mp4
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
    enrich_narration_with_llm,
)
from video_encoder import encode_video, load_clips_from_json


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
    vlm_key: str,
    vlm_model: str,
    interval_seconds: float = 5.0,
    batch_size: int = 8,
    max_concurrency: int = 2,
    custom_prompt: str = "",
    vlm_timeout: int = 120,
    encode: bool = True,
    encoder: Optional[str] = None,
    vaapi_device: str = "/dev/dri/renderD128",
    llm_url: Optional[str] = None,
    llm_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_system_prompt: str = "你是一位专业的视频解说撰稿人，请根据画面描述写出简洁有吸引力的旁白解说词。",
) -> dict:
    """
    完整 pipeline：抽帧 -> VLM 分析 -> (可选 LLM 解说) -> 视频编码

    Returns:
        dict，key 为视频路径，value 为该视频的输出结果 dict
    """
    results = {}

    for vp in video_paths:
        video_name = Path(vp).stem
        video_out_dir = os.path.join(output_root, video_name)
        frames_dir = os.path.join(video_out_dir, "frames")
        analysis_path = os.path.join(video_out_dir, "analysis.json")
        final_video_path = os.path.join(video_out_dir, f"{video_name}_output.mp4")

        logger.info(f"========== 处理视频: {video_name} ==========")
        timing: dict = {}

        # 原始视频分辨率
        vinfo = get_video_info(vp)
        logger.info(
            f"原始视频: {vinfo.get('width', '?')}x{vinfo.get('height', '?')}  "
            f"fps={float(vinfo.get('fps', 0)):.2f}  "
            f"时长={float(vinfo.get('duration', 0)):.1f}s"
        )

        # 步骤 1：抽帧（ffmpeg 解码）
        logger.info("步骤 1/3: 抽帧（ffmpeg 解码）")
        t0 = time.perf_counter()
        try:
            frame_paths = extract_frames(vp, frames_dir, interval_seconds)
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
        logger.info("步骤 2/3: VLM 分析")
        t0 = time.perf_counter()
        try:
            batch_results = analyze_frames(
                frame_paths=frame_paths,
                api_key=vlm_key,
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
        except Exception as e:
            logger.error(f"[{video_name}] VLM 分析失败: {e}")
            results[vp] = {"error": str(e), "stage": "vlm_analyze"}
            continue

        # 步骤 2.5：可选 LLM 解说生成
        if llm_url and llm_key and llm_model:
            logger.info("步骤 2.5/3: LLM 生成解说词（可选）")
            t0 = time.perf_counter()
            try:
                clips = enrich_narration_with_llm(
                    clips=clips,
                    api_key=llm_key,
                    model=llm_model,
                    base_url=llm_url,
                    system_prompt=llm_system_prompt,
                )
                artifact["video_clip_json"] = clips
                save_analysis_artifact(artifact, analysis_path)
                t_llm = time.perf_counter() - t0
                timing["llm_narration_s"] = round(t_llm, 3)
                logger.info(f"LLM 解说生成完成  [{t_llm:.2f}s]")
            except Exception as e:
                logger.warning(f"[{video_name}] LLM 解说生成失败（继续流程）: {e}")

        # 步骤 3：视频编码
        if encode:
            if not encoder:
                raise ValueError("必须通过 --encoder 指定编码器（h264_vaapi / h264_nvenc / libx264）")
            logger.info("步骤 3/3: 视频编码")
            t0 = time.perf_counter()
            try:
                encode_video(
                    video_path=vp,
                    clips=clips,
                    output_path=final_video_path,
                    encoder=encoder,
                    keep_audio=True,
                    vaapi_device=vaapi_device,
                )
                t_enc = time.perf_counter() - t0
                timing["encode_video_s"] = round(t_enc, 3)
                logger.info(f"视频编码完成: {final_video_path}  [{t_enc:.2f}s]")
            except Exception as e:
                logger.error(f"[{video_name}] 视频编码失败: {e}")
                results[vp] = {
                    "analysis_json": analysis_path, "clips": clips,
                    "timing": timing, "error": str(e), "stage": "encode_video",
                }
                continue

        total_s = sum(timing.values())
        timing["total_s"] = round(total_s, 3)
        logger.info(
            f"[{video_name}] 全流程完成 ✓  "
            f"抽帧={timing.get('extract_frames_s', '-')}s  "
            f"VLM={timing.get('vlm_analyze_s', '-')}s  "
            f"编码={timing.get('encode_video_s', '-')}s  "
            f"合计={total_s:.2f}s"
        )

        results[vp] = {
            "analysis_json": analysis_path,
            "clips": clips,
            "output_video": final_video_path if encode else None,
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
        vlm_key=args.vlm_key,
        vlm_model=args.vlm_model,
        interval_seconds=args.interval,
        batch_size=args.batch_size,
        max_concurrency=args.concurrency,
        custom_prompt=args.prompt or "",
        vlm_timeout=args.timeout,
        encode=not args.no_encode,
        encoder=args.encoder,
        vaapi_device=args.vaapi_device,
        llm_url=args.llm_url,
        llm_key=args.llm_key,
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


def _cmd_encode(args: argparse.Namespace) -> None:
    clips = load_clips_from_json(args.json)
    encode_video(
        video_path=args.video,
        clips=clips,
        output_path=args.output,
        encoder=args.encoder,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description="视频 VLM 分析 Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="完整流程：抽帧 + VLM 分析 + 视频编码")
    run_p.add_argument("--videos", nargs="+", required=True, help="输入视频路径（可多个）")
    run_p.add_argument("--output", required=True, help="输出根目录")
    run_p.add_argument("--vlm-url", required=True, help="VLM API base URL（OpenAI 兼容）")
    run_p.add_argument("--vlm-key", default="EMPTY", help="VLM API Key")
    run_p.add_argument("--vlm-model", required=True, help="VLM 模型名称")
    run_p.add_argument("--interval", type=float, default=5.0, help="抽帧间隔秒数（默认 5）")
    run_p.add_argument("--batch-size", type=int, default=8, help="每批帧数（默认 8）")
    run_p.add_argument("--concurrency", type=int, default=2, help="VLM 并发批次数（默认 2）")
    run_p.add_argument("--prompt", default="", help="自定义 VLM 提示词")
    run_p.add_argument("--timeout", type=int, default=120, help="VLM 请求超时秒数（默认 120）")
    run_p.add_argument("--no-encode", action="store_true", help="只分析，不编码视频")
    run_p.add_argument("--encoder", required=True,
                        help="编码器，必须指定：h264_vaapi / h264_nvenc / libx264")
    run_p.add_argument("--vaapi-device", default="/dev/dri/renderD128",
                        help="Intel VAAPI 设备节点（默认 /dev/dri/renderD128）")
    run_p.add_argument("--llm-url", default=None, help="（可选）LLM API base URL，用于生成解说词")
    run_p.add_argument("--llm-key", default=None, help="（可选）LLM API Key")
    run_p.add_argument("--llm-model", default=None, help="（可选）LLM 模型名称")
    run_p.set_defaults(func=_cmd_run)

    # extract
    ext_p = sub.add_parser("extract", help="只抽帧，不做 VLM 分析")
    ext_p.add_argument("--videos", nargs="+", required=True)
    ext_p.add_argument("--output", required=True, help="输出根目录")
    ext_p.add_argument("--interval", type=float, default=5.0)
    ext_p.add_argument("--hwaccel", default=None, help="ffmpeg 硬件解码方式（可选）：vaapi / cuda / qsv")
    ext_p.add_argument("--hwaccel-device", default=None, help="ffmpeg 硬件设备（可选），如 /dev/dri/renderD128")
    ext_p.set_defaults(func=_cmd_extract)

    # encode
    enc_p = sub.add_parser("encode", help="根据已有 JSON 直接编码视频")
    enc_p.add_argument("--video", required=True, help="原始视频路径")
    enc_p.add_argument("--json", required=True, help="analysis.json 路径")
    enc_p.add_argument("--output", required=True, help="输出视频路径")
    enc_p.add_argument("--encoder", required=True,
                        help="编码器，必须指定：h264_vaapi / h264_nvenc / libx264")
    enc_p.add_argument("--vaapi-device", default="/dev/dri/renderD128",
                        help="Intel VAAPI 设备节点（默认 /dev/dri/renderD128）")
    enc_p.set_defaults(func=_cmd_encode)

    return parser


def main() -> None:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
