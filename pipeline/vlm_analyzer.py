"""
VLM 视觉分析模块

将抽帧结果批量发送给 VLM（视觉语言模型）进行分析，
输出结构化 JSON，包含每帧的观察描述和每批次的整体总结。

支持 OpenAI 兼容接口（openai / vllm / ollama 等）。
LLM 后处理为可选步骤，默认不启用。
"""

import asyncio
import base64
import json
import os
import re
from datetime import datetime
from typing import Any, List, Optional

import httpx
from loguru import logger
from tqdm import tqdm


def _timestamp_from_keyframe_name(filename: str) -> str:
    match = re.search(r"keyframe_\d{6}_(\d{9})\.jpg$", os.path.basename(filename))
    if not match:
        return "00:00:00,000"
    token = match.group(1)
    h, m, s, ms = int(token[0:2]), int(token[2:4]), int(token[4:6]), int(token[6:9])
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _time_range_from_batch(batch_files: List[str]) -> str:
    if not batch_files:
        return "00:00:00,000-00:00:00,000"
    start = _timestamp_from_keyframe_name(batch_files[0])
    end = _timestamp_from_keyframe_name(batch_files[-1])
    return f"{start}-{end}"


def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_messages(frame_paths: List[str], prompt: str) -> List[dict]:
    content: List[dict] = [{"type": "text", "text": prompt}]
    for fp in frame_paths:
        b64 = _encode_image_b64(fp)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [{"role": "user", "content": content}]


DEFAULT_PROMPT_TEMPLATE = """
我提供了 {frame_count} 张视频帧，它们按时间顺序排列，代表一个连续的视频片段。
首先，请详细描述每一帧的关键视觉信息（主要内容、人物、动作、场景）。
然后，基于所有帧，用简洁的语言总结整个视频片段发生的主要活动或事件流程。
请以 JSON 格式输出，结构如下：
{{
  "frame_observations": [
    {{"timestamp": "HH:MM:SS,mmm", "observation": "帧描述"}}
  ],
  "overall_activity_summary": "整体活动总结"
}}
frame_observations 长度必须等于 {frame_count}。
只输出 JSON，不要附加任何说明文字。
""".strip()


async def _call_vlm_api(messages, api_key, model, base_url, timeout=120):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 4096, "temperature": 0.2}
    url = base_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _clean_json_response(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _parse_batch_response(raw, frame_paths, time_range, batch_index):
    try:
        parsed = json.loads(_clean_json_response(raw))
        observations = parsed.get("frame_observations", [])
        for i, obs in enumerate(observations):
            if not obs.get("timestamp") and i < len(frame_paths):
                obs["timestamp"] = _timestamp_from_keyframe_name(frame_paths[i])
        return {
            "batch_index": batch_index, "time_range": time_range, "status": "success",
            "frame_paths": frame_paths, "frame_observations": observations,
            "overall_activity_summary": parsed.get("overall_activity_summary", ""),
            "fallback_summary": "", "raw_response": raw, "error_message": "",
        }
    except json.JSONDecodeError as e:
        logger.warning(f"批次 {batch_index} JSON 解析失败: {e}")
        return {
            "batch_index": batch_index, "time_range": time_range, "status": "failed",
            "frame_paths": frame_paths, "frame_observations": [],
            "overall_activity_summary": "", "fallback_summary": raw[:500],
            "raw_response": raw, "error_message": str(e),
        }


async def _analyze_batches_async(batches, api_key, model, base_url, custom_prompt, max_concurrency, timeout):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_one(batch_files, batch_idx):
        time_range = _time_range_from_batch(batch_files)
        prompt = (custom_prompt or DEFAULT_PROMPT_TEMPLATE).format(frame_count=len(batch_files))
        messages = _build_messages(batch_files, prompt)
        async with semaphore:
            try:
                raw = await _call_vlm_api(messages, api_key, model, base_url, timeout)
                return _parse_batch_response(raw, batch_files, time_range, batch_idx)
            except Exception as e:
                logger.error(f"批次 {batch_idx} VLM 调用失败: {e}")
                return {
                    "batch_index": batch_idx, "time_range": time_range, "status": "failed",
                    "frame_paths": batch_files, "frame_observations": [],
                    "overall_activity_summary": "", "fallback_summary": f"分析失败: {e}",
                    "raw_response": "", "error_message": str(e),
                }

    tasks = [analyze_one(b, i) for i, b in enumerate(batches)]
    results = []
    with tqdm(total=len(tasks), desc="VLM 分析", unit="批次") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    results.sort(key=lambda r: r["batch_index"])
    return results


def analyze_frames(
    frame_paths: List[str],
    api_key: str,
    model: str,
    base_url: str,
    batch_size: int = 8,
    max_concurrency: int = 2,
    custom_prompt: str = "",
    timeout: int = 120,
) -> List[dict]:
    """
    将帧列表分批送给 VLM 分析。

    Args:
        frame_paths: 抽帧文件路径列表（需按时间排序）
        api_key: VLM API Key
        model: 模型名称
        base_url: OpenAI 兼容接口 base URL
        batch_size: 每批发送的帧数
        max_concurrency: 最大并发批次数
        custom_prompt: 自定义提示词（可选）
        timeout: 单次请求超时秒数

    Returns:
        批次分析结果列表
    """
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")
    batches = [frame_paths[i:i + batch_size] for i in range(0, len(frame_paths), batch_size)]
    logger.info(f"共 {len(frame_paths)} 帧，分为 {len(batches)} 批（batch_size={batch_size}）")

    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                _analyze_batches_async(batches, api_key, model, base_url, custom_prompt, max_concurrency, timeout),
            ).result()
    except RuntimeError:
        return asyncio.run(
            _analyze_batches_async(batches, api_key, model, base_url, custom_prompt, max_concurrency, timeout)
        )


def build_analysis_artifact(batch_results, video_path, frame_interval_seconds, model, base_url):
    clips = []
    for batch in batch_results:
        picture = batch.get("overall_activity_summary") or batch.get("fallback_summary") or ""
        if not picture:
            observations = batch.get("frame_observations", [])
            picture = " ".join(o.get("observation", "") for o in observations if o.get("observation"))
        clips.append({
            "_id": batch["batch_index"] + 1,
            "timestamp": batch["time_range"],
            "picture": picture,
            "narration": "",
            "OST": 2,
        })
    return {
        "artifact_version": "pipeline-vlm-v1",
        "generated_at": datetime.now().isoformat(),
        "video_path": video_path,
        "frame_interval_seconds": frame_interval_seconds,
        "vision_model": model,
        "vision_base_url": base_url,
        "batches": batch_results,
        "video_clip_json": clips,
    }


def save_analysis_artifact(artifact: dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    logger.info(f"分析结果已保存: {output_path}")
    return output_path


def enrich_narration_with_llm(
    clips: List[dict],
    api_key: str,
    model: str,
    base_url: str,
    system_prompt: str = "你是一位专业的视频解说撰稿人，请根据画面描述写出简洁有吸引力的旁白解说词。",
    timeout: int = 60,
) -> List[dict]:
    """
    （可选）逐条调用 LLM，将 picture 字段的画面描述转为 narration 解说文案。
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    enriched = []
    for clip in tqdm(clips, desc="LLM 生成解说", unit="片段"):
        picture = clip.get("picture", "")
        if not picture:
            enriched.append(clip)
            continue
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"画面描述：{picture}\n请写出对应的旁白解说词（50字以内）。"},
            ],
            "max_tokens": 200, "temperature": 0.7,
        }
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            narration = resp.json()["choices"][0]["message"]["content"].strip()
            enriched.append({**clip, "narration": narration, "OST": 0})
        except Exception as e:
            logger.warning(f"片段 {clip.get('_id')} LLM 解说生成失败: {e}")
            enriched.append(clip)
    return enriched
