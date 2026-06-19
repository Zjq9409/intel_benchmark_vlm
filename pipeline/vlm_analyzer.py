"""
VLM 视觉分析模块

将抽帧结果发送给 VLM（视觉语言模型）进行分析，
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

ANALYZE_MODE_SINGLE = "single"
ANALYZE_MODE_BATCH = "batch"


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
    return f"{_timestamp_from_keyframe_name(batch_files[0])}-{_timestamp_from_keyframe_name(batch_files[-1])}"


def _encode_image_b64(image_path: str, max_edge: int = 0) -> str:
    if max_edge > 0:
        from PIL import Image
        import io
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) > max_edge:
                scale = max_edge / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_messages(frame_paths: List[str], prompt: str, max_edge: int = 0, single_image: bool = False) -> List[dict]:
    content: List[dict] = [{"type": "text", "text": prompt}]
    image_paths = frame_paths[:1] if single_image else frame_paths
    if single_image and len(image_paths) != 1:
        raise ValueError("single_image 模式下必须只传入 1 张图片")
    for fp in image_paths:
        b64 = _encode_image_b64(fp, max_edge=max_edge)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [
        {"role": "system", "content": "你是一个只输出 JSON 的助手。只输出原始 JSON 对象，不要 markdown 代码块，不要解释，不要额外文字。"},
        {"role": "user", "content": content},
    ]


DEFAULT_SINGLE_PROMPT_TEMPLATE = """
你将看到 1 张视频帧。请用简体中文分析画面内容，并只输出一个 JSON 对象。
JSON 结构如下：
{{
  "frame_observations": [{{"timestamp": "HH:MM:SS,mmm", "observation": "该帧画面的中文描述"}}],
  "overall_activity_summary": "这张图的一句话中文总结"
}}
要求：所有描述必须使用简体中文；只输出原始 JSON，不要 markdown 代码块，不要额外说明文字。
""".strip()


DEFAULT_BATCH_PROMPT_TEMPLATE = """
你将看到 {frame_count} 张按时间顺序排列的视频帧。请用简体中文分析画面内容，并只输出一个 JSON 对象。
JSON 结构如下：
{{
  "frame_observations": [{{"timestamp": "HH:MM:SS,mmm", "observation": "该帧画面的中文描述"}}],
  "overall_activity_summary": "整个片段的一句话中文总结"
}}
要求：所有描述必须使用简体中文；frame_observations 必须正好包含 {frame_count} 项；只输出原始 JSON，不要 markdown 代码块，不要额外说明文字。
""".strip()


async def _call_vlm_api(messages: List[dict], model: str, base_url: str, timeout: int = 120, max_retries: int = 3, retry_delay: float = 2.0) -> str:
    import asyncio as _asyncio
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = base_url.rstrip("/") + "/chat/completions"
    last_exc = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (403, 429, 500, 502, 503):
                    logger.warning(f"VLM 返回 {resp.status_code}，body={resp.text[:200]}  (尝试 {attempt+1}/{max_retries})")
                    last_exc = httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                    if attempt < max_retries - 1:
                        await _asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError:
            raise
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_exc = e
            logger.warning(f"VLM 连接异常: {e}  (尝试 {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await _asyncio.sleep(retry_delay * (attempt + 1))
    raise last_exc


def _clean_json_response(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    idx = 0
    best = None
    while True:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            _obj, end = decoder.raw_decode(text, start)
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
            idx = end
        except json.JSONDecodeError:
            idx = start + 1
    if best is not None:
        return text[best[0]:best[1]].strip()
    return text.strip()


def _parse_batch_response(raw: str, frame_paths: List[str], time_range: str, batch_index: int) -> dict:
    try:
        parsed = json.loads(_clean_json_response(raw))
        observations = parsed.get("frame_observations", [])
        for i, obs in enumerate(observations):
            if not obs.get("timestamp") and i < len(frame_paths):
                obs["timestamp"] = _timestamp_from_keyframe_name(frame_paths[i])
        return {
            "batch_index": batch_index,
            "time_range": time_range,
            "status": "success",
            "frame_paths": frame_paths,
            "frame_observations": observations,
            "overall_activity_summary": parsed.get("overall_activity_summary", ""),
            "fallback_summary": "",
            "raw_response": raw,
            "error_message": "",
        }
    except json.JSONDecodeError as e:
        logger.warning(f"批次 {batch_index} JSON 解析失败: {e}，使用原始文本作为 fallback")
        return {
            "batch_index": batch_index,
            "time_range": time_range,
            "status": "failed",
            "frame_paths": frame_paths,
            "frame_observations": [],
            "overall_activity_summary": "",
            "fallback_summary": raw[:500],
            "raw_response": raw,
            "error_message": str(e),
        }


def _render_prompt(template: str, frame_count: int) -> str:
    return template.format(frame_count=frame_count)


async def _analyze_batches_async(batches: List[List[str]], model: str, base_url: str, custom_prompt: str, max_concurrency: int, timeout: int, max_edge: int = 0) -> List[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_one(batch_files: List[str], batch_idx: int) -> dict:
        time_range = _time_range_from_batch(batch_files)
        prompt = _render_prompt(custom_prompt or DEFAULT_BATCH_PROMPT_TEMPLATE, len(batch_files))
        messages = _build_messages(batch_files, prompt, max_edge=max_edge)
        async with semaphore:
            try:
                raw = await _call_vlm_api(messages, model, base_url, timeout)
                return _parse_batch_response(raw, batch_files, time_range, batch_idx)
            except Exception as e:
                logger.error(f"批次 {batch_idx} VLM 调用失败: {e}")
                return {
                    "batch_index": batch_idx,
                    "time_range": time_range,
                    "status": "failed",
                    "frame_paths": batch_files,
                    "frame_observations": [],
                    "overall_activity_summary": "",
                    "fallback_summary": f"分析失败: {e}",
                    "raw_response": "",
                    "error_message": str(e),
                }

    tasks = [analyze_one(b, i) for i, b in enumerate(batches)]
    results = []
    with tqdm(total=len(tasks), desc="VLM 分析", unit="批次") as pbar:
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
            pbar.update(1)
    results.sort(key=lambda r: r["batch_index"])
    return results


async def _analyze_single_frames_async(frame_paths: List[str], model: str, base_url: str, custom_prompt: str, max_concurrency: int, timeout: int, max_edge: int = 0) -> List[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_one(frame_path: str, frame_idx: int) -> dict:
        time_range = _time_range_from_batch([frame_path])
        prompt = _render_prompt(custom_prompt or DEFAULT_SINGLE_PROMPT_TEMPLATE, 1)
        messages = _build_messages([frame_path], prompt, max_edge=max_edge, single_image=True)
        async with semaphore:
            try:
                raw = await _call_vlm_api(messages, model, base_url, timeout)
                return _parse_batch_response(raw, [frame_path], time_range, frame_idx)
            except Exception as e:
                logger.error(f"帧 {frame_idx} VLM 调用失败: {e}")
                return {
                    "batch_index": frame_idx,
                    "time_range": time_range,
                    "status": "failed",
                    "frame_paths": [frame_path],
                    "frame_observations": [],
                    "overall_activity_summary": "",
                    "fallback_summary": f"分析失败: {e}",
                    "raw_response": "",
                    "error_message": str(e),
                }

    tasks = [analyze_one(fp, i) for i, fp in enumerate(frame_paths)]
    results = []
    with tqdm(total=len(tasks), desc="VLM 分析", unit="帧") as pbar:
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
            pbar.update(1)
    results.sort(key=lambda r: r["batch_index"])
    return results


def analyze_frames(frame_paths: List[str], model: str, base_url: str, batch_size: int = 8, max_concurrency: int = 2, custom_prompt: str = "", timeout: int = 120, mode: str = ANALYZE_MODE_SINGLE, max_edge: int = 1280) -> List[dict]:
    if not frame_paths:
        raise ValueError("frame_paths 不能为空")
    if mode not in {ANALYZE_MODE_SINGLE, ANALYZE_MODE_BATCH}:
        raise ValueError(f"不支持的 mode: {mode}")
    if mode == ANALYZE_MODE_SINGLE:
        logger.info(f"共 {len(frame_paths)} 帧，采用单图逐帧分析（mode=single）")
        batches = [[fp] for fp in frame_paths]
    else:
        batches = [frame_paths[i:i + batch_size] for i in range(0, len(frame_paths), batch_size)]
        logger.info(f"共 {len(frame_paths)} 帧，分为 {len(batches)} 批（batch_size={batch_size}）")
    if max_edge > 0:
        logger.info(f"VLM 输入图片将缩放至最大边长 {max_edge}px")

    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            if mode == ANALYZE_MODE_SINGLE:
                return pool.submit(asyncio.run, _analyze_single_frames_async(frame_paths, model, base_url, custom_prompt, max_concurrency, timeout, max_edge)).result()
            return pool.submit(asyncio.run, _analyze_batches_async(batches, model, base_url, custom_prompt, max_concurrency, timeout, max_edge)).result()
    except RuntimeError:
        if mode == ANALYZE_MODE_SINGLE:
            return asyncio.run(_analyze_single_frames_async(frame_paths, model, base_url, custom_prompt, max_concurrency, timeout, max_edge))
        return asyncio.run(_analyze_batches_async(batches, model, base_url, custom_prompt, max_concurrency, timeout, max_edge))


def build_analysis_artifact(batch_results: List[dict], video_path: str, frame_interval_seconds: float, model: str, base_url: str) -> dict:
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


_SUMMARY_SYSTEM_PROMPT = "你是一位专业的视频内容分析师。请根据各时间段的视频片段描述，为整段视频生成一份简洁、连贯的内容总结。"


def generate_video_summary(batch_results: List[dict], model: Optional[str] = None, base_url: Optional[str] = None, system_prompt: str = _SUMMARY_SYSTEM_PROMPT, timeout: int = 60) -> str:
    segments: List[str] = []
    for batch in batch_results:
        time_range = batch.get("time_range", "")
        text = batch.get("overall_activity_summary") or batch.get("fallback_summary") or ""
        if text:
            segments.append(f"[{time_range}] {text}")
    if not segments:
        return ""
    combined = "\n".join(segments)
    if not model or not base_url:
        return combined

    import httpx as _httpx
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是视频各时间段的内容描述：\n\n{combined}\n\n请基于以上内容，生成一段完整、连贯的视频内容总结（200字以内）。"},
        ],
        "max_tokens": 512,
        "temperature": 0.5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = _httpx.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in (403, 429, 500, 502, 503):
                logger.warning(f"LLM summary {resp.status_code}，body={resp.text[:200]}  (尝试 {attempt+1}/3)")
                if attempt < 2:
                    import time as _time
                    _time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"LLM summary 生成失败: {e}，降级为批次摘要拼接")
            return combined
    return combined
