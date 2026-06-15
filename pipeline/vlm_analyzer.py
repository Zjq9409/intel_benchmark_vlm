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


def _build_messages(frame_paths: List[str], prompt: str, max_edge: int = 0) -> List[dict]:
    content: List[dict] = [{"type": "text", "text": prompt}]
    for fp in frame_paths:
        b64 = _encode_image_b64(fp, max_edge=max_edge)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [
        {"role": "system", "content": "You are a JSON-only assistant. Output ONLY the raw JSON object, no markdown fences, no reasoning, no extra text."},
        {"role": "user", "content": content},
    ]


DEFAULT_PROMPT_TEMPLATE = """
你将看到 {frame_count} 张按时间顺序排列的视频帧。请用简体中文分析画面内容，并只输出一个 JSON 对象。
JSON 结构如下：
{{
  "frame_observations": [{{"timestamp": "HH:MM:SS,mmm", "observation": "该帧画面的中文描述"}}],
  "overall_activity_summary": "整个片段的一句话中文总结"
}}
要求：所有描述必须使用简体中文；frame_observations 必须正好包含 {frame_count} 项；只输出原始 JSON，不要 markdown 代码块，不要额外说明文字。
""".strip()


async def _call_vlm_api(messages, model, base_url, timeout=120):
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 4096, "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False}}
    url = base_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _clean_json_response(text: str) -> str:
    """从模型输出中稳健地提取 JSON 对象。

    模型可能先输出思考文本（含 </think> 标签、markdown 围栏，文本中可能带 { }），
    再输出真正的 JSON。这里去掉 </think> 之前内容和围栏后，
    从每个 '{' 起点尝试 raw_decode，选出能成功解析且跨度最大的 JSON 块。
    """
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


async def _analyze_batches_async(batches, model, base_url, custom_prompt, max_concurrency, timeout):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze_one(batch_files, batch_idx):
        time_range = _time_range_from_batch(batch_files)
        prompt = (custom_prompt or DEFAULT_PROMPT_TEMPLATE).format(frame_count=len(batch_files))
        messages = _build_messages(batch_files, prompt)
        async with semaphore:
            try:
                raw = await _call_vlm_api(messages, model, base_url, timeout)
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
                _analyze_batches_async(batches, model, base_url, custom_prompt, max_concurrency, timeout),
            ).result()
    except RuntimeError:
        return asyncio.run(
            _analyze_batches_async(batches, model, base_url, custom_prompt, max_concurrency, timeout)
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
    headers = {"Content-Type": "application/json"}
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


# ---------------------------------------------------------------------------
# 视频整体 Summary 生成
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = (
    "你是一位专业的视频内容分析师。"
    "请根据各时间段的视频片段描述，为整段视频生成一份简洁、连贯的内容总结。"
)


def generate_video_summary(
    batch_results,
    model=None,
    base_url=None,
    system_prompt=_SUMMARY_SYSTEM_PROMPT,
    timeout=60,
):
    """
    将各批次 overall_activity_summary 汇总，调用 LLM 生成全视频 summary。
    """
    segments = []
    for batch in batch_results:
        time_range = batch.get("time_range", "")
        text = batch.get("overall_activity_summary") or batch.get("fallback_summary") or ""
        if text:
            segments.append(f"[{time_range}] {text}")

    if not segments:
        return ""

    combined = "\n".join(segments)

    if not (model and base_url):
        logger.info("未配置 LLM，跳过 summary 生成")
        return ""

    import httpx as _httpx

    user_message = (
        f"以下是视频各时间段的内容描述：\n\n{combined}\n\n"
        "请基于以上内容，生成一段完整、连贯的视频内容总结（200字以内）。"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 512,
        "temperature": 0.5,
    }
    try:
        resp = _httpx.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"].strip()
        logger.info("视频 summary 生成成功")
        return summary
    except Exception as e:
        logger.warning(f"LLM summary 生成失败: {e}")
        return ""
