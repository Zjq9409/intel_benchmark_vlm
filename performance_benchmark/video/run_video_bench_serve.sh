#!/bin/bash
set -e

# ================================================================
# 用官方 vllm bench serve 跑本地视频文件的在线性能测试
#
# 原理: vLLM 的 sharegpt dataset 支持条目里带 "video" 字段，
#       process_video() 会把本地路径转成 file://，所以只要按
#       ShareGPT 格式把本地 mp4 包一层 JSON 就能直接喂给 vllm bench serve。
#       服务端需带 --allowed-local-media-path（start_video_server.sh 已带）。
#
# 用法: bash run_video_bench_serve.sh [model] [port] [concurrency_list] [num_prompts]
#       model 留空自动从 /v1/models 读取
# 示例: bash run_video_bench_serve.sh
#       bash run_video_bench_serve.sh "" 8012 "1 4 8" 16
# ================================================================

if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    CONTAINER_NAME="${VLLM_CONTAINER:-}"
    if [ -z "$CONTAINER_NAME" ]; then
        CONTAINER_NAME=$(sudo docker ps --filter "name=vllm" --format '{{.Names}}' 2>/dev/null | head -1)
    fi
    if [ -z "$CONTAINER_NAME" ]; then
        echo "ERROR: 找不到运行中的 vllm 容器，用 VLLM_CONTAINER=<name> 指定。" >&2
        exit 1
    fi
    SELF="$(realpath "$0")"
    REPO_BASE="$(dirname "$(dirname "$(dirname "$SELF")")")"
    [[ "$SELF" == "$REPO_BASE"* ]] && SELF="/llm${SELF#$REPO_BASE}"
    exec sudo docker exec -i \
        -e HOST="${HOST:-}" -e VIDEO_DIRS="${VIDEO_DIRS:-}" \
        -e PROMPT="${PROMPT:-}" -e OUTPUT_LEN="${OUTPUT_LEN:-}" \
        -e TOKENIZER="${TOKENIZER:-}" \
        -e EXTRA_ARGS="${EXTRA_ARGS:-}" \
        "$CONTAINER_NAME" bash "$SELF" "$@"
fi

cd "$(dirname "$(realpath "$0")")"

MODEL="${1:-}"
PORT="${2:-8012}"
CONCURRENCY_LIST="${3:-1 4 8}"
NUM_PROMPTS="${4:-16}"

HOST="${HOST:-localhost}"
VIDEO_DIRS="${VIDEO_DIRS:-/llm/videos_h264 /llm/videos_h265}"
PROMPT="${PROMPT:-Describe what happens in this video in detail.}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

MODELS_JSON=$(curl -sf -m 10 "http://${HOST}:${PORT}/v1/models") || {
    echo "ERROR: server not ready at http://${HOST}:${PORT}，先运行 start_video_server.sh" >&2
    exit 1
}
if [ -z "$MODEL" ]; then
    MODEL=$(echo "$MODELS_JSON" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
fi
[ -z "$MODEL" ] && { echo "ERROR: cannot detect model" >&2; exit 1; }

# served model name 不是本地路径时，tokenizer 要显式指到权重目录
TOKENIZER="${TOKENIZER:-}"
if [ -z "$TOKENIZER" ]; then
    if [[ "$MODEL" == /* ]] && [ -d "$MODEL" ]; then
        TOKENIZER="$MODEL"
    elif [ -d "${MODELS_ROOT:-/llm/models}/$MODEL" ]; then
        TOKENIZER="${MODELS_ROOT:-/llm/models}/$MODEL"
    else
        TOKENIZER="$MODEL"
    fi
fi

# ----------------------------------------------------------------
# 把本地视频包装成 ShareGPT 格式的 JSON
# ----------------------------------------------------------------
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR:-$(pwd)/${MODEL//\//_}}"
mkdir -p "$LOG_DIR"; chmod 777 "$LOG_DIR" 2>/dev/null || true
RESULT_DIR="$LOG_DIR/${RUN_TS}_bench"
mkdir -p "$RESULT_DIR"; chmod 777 "$RESULT_DIR" 2>/dev/null || true
DATASET_JSON="$RESULT_DIR/local_videos_sharegpt.json"

python3 - "$DATASET_JSON" "$PROMPT" $VIDEO_DIRS <<'PY'
import json, os, sys
out, prompt, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
vids = []
for d in dirs:
    if os.path.isfile(d):
        vids.append(d)
    elif os.path.isdir(d):
        vids += [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.lower().endswith((".mp4", ".mkv", ".webm"))]
if not vids:
    sys.exit(f"ERROR: no video files found in {dirs}")
# sharegpt loader 要求 conversations >= 2 轮；第二轮只用于推断默认输出长度
data = [{"id": f"video_{i}", "video": v,
         "conversations": [{"from": "human", "value": prompt},
                           {"from": "gpt", "value": "placeholder answer " * 32}]}
        for i, v in enumerate(vids)]
json.dump(data, open(out, "w"))
print(f"{len(data)} videos -> {out}")
PY

LOG_FILE="$RESULT_DIR/bench_serve.log"

for conc in $CONCURRENCY_LIST; do
    echo "" | tee -a "$LOG_FILE"
    echo "########## max-concurrency=$conc ##########" | tee -a "$LOG_FILE"
    vllm bench serve \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --host "$HOST" --port "$PORT" \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_JSON" \
        --sharegpt-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$conc" \
        --ignore-eos \
        --save-result --result-dir "$RESULT_DIR" \
        $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $LOG_FILE" | tee -a "$LOG_FILE"
