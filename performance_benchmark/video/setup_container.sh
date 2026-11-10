#!/bin/bash
set -e

# ================================================================
# 起容器（视频测试用）。仓库根目录挂到容器内 /llm，
# 所以 videos_h264 / videos_h265 在容器里是 /llm/videos_h26x。
#
# 用法: bash setup_container.sh [container-name] [weights-dir]
# 示例: bash setup_container.sh vllm-${USER}-nv /DISK0
# ================================================================

REPO_ROOT="$(realpath "$(dirname "$(realpath "$0")")/../..")"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    DEFAULT_NAME="vllm-video-nv"
else
    DEFAULT_NAME="vllm-video-xpu"
fi

CONTAINER_NAME="${1:-${VLLM_CONTAINER:-$DEFAULT_NAME}}"
WEIGHTS_DIR="${2:-${WEIGHTS_DIR:-/home/intel/weights/}}"
INTEL_IMAGE="${INTEL_IMAGE:-vllm/vllm-openai-xpu:v0.32.0}"
NV_IMAGE="${NV_IMAGE:-vllm/vllm-openai:v0.32.0}"

echo "Container : $CONTAINER_NAME"
echo "Weights   : $WEIGHTS_DIR  -> /llm/models"
echo "Repo      : $REPO_ROOT    -> /llm"

bash "$REPO_ROOT/setup_env.sh" \
    --container-name "$CONTAINER_NAME" \
    --weights-dir "$WEIGHTS_DIR" \
    --script-dir "$REPO_ROOT" \
    --intel-image "$INTEL_IMAGE" \
    --nv-image "$NV_IMAGE"

for d in videos_h264 videos_h265; do
    if [ -d "$REPO_ROOT/$d" ]; then
        echo "Video data: /llm/$d ($(ls "$REPO_ROOT/$d" | wc -l) files)"
    else
        echo "WARN: $REPO_ROOT/$d not found — 视频测试需要先准备好数据" >&2
    fi
done

echo ""
echo "下一步（容器名可省略，start_video_server.sh 会自动探测）:"
echo "  VLLM_CONTAINER=$CONTAINER_NAME bash start_video_server.sh 4b 8012 2"
