#!/bin/bash
set -e

# 用户在 docker 组时直接用 docker，否则回退 sudo（远端机器可能没有免密 sudo）
DOCKER="docker"
docker info &>/dev/null || DOCKER="sudo docker"
SERVER_PORT="${SERVER_PORT:-8012}"

# ================================================================
# 启动支持 video 输入的 vLLM server
#
# 用法: bash start_video_server.sh [model] [tp] [device]
#   端口由文件开头的 SERVER_PORT 设置，默认 8012
#   参数都可省略，默认模型 4b / 模型默认 TP / 所有卡
#
# model 可以是:
#   bash start_video_server.sh 4b                     # ../online/model_config.sh 的预置 key
#   bash start_video_server.sh Qwen3-VL-8B-Instruct   # /llm/models 下的目录名
#   bash start_video_server.sh /llm/models/my-model   # 容器内绝对路径
#   bash start_video_server.sh list                   # 列出 /llm/models 下可用模型
#
# 解码后端 (VIDEO_BACKEND, 默认 opencv):
#   opencv          CPU 解码，兼容性最好，高并发时 CPU 容易打满（Intel XPU 只能用这个）
#   pynvvideocodec  [NVIDIA only] NVDEC 硬解，帧留在显存，建议配 CUDA MPS + MM_IPC_GPU_MEM_GB
#   torchcodec      [NVIDIA only] torchcodec 解码，直接输出 GPU tensor
#   deepstream      [NVIDIA only] DeepStream/GStreamer 流式硬解，需 vllm[deepstream]
#   VIDEO_BACKEND=pynvvideocodec HW_DECODERS=2 MM_IPC_GPU_MEM_GB=4 bash start_video_server.sh 4b
# ================================================================

# ----------------------------------------------------------------
# Bare-metal guard: 不在容器里就 re-exec 进容器
# ----------------------------------------------------------------
if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    # 容器名优先级: VLLM_CONTAINER > VLLM_NV/XPU_CONTAINER > 自动探测 > vllm-video-{nv,xpu}
    CONTAINER_NAME="${VLLM_CONTAINER:-}"
    if [ -z "$CONTAINER_NAME" ]; then
        if nvidia-smi &>/dev/null; then
            CONTAINER_NAME="${VLLM_NV_CONTAINER:-}"
            DEFAULT_NAME="vllm-video-nv"
        else
            CONTAINER_NAME="${VLLM_XPU_CONTAINER:-}"
            DEFAULT_NAME="vllm-video-xpu"
        fi
    fi
    if [ -z "$CONTAINER_NAME" ]; then
        CONTAINER_NAME=$($DOCKER ps --filter "name=vllm" --format '{{.Names}}' 2>/dev/null | head -1)
        [ -n "$CONTAINER_NAME" ] && echo "Auto-detected container: $CONTAINER_NAME"
    fi
    [ -z "$CONTAINER_NAME" ] && CONTAINER_NAME="$DEFAULT_NAME"

    SELF="$(realpath "$0")"
    REPO_BASE="$(dirname "$(dirname "$(dirname "$SELF")")")"
    [[ "$SELF" == "$REPO_BASE"* ]] && SELF="/llm${SELF#$REPO_BASE}"

    if ! $DOCKER inspect --format '{{.State.Status}}' "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "ERROR: Container '$CONTAINER_NAME' does not exist."
        echo "  先运行 setup_container.sh，或用 VLLM_CONTAINER=<name> 指定容器名。"
        exit 1
    fi
    exec $DOCKER exec -it \
        -e SERVER_PORT="$SERVER_PORT" \
        -e VIDEO_LIMIT="${VIDEO_LIMIT:-}" \
        -e VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-}" \
        -e VIDEO_BACKEND="${VIDEO_BACKEND:-}" \
        -e VIDEO_IO_KWARGS="${VIDEO_IO_KWARGS:-}" \
        -e HW_DECODERS="${HW_DECODERS:-}" \
        -e DEEPSTREAM_FPS="${DEEPSTREAM_FPS:-}" \
        -e MM_IPC_GPU_MEM_GB="${MM_IPC_GPU_MEM_GB:-}" \
        -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-}" \
        -e GPU_MEM_UTIL="${GPU_MEM_UTIL:-}" \
        -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-}" \
        -e QUANT="${QUANT:-}" \
        -e MODEL_PATH="${MODEL_PATH:-}" \
        "$CONTAINER_NAME" bash "$SELF" "$@"
fi

cd "$(dirname "$(realpath "$0")")"

MODEL_SELECT="${1:-4b}"
TP_OVERRIDE="${2:-}"
DEVICE="${3:-}"

VIDEO_LIMIT="${VIDEO_LIMIT:-1}"          # 每请求最多几个视频
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-16}"  # 服务端解码采样帧数
# 解码后端: opencv(CPU 默认) | pynvvideocodec(NVDEC) | torchcodec | deepstream
VIDEO_BACKEND="${VIDEO_BACKEND:-opencv}"
HW_DECODERS="${HW_DECODERS:-2}"          # pynvvideocodec 使用的硬件解码器数量
DEEPSTREAM_FPS="${DEEPSTREAM_FPS:-1}"    # deepstream 抽帧 fps
MM_IPC_GPU_MEM_GB="${MM_IPC_GPU_MEM_GB:-}"  # NVDEC 路径建议预留的显存 (GB)
VIDEO_IO_KWARGS="${VIDEO_IO_KWARGS:-}"   # 直接给原始 JSON，设了就忽略上面的拼装
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
QUANT="${QUANT:-fp8}"
# 视频素材所在目录（容器内路径），需在 --allowed-local-media-path 之下
MEDIA_PATH="${MEDIA_PATH:-/llm}"

MODELS_ROOT="${MODELS_ROOT:-/llm/models}"

list_models() {
    echo "Available models under ${MODELS_ROOT}:"
    find "$MODELS_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '  %f\n' 2>/dev/null | sort
}

if [ "$MODEL_SELECT" = "list" ] || [ "$MODEL_SELECT" = "--list" ]; then
    list_models
    exit 0
fi

# 非预置 key 时，允许直接写 $MODELS_ROOT 下的目录名
if [[ "$MODEL_SELECT" != /* ]] && [ -z "${MODEL_PATH:-}" ] \
   && [ -d "$MODELS_ROOT/$MODEL_SELECT" ]; then
    MODEL_SELECT="$MODELS_ROOT/$MODEL_SELECT"
fi

# shellcheck source=../online/model_config.sh
source "$(dirname "$(realpath "$0")")/../online/model_config.sh"
if ! resolve_model "$MODEL_SELECT"; then
    list_models
    exit 1
fi
TP="${TP_OVERRIDE:-${TP:-1}}"

mkdir -p "$SERVER_MODEL_NAME"
chmod 777 "$SERVER_MODEL_NAME" 2>/dev/null || true   # 容器内 root 建的目录，裸机客户端也要能写
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 \
    | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
QUANT_TAG=""
[ "$QUANT" != "none" ] && QUANT_TAG="_${QUANT}"
SERVER_LOG="${SERVER_MODEL_NAME}/$(date +%Y%m%d_%H%M%S)_video${QUANT_TAG}_${GPU_TYPE}_server.log"

if [ -n "$DEVICE" ]; then
    if [ "$GPU_TYPE" = "XPU" ]; then
        export ZE_AFFINITY_MASK="$DEVICE"
    else
        export CUDA_VISIBLE_DEVICES="$DEVICE"
    fi
fi

# NVDEC / DeepStream 依赖 NVIDIA 硬件解码器
case "$VIDEO_BACKEND" in
    pynvvideocodec|deepstream|torchcodec)
        if [ "$GPU_TYPE" = "XPU" ]; then
            echo "ERROR: VIDEO_BACKEND='$VIDEO_BACKEND' 需要 NVIDIA GPU，当前是 XPU。" >&2
            echo "  Intel 平台请使用 VIDEO_BACKEND=opencv (默认)。" >&2
            exit 1
        fi
        ;;
esac

if [ -z "$VIDEO_IO_KWARGS" ]; then
    case "$VIDEO_BACKEND" in
        pynvvideocodec)
            VIDEO_IO_KWARGS="{\"video\": {\"backend\": \"pynvvideocodec\", \"min_frames\": ${VIDEO_NUM_FRAMES}, \"max_frames\": ${VIDEO_NUM_FRAMES}, \"hw_decoders\": ${HW_DECODERS}}}"
            ;;
        torchcodec)
            VIDEO_IO_KWARGS="{\"video\": {\"backend\": \"torchcodec\", \"device\": \"cuda\", \"num_frames\": ${VIDEO_NUM_FRAMES}}}"
            ;;
        deepstream)
            VIDEO_IO_KWARGS="{\"video\": {\"backend\": \"deepstream\", \"fps\": ${DEEPSTREAM_FPS}}}"
            ;;
        opencv|"")
            VIDEO_IO_KWARGS="{\"video\": {\"backend\": \"opencv\", \"num_frames\": ${VIDEO_NUM_FRAMES}}}"
            ;;
        *)
            echo "ERROR: unknown VIDEO_BACKEND='$VIDEO_BACKEND'" >&2
            echo "  可选: opencv | pynvvideocodec | torchcodec | deepstream" >&2
            exit 1
            ;;
    esac
fi

VLLM_SERVER_ARGS=(
    --model "$SERVER_MODEL"
    --served-model-name "$SERVER_MODEL_NAME"
    --dtype float16
    --host 0.0.0.0
    --port "$SERVER_PORT"
    --trust-remote-code
    --no-enable-prefix-caching
    --gpu-memory-util "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --mm-processor-cache-gb 0
    --limit-mm-per-prompt "{\"image\": 0, \"video\": ${VIDEO_LIMIT}}"
    --media-io-kwargs "$VIDEO_IO_KWARGS"
    --allowed-local-media-path "$MEDIA_PATH"
    -tp "$TP"
)
[ -n "$MAX_NUM_SEQS" ] && VLLM_SERVER_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
[ "$QUANT" != "none" ] && VLLM_SERVER_ARGS+=(--quantization "$QUANT")
[ -n "$MM_IPC_GPU_MEM_GB" ] && VLLM_SERVER_ARGS+=(--mm-ipc-gpu-memory-gb "$MM_IPC_GPU_MEM_GB")

export PYTORCH_ALLOC_CONF="expandable_segments:True"
if [ "$GPU_TYPE" = "XPU" ]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1
    VLLM_SERVER_ARGS+=(--block-size 64)
else
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export NCCL_P2P_LEVEL=SYS
fi

echo "---------------------------------------------------"
echo "Model:            $SERVER_MODEL"
echo "Served name:      $SERVER_MODEL_NAME"
echo "Port / TP:        $SERVER_PORT / $TP"
echo "Videos/request:   $VIDEO_LIMIT  (num_frames=$VIDEO_NUM_FRAMES)"
echo "Decode backend:   $VIDEO_BACKEND"
echo "media-io-kwargs:  $VIDEO_IO_KWARGS"
echo "Allowed media:    $MEDIA_PATH"
echo "Server log:       $SERVER_LOG"
echo "---------------------------------------------------"

PID_FILE="/tmp/vllm_video_server_${SERVER_PORT}.pid"
if curl -sf "http://localhost:${SERVER_PORT}/v1/models" 2>/dev/null | grep -q "$SERVER_MODEL_NAME"; then
    echo "Server already running on port $SERVER_PORT, nothing to do."
    exit 0
fi

nohup vllm serve "${VLLM_SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "Server PID: $(cat "$PID_FILE")"

for i in $(seq 1 120); do
    if curl -s "http://localhost:${SERVER_PORT}/v1/models" | grep -q "$SERVER_MODEL_NAME"; then
        echo "Server ready after $((i*5))s."
        exit 0
    fi
    sleep 5
    echo "Waiting for server... ($i/120)"
done

echo "Server failed to start. Last 30 log lines:"
tail -n 30 "$SERVER_LOG"
kill "$(cat "$PID_FILE")" 2>/dev/null || true
rm -f "$PID_FILE"
exit 1
