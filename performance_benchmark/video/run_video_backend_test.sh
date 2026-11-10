#!/usr/bin/env bash
set -e

# Compare PyNvVideoCodec and OpenCV using one command. Edit these fixed settings as needed.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="Qwen3.5-2B"
TP=1
# DEVICE=1
VIDEO_NUM_FRAMES=32
QUANT=fp8
HW_DECODERS=2
MM_IPC_GPU_MEM_GB=4
VIDEO_DIRS="/llm/performance_benchmark/video/videos_h264_30fps /llm/performance_benchmark/video/videos_h265_30fps"
OUTPUT_LEN=32
CONCURRENCY_LIST="1 4"

# Use eight requests per worker at the highest concurrency, giving multiple waves.
MAX_CONCURRENCY=0
for concurrency in $CONCURRENCY_LIST; do
    if ((concurrency > MAX_CONCURRENCY)); then
        MAX_CONCURRENCY="$concurrency"
    fi
done
NUM_PROMPTS=$((MAX_CONCURRENCY * 8))

DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
CONTAINER_NAME="${VLLM_CONTAINER:-$("${DOCKER[@]}" ps --filter 'name=vllm' --format '{{.Names}}' | head -n 1)}"
CURRENT_PORT=""

stop_current_server() {
    if [[ -n "$CURRENT_PORT" ]]; then
        "${DOCKER[@]}" exec "$CONTAINER_NAME" bash -lc \
            'pid_file="$1"; if [[ -s "$pid_file" ]]; then kill "$(cat "$pid_file")" 2>/dev/null || true; rm -f "$pid_file"; fi' \
            _ "/tmp/vllm_video_server_${CURRENT_PORT}.pid" || true
        CURRENT_PORT=""
    fi
}
trap stop_current_server EXIT

for BACKEND in pynvvideocodec opencv; do
    if [[ "$BACKEND" == "pynvvideocodec" ]]; then
        SERVER_PORT=8013
    else
        SERVER_PORT=8014
    fi

    if curl -sf -m 2 "http://localhost:${SERVER_PORT}/v1/models" >/dev/null; then
        echo "ERROR: port ${SERVER_PORT} is occupied. Stop its service before testing." >&2
        exit 1
    fi

    export SERVER_PORT VIDEO_BACKEND="$BACKEND" VIDEO_NUM_FRAMES QUANT HW_DECODERS
    if [[ "$BACKEND" == "pynvvideocodec" ]]; then
        export MM_IPC_GPU_MEM_GB
    else
        unset MM_IPC_GPU_MEM_GB
    fi

    echo "=== Testing $BACKEND on port $SERVER_PORT ==="
    bash "$SCRIPT_DIR/start_video_server.sh" "$MODEL" "$TP" "$DEVICE"
    CURRENT_PORT="$SERVER_PORT"

    VIDEO_DIRS="$VIDEO_DIRS" OUTPUT_LEN="$OUTPUT_LEN" \
        bash "$SCRIPT_DIR/run_video_bench_serve.sh" \
        "$MODEL" "$SERVER_PORT" "$CONCURRENCY_LIST" "$NUM_PROMPTS"

    stop_current_server
done
