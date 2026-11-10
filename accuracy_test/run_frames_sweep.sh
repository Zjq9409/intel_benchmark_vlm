#!/bin/bash
set -e

# ================================================================
# 帧数扫描测试：固定 batch，逐增 images/request
# 测试矩阵: 分辨率 × input_len × output_len × frames
# 指标：E2E < 60s 下的最大帧数 & 最大 TPS
#
# 用法: bash run_frames_sweep.sh [model] [batch] [device] [quant]
# 示例: bash run_frames_sweep.sh 4b 1 "" fp8
# ================================================================
MODEL="${1:-4b}"
FIXED_BATCH="${2:-1}"
DEVICE="${3:-}"
QUANT="${4:-fp8}"
E2E_LIMIT=60   # seconds

export VLLM_NV_CONTAINER="${VLLM_NV_CONTAINER:-vllm-nv-container}"
export VLLM_XPU_CONTAINER="${VLLM_XPU_CONTAINER:-lsv-container-b8.1}"

RUN_START=$(date "+%Y-%m-%d %H:%M:%S")
RUN_START_TS=$(date +%s)
RUN_START_FILE=$(date "+%Y%m%d_%H%M%S")
export SWEEP_TIMESTAMP="$RUN_START_FILE"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if [ "$MODEL" = "4b" ]; then
    MODEL_DIR="Qwen3-VL-4B-Instruct"
elif [ "$MODEL" = "32b" ]; then
    MODEL_DIR="Qwen3-VL-32B-Instruct"
else
    MODEL_DIR="Qwen3-VL-30B-A3B-Instruct"
fi
_TS_REF=$(mktemp)  # 用于 find -newer 的时间戳参考文件


# ----------------------------------------------------------------
# Helper: stop vllm server (bare metal -> container, or in-container)
# ----------------------------------------------------------------
stop_server() {
    if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
        if nvidia-smi &>/dev/null; then
            C="${VLLM_NV_CONTAINER:-vllm-nv-container}"
        else
            C="${VLLM_XPU_CONTAINER:-lsv-container-b8}"
        fi
        echo "Stopping vllm server in container $C..."
        sudo docker exec "$C" bash -c "
            PF='/tmp/vllm_server_8006.pid'
            if [ -f \"\$PF\" ]; then
                kill \"\$(cat \$PF)\" 2>/dev/null || true
                rm -f \"\$PF\"
            else
                pkill -f 'vllm serve' 2>/dev/null || true
            fi
            echo 'Server stopped.'
        " 2>/dev/null || true
    else
        PF="/tmp/vllm_server_8006.pid"
        [ -f "$PF" ] && kill "$(cat "$PF")" 2>/dev/null || true && rm -f "$PF" || pkill -f "vllm serve" 2>/dev/null || true
        echo "Server stopped."
    fi
}

trap 'echo "Interrupted — stopping server..."; stop_server; exit 1' INT TERM

echo "========================================"
echo "Frames Sweep started at: $RUN_START"
echo "Model=$MODEL  batch=$FIXED_BATCH  quant=$QUANT  E2E_LIMIT=${E2E_LIMIT}s"
echo "========================================"

for input_len in 512 1024; do
    for output_len in 128 1024; do
        for res in "1280 720" "1920 1080"; do
            w=${res% *}; h=${res#* }

            echo ""
            echo "================================================================"
            echo "Resolution: ${w}x${h}  input_len=${input_len}  output_len=${output_len}"
            echo "================================================================"

            MAX_FRAMES_REACHED=0
            MAX_TPS_REACHED=0

            for imgs in 1 2 4 6 8 10 12 16 20 24 28 32 40 48 56 64; do
                echo ""
                echo "--- ${MODEL} ${w}x${h} frames=${imgs} batch=${FIXED_BATCH} quant=${QUANT} in=${input_len} out=${output_len} ---"

                if ! bash "$SCRIPT_DIR/vllm_random_benchmark_server.sh" \
                    "$MODEL" "$w" "$h" "$imgs" off "$QUANT" "$DEVICE" "$output_len" "$input_len" "$FIXED_BATCH" "1" "64" "$RUN_START_FILE"; then
                    echo "  ERROR: benchmark failed for frames=${imgs} (OOM or Bad Request) — stopping sweep"
                    break
                fi

                LATEST_LOG=$(find "$SCRIPT_DIR/$MODEL_DIR" -name "*_client.log" \
                    -newer "$SCRIPT_DIR/run_frames_sweep.sh" 2>/dev/null | sort | tail -1)

                if [ -z "$LATEST_LOG" ]; then
                    echo "  WARNING: could not find log file, skipping check"
                    continue
                fi

                E2E=$(grep 'Benchmark duration (s):' "$LATEST_LOG" | tail -1 | awk '{print $NF}')
                TPS=$(grep 'Output token throughput (tok/s):' "$LATEST_LOG" | tail -1 | awk '{print $NF}')

                echo "  frames=${imgs}  E2E=${E2E}s  TPS=${TPS} tok/s"

                OVER=$(awk -v e="${E2E:-0}" -v lim="$E2E_LIMIT" 'BEGIN { print (e > lim) ? 1 : 0 }')
                if [ "$OVER" = "1" ]; then
                    echo "  E2E ${E2E}s > ${E2E_LIMIT}s limit — stopping frames sweep"
                    break
                fi

                MAX_FRAMES_REACHED=$imgs
                MAX_TPS_REACHED=$TPS
            done
rm -f "$_TS_REF"

            echo ""
            echo "  >> Result: ${w}x${h} in=${input_len} out=${output_len} batch=${FIXED_BATCH}"
            echo "     Max frames @ E2E<${E2E_LIMIT}s : $MAX_FRAMES_REACHED"
            echo "     TPS at max frames              : $MAX_TPS_REACHED tok/s"
        done
    done
done

stop_server

echo ""
echo "========================================"
RUN_END=$(date "+%Y-%m-%d %H:%M:%S")
RUN_END_TS=$(date +%s)
ELAPSED=$(( RUN_END_TS - RUN_START_TS ))
printf "Run started at:  %s\n" "$RUN_START"
printf "Run finished at: %s\n" "$RUN_END"
printf "Total elapsed:   %02dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
echo "========================================"

# 解析本次运行生成的日志 → CSV
echo "Parsing logs..."
find "$SCRIPT_DIR/$MODEL_DIR" -name "*_client.log" -newer "$_TS_REF" 2>/dev/null | sort | while read -r log; do
    echo "  Parsing: $log"
    python3 "$SCRIPT_DIR/parse_log.py" "$log"
done
