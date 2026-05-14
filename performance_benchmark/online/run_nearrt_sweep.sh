#!/bin/bash
set -e

# ================================================================
# 准实时场景扫描：一个 Server，测试所有 (分辨率 × in × out × 帧数) 组合
# 指标：E2E < 60s 下的最大帧数 & 最大 TPS
#
# 用法: bash run_nearrt_sweep.sh [model] [device] [quant] [mtp]
# 示例: bash run_nearrt_sweep.sh 4b "" fp8
#        bash run_nearrt_sweep.sh 4b 4 fp8      # 只用 GPU 4
# ================================================================
MODEL="${1:-4b}"
DEVICE="${2:-}"
QUANT="${3:-fp8}"
MTP="${4:-off}"
FIXED_BATCH=1
E2E_LIMIT=30   # seconds

export VLLM_NV_CONTAINER="${VLLM_NV_CONTAINER:-vllm-nv-container}"
export VLLM_XPU_CONTAINER="${VLLM_XPU_CONTAINER:-lsv-container-b8}"

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

PORT=8006

echo "========================================"
echo "Near-RT Sweep started at: $RUN_START"
echo "Model=$MODEL  device=${DEVICE:-all}  quant=$QUANT  mtp=$MTP  E2E_LIMIT=${E2E_LIMIT}s"
echo "Matrix: res=[720p,1080p] in=[512,1024] out=[128,512,1024] frames=[4,6,8,10]"
echo "========================================"

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
            PF='/tmp/vllm_server_${PORT}.pid'
            if [ -f \"\$PF\" ]; then
                kill \"\$(cat \$PF)\" 2>/dev/null || true
                rm -f \"\$PF\"
                echo 'Server stopped (PID file).'
            else
                pkill -f 'vllm serve' 2>/dev/null || true
                echo 'Server stopped (pkill).'
            fi
        " 2>/dev/null || true
    else
        PF="/tmp/vllm_server_${PORT}.pid"
        if [ -f "$PF" ]; then
            kill "$(cat "$PF")" 2>/dev/null || true
            rm -f "$PF"
        else
            pkill -f "vllm serve" 2>/dev/null || true
        fi
        echo "Server stopped."
    fi
}

trap 'echo ""; echo "Interrupted — stopping server..."; stop_server; exit 1' INT TERM

# ----------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------
for res in "1280 720" "1920 1080"; do
    w=${res% *}; h=${res#* }

    for input_len in 512 1024; do
        for output_len in 128 1024; do

            echo ""
            echo "================================================================"
            echo "Resolution: ${w}x${h}  input_len=${input_len}  output_len=${output_len}"
            echo "================================================================"

            MAX_FRAMES_REACHED=0
            MAX_TPS_REACHED=0

            for imgs in 1 4 6 8 10 12; do
                echo ""
                echo "--- ${MODEL} ${w}x${h} frames=${imgs} in=${input_len} out=${output_len} quant=${QUANT} ---"

                # arg11="1" → KEEP_SERVER_UP=1: server starts on first call, reused afterwards
                if ! bash "$SCRIPT_DIR/vllm_random_benchmark_server.sh" \
                    "$MODEL" "$w" "$h" "$imgs" "$MTP" "$QUANT" "$DEVICE" \
                    "$output_len" "$input_len" "$FIXED_BATCH" "1" "10" "$RUN_START_FILE"; then
                    echo "  ERROR: benchmark failed (OOM / Bad Request) — aborting sweep"
                    stop_server
                    exit 1
                fi

                # Find the log file for this exact combo by pattern
                LATEST_LOG=$(find "$SCRIPT_DIR/$MODEL_DIR" \
                    -name "*_${w}x${h}_f${imgs}_in${input_len}_out${output_len}_client.log" \
                    2>/dev/null | sort | tail -1)
                # Fallback: newest client log newer than this script
                if [ -z "$LATEST_LOG" ]; then
                    LATEST_LOG=$(find "$SCRIPT_DIR/$MODEL_DIR" -name "*_client.log" \
                        -newer "$SCRIPT_DIR/run_nearrt_sweep.sh" 2>/dev/null | sort | tail -1)
                fi

                if [ -z "$LATEST_LOG" ]; then
                    echo "  WARNING: could not find log file, skipping E2E check"
                    continue
                fi

                E2E=$(grep 'Benchmark duration (s):' "$LATEST_LOG" | tail -1 | awk '{print $NF}')
                TPS=$(grep 'Output token throughput (tok/s):' "$LATEST_LOG" | tail -1 | awk '{print $NF}')

                echo "  frames=${imgs}  E2E=${E2E}s  TPS=${TPS} tok/s"

                OVER=$(awk -v e="${E2E:-0}" -v lim="$E2E_LIMIT" 'BEGIN { print (e > lim) ? 1 : 0 }')
                if [ "$OVER" = "1" ]; then
                    echo "  E2E ${E2E}s > ${E2E_LIMIT}s limit — stopping frames sweep for this config"
                    break
                fi

                MAX_FRAMES_REACHED=$imgs
                MAX_TPS_REACHED=$TPS
            done

            echo ""
            echo "  >> Result: ${w}x${h} in=${input_len} out=${output_len}"
            echo "     Max frames @ E2E<${E2E_LIMIT}s : $MAX_FRAMES_REACHED"
            echo "     TPS at max frames              : $MAX_TPS_REACHED tok/s"
        done
    done
done

# ----------------------------------------------------------------
# Stop server
# ----------------------------------------------------------------
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

# Parse all client logs generated in this sweep → CSV
echo "Parsing logs..."
find "$SCRIPT_DIR/$MODEL_DIR" -name "*_client.log" \
    -newer "$SCRIPT_DIR/run_nearrt_sweep.sh" 2>/dev/null | sort | while read -r log; do
    echo "  Parsing: $log"
    python3 "$SCRIPT_DIR/parse_log.py" "$log"
done
echo "All done."
