#!/bin/bash
set -e

# ================================================================
# 准实时场景扫描：一个 Server，测试所有 (分辨率 × in × out × 帧数) 组合
# 指标：E2E < E2E_LIMIT 下的最大帧数 & 最大 TPS
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
MAX_IMGS=24    # max frames in sweep (also server MM limit)

export VLLM_NV_CONTAINER="${VLLM_NV_CONTAINER:-vllm-nv-container}"
export VLLM_XPU_CONTAINER="${VLLM_XPU_CONTAINER:-lsv-container-b8}"

RUN_START=$(date "+%Y-%m-%d %H:%M:%S")
RUN_START_TS=$(date +%s)
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

if [ "$MODEL" = "4b" ]; then
    MODEL_DIR="Qwen3-VL-4B-Instruct"
    MODEL_LABEL="Qwen3-VL-4B"
elif [ "$MODEL" = "32b" ]; then
    MODEL_DIR="Qwen3-VL-32B-Instruct"
    MODEL_LABEL="Qwen3-VL-32B"
else
    MODEL_DIR="Qwen3-VL-30B-A3B-Instruct"
    MODEL_LABEL="Qwen3-VL-30B-A3B"
fi

MTP_LABEL=$([ "$MTP" = "on" ] && echo "mtp" || echo "nomtp")

# Detect GPU type on bare metal
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 \
    | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"

RUN_START_FILE=$(date "+%Y%m%d_%H%M%S")
SUMMARY_CSV="$SCRIPT_DIR/nearrt_summary_${RUN_START_FILE}_${MODEL}_${QUANT}_${GPU_TYPE}.csv"
echo "Device, Model, Precision, Image Size, Input Len, Output Len, Batch, Max Frames (E2E<${E2E_LIMIT}s), E2E(s), TPS(tok/s)" > "$SUMMARY_CSV"
_sweep_ref=$(mktemp)

PORT=8006

echo "========================================"
echo "Near-RT Sweep started at: $RUN_START"
echo "Model=$MODEL  device=${DEVICE:-all}  quant=$QUANT  mtp=$MTP  E2E_LIMIT=${E2E_LIMIT}s"
echo "Matrix: res=[720p,1080p] in=[512,1024] out=[128,512,1024] frames=[4,6,8,10,12,16,20,24]"
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

trap 'echo ""; echo "Interrupted — stopping server..."; stop_server; rm -f "$_sweep_ref"; exit 1' INT TERM

# ----------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------
for res in "1280 720" "1920 1080"; do
    w=${res% *}; h=${res#* }

    for input_len in 512 1024; do
        for output_len in 128 512 1024; do

            # One shared timestamp per (res x in x out) combo -> one shared log file
            COMBO_TS=$(date "+%Y%m%d_%H%M%S")
            COMBO_LOG="$SCRIPT_DIR/$MODEL_DIR/${COMBO_TS}_${MODEL}_${QUANT}_${MTP_LABEL}_32768_${GPU_TYPE}_client.log"

            echo ""
            echo "================================================================"
            echo "Resolution: ${w}x${h}  input_len=${input_len}  output_len=${output_len}"
            echo "Log: $(basename "$COMBO_LOG")"
            echo "================================================================"

            MAX_FRAMES_REACHED=0
            MAX_TPS_REACHED=0
            MAX_E2E_REACHED=0

            for imgs in 4 6 8 10 12 16 20 24; do
                echo ""
                echo "--- ${MODEL} ${w}x${h} frames=${imgs} in=${input_len} out=${output_len} quant=${QUANT} ---"

                # arg11="1"       -> KEEP_SERVER_UP (server reused across all combos)
                # arg12=MAX_IMGS  -> server allows up to MAX_IMGS imgs/req
                # arg13=COMBO_TS  -> all frames in this combo share one log file
                if ! bash "$SCRIPT_DIR/vllm_random_benchmark_server.sh" \
                    "$MODEL" "$w" "$h" "$imgs" "$MTP" "$QUANT" "$DEVICE" \
                    "$output_len" "$input_len" "$FIXED_BATCH" "1" "$MAX_IMGS" "$COMBO_TS"; then
                    echo "  ERROR: benchmark failed (OOM / Bad Request) — stopping frames sweep"
                    break
                fi

                E2E=$(grep 'Benchmark duration (s):' "$COMBO_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
                TPS=$(grep 'Output token throughput (tok/s):' "$COMBO_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')

                echo "  frames=${imgs}  E2E=${E2E}s  TPS=${TPS} tok/s"

                OVER=$(awk -v e="${E2E:-0}" -v lim="$E2E_LIMIT" 'BEGIN { print (e > lim) ? 1 : 0 }')
                if [ "$OVER" = "1" ]; then
                    echo "  E2E ${E2E}s > ${E2E_LIMIT}s limit — stopping frames sweep for this config"
                    break
                fi

                MAX_FRAMES_REACHED=$imgs
                MAX_TPS_REACHED=$TPS
                MAX_E2E_REACHED=$E2E
            done

            echo ""
            echo "  >> Result: ${w}x${h} in=${input_len} out=${output_len}"
            echo "     Max frames @ E2E<${E2E_LIMIT}s : $MAX_FRAMES_REACHED"
            echo "     E2E at max frames              : ${MAX_E2E_REACHED}s"
            echo "     TPS at max frames              : $MAX_TPS_REACHED tok/s"
            echo "$GPU_TYPE, $MODEL_LABEL, ${QUANT^^}, ${w}x${h}, $input_len, $output_len, $FIXED_BATCH, $MAX_FRAMES_REACHED, $MAX_E2E_REACHED, $MAX_TPS_REACHED" >> "$SUMMARY_CSV"
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

# Parse all combo logs generated in this sweep -> per-combo CSV
echo "Parsing logs..."
find "$SCRIPT_DIR/$MODEL_DIR" -name "*_client.log" \
    -newer "$_sweep_ref" 2>/dev/null | sort | while read -r log; do
    echo "  Parsing: $log"
    python3 "$SCRIPT_DIR/parse_log.py" "$log"
done
rm -f "$_sweep_ref"

echo ""
echo "Summary CSV: $SUMMARY_CSV"
echo "All done."
