#!/bin/bash
set -e

# ================================================================
# 准实时场景扫描：一个 Server，测试所有 (分辨率 × in × out × 帧数) 组合
# 指标：E2E < E2E_LIMIT 下每帧数的最大并发 Batch
#
# 用法: bash run_nearrt_sweep.sh [model] [device] [quant] [mtp]
# 示例: bash run_nearrt_sweep.sh 4b "" fp8
#        bash run_nearrt_sweep.sh 4b 4 fp8      # 只用 GPU 4
# ================================================================
MODEL="${1:-4b}"
DEVICE="${2:-}"
QUANT="${3:-fp8}"
MTP="${4:-off}"
E2E_LIMIT=30   # seconds per batch — multi-image 准实时阈值
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
mkdir -p "$SCRIPT_DIR/$MODEL_DIR/LOG"
SUMMARY_CSV="$SCRIPT_DIR/$MODEL_DIR/LOG/nearrt_summary_${RUN_START_FILE}_${MODEL}_${QUANT}_${GPU_TYPE}.csv"
echo "Device, Model, Precision, Image Size, Input Len, Output Len, Frames/Req, Max Batch (E2E<${E2E_LIMIT}s), E2E(s), TPS(tok/s)" > "$SUMMARY_CSV"
_sweep_ref=$(mktemp)

PORT=8006

echo "========================================"
echo "Near-RT Sweep started at: $RUN_START"
echo "Model=$MODEL  device=${DEVICE:-all}  quant=$QUANT  mtp=$MTP  E2E_LIMIT=${E2E_LIMIT}s"
echo "Matrix: res=[720p,1080p] in=[512,1024] out=[128,512,1024] frames=[4,6,8,10,12,16,20,24]"
echo "Mode: dynamic batch sweep per frame count (max concurrent batch @ E2E<${E2E_LIMIT}s)"
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
        for output_len in 128 1024; do

            echo ""
            echo "================================================================"
            echo "Resolution: ${w}x${h}  input_len=${input_len}  output_len=${output_len}"
            echo "Dynamic batch sweep (max batch @ E2E<${E2E_LIMIT}s per frame count)"
            echo "================================================================"

            for imgs in 1 4 6 8 10 12 16 20 24; do
                # Each (imgs) gets its own COMBO_TS -> its own log (holds all batch sweep entries)
                COMBO_TS=$(date "+%Y%m%d_%H%M%S")
                COMBO_LOG="$SCRIPT_DIR/$MODEL_DIR/${COMBO_TS}_${MODEL}_${QUANT}_${MTP_LABEL}_32768_${GPU_TYPE}_client.log"

                echo ""
                echo "--- ${MODEL} ${w}x${h} frames=${imgs} in=${input_len} out=${output_len} quant=${QUANT} ---"
                echo "Log: $(basename "$COMBO_LOG")"

                # arg10="" -> dynamic batch sweep in vllm_random_benchmark_server.sh
                # arg11="1" -> KEEP_SERVER_UP (server reused across combos)
                # arg12=MAX_IMGS -> server allows up to MAX_IMGS imgs/req
                # arg13=COMBO_TS -> log filename key
                if ! bash "$SCRIPT_DIR/vllm_random_benchmark_server.sh" \
                    "$MODEL" "$w" "$h" "$imgs" "$MTP" "$QUANT" "$DEVICE" \
                    "$output_len" "$input_len" "" "1" "$MAX_IMGS" "$COMBO_TS"; then
                    echo "  ERROR: benchmark failed (OOM / Bad Request) — stopping frames sweep"
                    break
                fi

                # Extract max batch where E2E <= E2E_LIMIT from this combo's log
                read MAX_BATCH MAX_E2E MAX_TPS <<< "$(awk -v lim="$E2E_LIMIT" '
                    /=== BENCHMARK batch=/ {
                        for (i=1;i<=NF;i++) if ($i ~ /^batch=/) { sub(/batch=/,"",$i); cur_batch=$i+0 }
                    }
                    /Benchmark duration \(s\):/ {
                        e2e = $NF+0
                        if (e2e <= lim) { mb=cur_batch; me=e2e }
                    }
                    /Output token throughput \(tok\/s\):/ {
                        if (cur_batch+0 == mb+0) mt=$NF+0
                    }
                    END { print mb+0, me+0, mt+0 }
                ' "$COMBO_LOG" 2>/dev/null)"

                echo "  frames=${imgs}  MaxBatch=${MAX_BATCH}  E2E=${MAX_E2E}s  TPS=${MAX_TPS} tok/s"
                echo "$GPU_TYPE, $MODEL_LABEL, ${QUANT^^}, ${w}x${h}, $input_len, $output_len, $imgs, $MAX_BATCH, $MAX_E2E, $MAX_TPS" >> "$SUMMARY_CSV"
            done
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
