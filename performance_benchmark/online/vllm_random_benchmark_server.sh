#!/bin/bash

# ----------------------------------------------------------------
# Bare-metal guard: if not inside Docker, re-exec inside container
# ----------------------------------------------------------------
if [ ! -f "/.dockerenv" ] && ! grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    if nvidia-smi &>/dev/null; then
        CONTAINER_NAME="${VLLM_NV_CONTAINER:-vllm-nv-container}"
    else
        CONTAINER_NAME="${VLLM_XPU_CONTAINER:-lsv-container-b8}"
    fi

    SCRIPT_IN_CONTAINER="$(realpath "$0")"

    _SELF_DIR="$(dirname "$(realpath "$0")")"
    _HOST_REPO_BASE="$(dirname "$(dirname "$_SELF_DIR")")"
    # Translate host script path to container path (e.g. /home/intel/jane/intel_benchmark_vlm -> /llm)
    if [[ "$SCRIPT_IN_CONTAINER" == "$_HOST_REPO_BASE"* ]]; then
        SCRIPT_IN_CONTAINER="/llm${SCRIPT_IN_CONTAINER#$_HOST_REPO_BASE}"
    fi
    _WEIGHTS_DIR="${WEIGHTS_DIR:-$(dirname "$_HOST_REPO_BASE")/weights}"
    _WEIGHTS_DIR="$(realpath "$_WEIGHTS_DIR" 2>/dev/null || echo "${_WEIGHTS_DIR%/}")"

    TRANSLATED_ARGS=()
    for arg in "$@"; do
        arg_real="$(realpath "$arg" 2>/dev/null || echo "${arg%/}")"
        if [[ "$arg_real" == "$_WEIGHTS_DIR"* ]]; then
            rel="${arg_real#$_WEIGHTS_DIR}"
            arg="/llm/models${rel}"
        fi
        TRANSLATED_ARGS+=("$arg")
    done

    CONTAINER_STATE=$(sudo docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)
    if [ -z "$CONTAINER_STATE" ]; then
        echo "ERROR: Container '$CONTAINER_NAME' does not exist. Run setup_env.sh first."
        exit 1
    fi
    # echo "Bare-metal detected -- restarting container '$CONTAINER_NAME' to kill stale processes..."
    # sudo docker restart "$CONTAINER_NAME"
    # echo "Waiting for container to be ready..."
    # sleep 5
    # echo "Bare-metal detected -- re-executing inside container '$CONTAINER_NAME'..."
    exec sudo docker exec -it "$CONTAINER_NAME" bash "$SCRIPT_IN_CONTAINER" "${TRANSLATED_ARGS[@]}"
fi

# Ensure we run from the script's own directory so LOG/ lands in the right place
cd "$(dirname "$(realpath "$0")")"

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
# Select model via first argument: "4b" | "q35-4b" | "30b" (default)
# 4th argument: images per request (default 1; set 10 to simulate NarratoAI multi-image)
MODEL_SELECT="${1:-30b}"
MM_ITEMS="${4:-1}"
MTP="${5:-off}"   # on=enable speculative decoding, off=disable
QUANT="${6:-fp8}"  # fp8=enable fp8 quantization, none=disable
DEVICE="${7:-}"     # GPU device ID(s), e.g. 4 or 0,1,2,3; empty=use all
OUTPUT_LEN="${8:-1024}"  # output token length; 128=realtime, 512=near-realtime, 1024=batch
E2E_LIMIT_SEC="${15:-60}"  # E2E threshold passed from sweep wrapper (run_nearrt_sweep.sh E2E_LIMIT)
INPUT_LEN="${9:-1024}"   # input token length; 512=short prompt, 1024=standard
FIXED_BATCH="${10:-}"    # if set, run only this single batch size (skip sweep)
KEEP_SERVER_UP="${11:-}"  # if "1", keep server running after benchmark (multi-combo sweep)
SERVER_MM_LIMIT="${12:-$MM_ITEMS}"  # server --limit-mm-per-prompt max (set to sweep max for KEEP_SERVER_UP mode)
SWEEP_TS="${13:-}"            # timestamp passed from sweep wrapper (survives docker exec)
PORT="${14:-8008}"           # vllm server port (passed from sweep wrapper)
MAX_BATCHED_TOKENS="${16:-32768}"  # max batched tokens (passed from sweep wrapper)
MAX_MODEL_LEN="${17:-32768}"       # max model context length (passed from sweep wrapper)
GPU_MEM_UTIL="${18:-0.9}"          # GPU memory utilization (passed from sweep wrapper)
MAX_NUM_SEQS="${19:-}"             # max number of sequences (empty=use vllm default)
TP_OVERRIDE="${20:-}"              # tensor parallelism override (empty=use model default)
MAX_SWEEP_BATCH_OVERRIDE="${21:-${MAX_SWEEP_BATCH:-}}"  # safety cap; sweep stops earlier once E2E exceeds E2E_LIMIT_SEC
BATCH_SWEEP_STEP="${22:-${BATCH_SWEEP_STEP:-4}}"        # fallback coarse batch sweep step; refined by 1 near the E2E limit
BATCH_SWEEP_LIST="${23:-${BATCH_SWEEP_LIST:-}}"          # aligned batch candidates across hardware
E2E_THRESHOLDS="${24:-${E2E_THRESHOLDS:-5 10 20 30}}"  # thresholds used for plotting/refinement
REFINE_BATCH_STEP="${25:-${REFINE_BATCH_STEP:-4}}"  # batch step used when refining threshold boundaries

# shellcheck source=model_config.sh
source "$(dirname "$(realpath "$0")")/model_config.sh"
resolve_model "$MODEL_SELECT"
[ -n "$TP_OVERRIDE" ] && TP="$TP_OVERRIDE"
MM_W="${2:-1280}"
MM_H="${3:-720}"

# Setup logging
mkdir -p "$SERVER_MODEL_NAME"
CURRENT_TIME="${SWEEP_TS:-${SWEEP_TIMESTAMP:-$(date "+%Y%m%d_%H%M%S")}}"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //g; s/GeForce //g; s/Quadro //g; s/Tesla //g' | tr -d ' \r')
[ -z "$GPU_TYPE" ] && GPU_TYPE="XPU"
MTP_TAG=$([ "$MTP" = "on" ] && echo "mtp_" || echo "nomtp_")
MTP_LABEL=$([ "$MTP" = "on" ] && echo "mtp" || echo "nomtp")
QUANT_TAG=$([ "$QUANT" = "none" ] && echo "fp16_" || echo "${QUANT}_")
MODEL_TAG="$SERVER_MODEL_NAME"
MODEL_TAG="${MODEL_TAG//\//_}"
LOG_FILE="${SERVER_MODEL_NAME}/${CURRENT_TIME}_${MODEL_TAG}_${QUANT}_${MTP_LABEL}_${MAX_BATCHED_TOKENS}_${GPU_TYPE}_client.log"
SERVER_LOG="${SERVER_MODEL_NAME}/${CURRENT_TIME}_${MODEL_TAG}_${QUANT}_${MTP_LABEL}_${MAX_BATCHED_TOKENS}_${GPU_TYPE}_server.log"
echo "Server LOG will be logged to: $SERVER_LOG"


{
echo "---------------------------------------------------"
echo "Starting ShareGPT Benchmark with the following parameters:"
echo "Server Model Path:  $SERVER_MODEL"
echo "Server Model Name:  $SERVER_MODEL_NAME"
echo "Port:               $PORT"
echo "TP:                 $TP"
echo "Max Batched Tokens: $MAX_BATCHED_TOKENS"
echo "Max Model Len:      $MAX_MODEL_LEN"
echo "GPU Mem Util:       $GPU_MEM_UTIL"
echo "GPU Type:           $GPU_TYPE"
echo "Images per request: $MM_ITEMS"
echo "Quantization:       $QUANT"
echo "GPU Device:         ${DEVICE:-all}"
echo "---------------------------------------------------"
} | tee -a "$LOG_FILE"

# Apply CUDA_VISIBLE_DEVICES if DEVICE is set (must be exported BEFORE vllm starts)
if [ -n "$DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    echo "Using GPU device(s): $DEVICE (CUDA_VISIBLE_DEVICES=$DEVICE)"
fi

# Start vllm server
echo "Starting vllm server..."

#--max-num-batched-tokens=$MAX_BATCHED_TOKENS
VLLM_SERVER_ARGS=(
    --model "$SERVER_MODEL"
    --served-model-name "$SERVER_MODEL_NAME"
    --dtype=float16
    --port $PORT
    --host 0.0.0.0
    --trust-remote-code
    --no-enable-prefix-caching
    --gpu-memory-util=$GPU_MEM_UTIL
    --max-model-len=$MAX_MODEL_LEN
    --mm-processor-cache-gb 0
    --async-scheduling 
    -tp=$TP
)
[ -n "$MAX_NUM_SEQS" ] && VLLM_SERVER_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")

# Quantization (6th arg: fp8/none, default fp8)
if [ "$QUANT" != "none" ]; then
    VLLM_SERVER_ARGS+=(--quantization "$QUANT")
fi

# MTP speculative decoding (5th arg: on/off)
if [ "$MTP" = "on" ]; then
    VLLM_SERVER_ARGS+=(--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
fi

# Remove the extra closing paren — restore block
export PYTORCH_ALLOC_CONF="expandable_segments:True"
if [ "$GPU_TYPE" = "XPU" ]; then
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1   
    VLLM_SERVER_ARGS+=(--block-size 64)
    if [ "$QUANT" != "none" ]; then
        VLLM_SERVER_ARGS+=(--enforce-eager)
    fi
else
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export NCCL_P2P_LEVEL=SYS
fi


SERVER_PID_FILE="/tmp/vllm_server_${PORT}.pid"
SERVER_ALREADY_UP=0
if curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVER_MODEL_NAME"; then
    SERVER_ALREADY_UP=1
    SERVER_PID=$(cat "$SERVER_PID_FILE" 2>/dev/null || echo "0")
    echo "Server already running (PID=${SERVER_PID}), skipping start..." | tee -a "$LOG_FILE"
fi
if [ "$SERVER_ALREADY_UP" = "0" ]; then
    nohup vllm serve "${VLLM_SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$SERVER_PID_FILE"
    echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
MAX_RETRIES=120
COUNT=0
SERVER_READY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "$SERVER_MODEL_NAME"; then
        SERVER_READY=1; break
    fi
    sleep 5
    COUNT=$((COUNT+1))
    echo "Waiting for server... ($COUNT/$MAX_RETRIES)"
done

if [ $SERVER_READY -eq 0 ]; then
    echo "Server failed to start within timeout. Checking logs:"
    tail -n 20 "$SERVER_LOG"
    kill $SERVER_PID
    rm -f "$SERVER_PID_FILE"
    exit 1
fi
fi  # SERVER_ALREADY_UP

# Run benchmarks
# Keep a safety cap for pathological cases; normal sweep stops once E2E exceeds E2E_LIMIT_SEC.
if [ "$MM_ITEMS" -gt 1 ]; then
    MAX_BSIZE=64
else
    MAX_BSIZE=512
fi
if [ -n "$MAX_SWEEP_BATCH_OVERRIDE" ] && [ "$MAX_SWEEP_BATCH_OVERRIDE" -gt 0 ] 2>/dev/null; then
    MAX_BSIZE="$MAX_SWEEP_BATCH_OVERRIDE"
fi
if ! [ "$BATCH_SWEEP_STEP" -gt 0 ] 2>/dev/null; then
    BATCH_SWEEP_STEP=4
fi
if ! [ "$REFINE_BATCH_STEP" -gt 0 ] 2>/dev/null; then
    REFINE_BATCH_STEP=4
fi

MM_BUCKET_CONFIG="{(${MM_H},${MM_W}, 1): 1.0}"

run_benchmark() {
    local bsize=$1
    echo "[$(date +%Y%m%d_%H%M%S)] === BENCHMARK batch=$bsize imgs=${MM_ITEMS} res=${MM_W}x${MM_H} in=${INPUT_LEN} out=${OUTPUT_LEN} quant=${QUANT} ===" | tee -a "$LOG_FILE"
    echo ">>> Running vllm bench serve with --num-prompts=$bsize" | tee -a "$LOG_FILE"
    ( set -o pipefail; vllm bench serve \
        --model "$SERVER_MODEL" \
        --served-model-name "$SERVER_MODEL_NAME" \
        --endpoint /v1/chat/completions \
        --dataset-name random-mm \
        --num-prompts $bsize \
        --max-concurrency $bsize \
        --ready-check-timeout-sec 1 --num-warmups 1 \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --random-mm-base-items-per-request $MM_ITEMS \
        --random-mm-limit-mm-per-prompt '{"image": '"${MM_ITEMS}"', "video": 0}' \
        --random-mm-bucket-config "$MM_BUCKET_CONFIG" \
        --request-rate inf \
        --backend openai-chat \
        --ignore-eos \
        --port=$PORT \
        --seed 42 2>&1 | tee -a "$LOG_FILE" )
}

STOP_SWEEP=0

check_stop() {
    # 用 Benchmark Duration (s) 作为 E2E Latency（最准确）
    local e2e_s e2e_ms e2e_limit
    e2e_s=$(grep 'Benchmark duration (s):' "$LOG_FILE" | tail -1 | awk '{print $NF}')
    [ -z "$e2e_s" ] && return 0
    e2e_ms=$(awk -v v="$e2e_s" 'BEGIN { printf "%.0f", v * 1000 }')
    # 停止阈值：统一阈值（准实时场景）
    e2e_limit=$((E2E_LIMIT_SEC * 1000))  # Convert to milliseconds
    echo "  E2E: ${e2e_s}s (limit: $(( e2e_limit / 1000 ))s)"
    if awk "BEGIN { exit !(${e2e_ms} > ${e2e_limit}) }"; then
        echo "E2E ${e2e_s}s exceeds $(( e2e_limit / 1000 ))s threshold."
        STOP_SWEEP=1
        python3 "$(dirname "$0")/parse_log.py" "$LOG_FILE"
        if [ -z "$KEEP_SERVER_UP" ]; then
            # Standalone mode: kill server and exit
            kill $SERVER_PID
            exit 0
        fi
        # Sweep mode: server stays up, caller breaks the loop
        return 1
    fi
    return 0
}

find_threshold_bounds() {
    local threshold=$1
    awk -v lim="$threshold" -v target_imgs="$MM_ITEMS" -v target_res="${MM_W}x${MM_H}" -v target_in="$INPUT_LEN" -v target_out="$OUTPUT_LEN" '
        /=== BENCHMARK batch=/ {
            is_target=0
            for (i=1;i<=NF;i++) {
                if ($i ~ /^batch=/) { sub(/batch=/,"",$i); cur_batch=$i+0 }
                if ($i ~ /^imgs=/) { sub(/imgs=/,"",$i); cur_imgs=$i+0 }
                if ($i ~ /^res=/) { sub(/res=/,"",$i); cur_res=$i }
                if ($i ~ /^in=/) { sub(/in=/,"",$i); cur_in=$i+0 }
                if ($i ~ /^out=/) { sub(/out=/,"",$i); cur_out=$i+0 }
            }
            if (cur_imgs+0 == target_imgs+0 && cur_res == target_res && cur_in+0 == target_in+0 && cur_out+0 == target_out+0) is_target=1
        }
        is_target && /Benchmark duration \(s\):/ {
            e2e=$NF+0
            if (e2e <= lim && cur_batch > ok) ok=cur_batch
            if (e2e > lim && (fail == 0 || cur_batch < fail)) fail=cur_batch
        }
        END { print ok+0, fail+0 }
    ' "$LOG_FILE" 2>/dev/null
}

batch_already_run() {
    local target_batch=$1
    awk -v target_batch="$target_batch" -v target_imgs="$MM_ITEMS" -v target_res="${MM_W}x${MM_H}" -v target_in="$INPUT_LEN" -v target_out="$OUTPUT_LEN" '
        /=== BENCHMARK batch=/ {
            batch=imgs=inlen=outlen=0; res=""
            for (i=1;i<=NF;i++) {
                if ($i ~ /^batch=/) { sub(/batch=/,"",$i); batch=$i+0 }
                if ($i ~ /^imgs=/) { sub(/imgs=/,"",$i); imgs=$i+0 }
                if ($i ~ /^res=/) { sub(/res=/,"",$i); res=$i }
                if ($i ~ /^in=/) { sub(/in=/,"",$i); inlen=$i+0 }
                if ($i ~ /^out=/) { sub(/out=/,"",$i); outlen=$i+0 }
            }
            if (batch == target_batch && imgs == target_imgs && res == target_res && inlen == target_in && outlen == target_out) found=1
        }
        END { exit(found ? 0 : 1) }
    ' "$LOG_FILE" 2>/dev/null
}

refine_thresholds() {
    local threshold low high fine_start fine_end b
    echo "  Refining E2E thresholds for plotting: ${E2E_THRESHOLDS}"
    for threshold in $E2E_THRESHOLDS; do
        if ! awk -v t="$threshold" -v max="$E2E_LIMIT_SEC" 'BEGIN { exit !(t > 0 && t <= max) }'; then
            echo "  Skip invalid/out-of-range E2E threshold: ${threshold}s"
            continue
        fi

        read low high <<< "$(find_threshold_bounds "$threshold")"
        if [ "$high" -gt 0 ] && [ "$low" -gt 0 ]; then
            fine_start=$((low + 1))
            fine_end=$((high - 1))
            if [ "$fine_start" -le "$fine_end" ]; then
                echo "  Refining E2E<=${threshold}s batch range ${fine_start}..${fine_end} with STEP=${REFINE_BATCH_STEP}"
                STOP_SWEEP=0
                b=$((low + REFINE_BATCH_STEP))
                while [ $b -le $fine_end ]; do
                    if ! batch_already_run "$b"; then
                        run_benchmark "$b"
                        check_stop || break
                    fi
                    b=$((b + REFINE_BATCH_STEP))
                done
            fi
        fi

        read low high <<< "$(find_threshold_bounds "$threshold")"
        if [ "$high" -gt 0 ]; then
            echo "  E2E<=${threshold}s max batch: ${low} (first >${threshold}s batch=${high})"
        else
            echo "  E2E<=${threshold}s max batch: ${low} (not crossed within tested range)"
        fi
    done
}


if [ -n "$FIXED_BATCH" ]; then
    # Frames-sweep mode: run only the specified batch size
    echo "FIXED_BATCH mode: running batch=$FIXED_BATCH"
    run_benchmark "$FIXED_BATCH"
else
    # Default: dynamic sweep to find max batch @ E2E < limit
    STOP_SWEEP=0
    last_ok=0
    failed_batch=0
    run_benchmark 1
    if check_stop; then
        last_ok=1
    else
        failed_batch=1
    fi

    if [ "$STOP_SWEEP" = "0" ]; then
        estimated_max=$MAX_BSIZE

        if [ -n "$BATCH_SWEEP_LIST" ]; then
            echo "  Using aligned batch candidates: ${BATCH_SWEEP_LIST} (MAX_BSIZE=${MAX_BSIZE}); refine STEP=1 near E2E limit"
            for b in $BATCH_SWEEP_LIST; do
                if ! [ "$b" -gt 0 ] 2>/dev/null; then
                    echo "  Skip invalid batch candidate: $b"
                    continue
                fi
                if [ "$b" -le "$last_ok" ]; then
                    continue
                fi
                if [ "$b" -gt "$estimated_max" ]; then
                    continue
                fi

                run_benchmark $b
                if check_stop; then
                    last_ok=$b
                else
                    failed_batch=$b
                    break
                fi
            done
        else
            # Fallback coarse sweep batch 1,2,6,10,... up to MAX_BSIZE.
            STEP=$BATCH_SWEEP_STEP
            echo "  Using fallback coarse STEP=${STEP} starting from 2 (MAX_BSIZE=${MAX_BSIZE}); refine STEP=1 near E2E limit"

            i=2
            while [ $i -le $estimated_max ] && [ "$STOP_SWEEP" = "0" ]; do
                run_benchmark $i
                if check_stop; then
                    last_ok=$i
                else
                    failed_batch=$i
                    break
                fi
                i=$((i + STEP))
            done
            if [ "$STOP_SWEEP" = "0" ] && [ "$last_ok" -lt "$estimated_max" ]; then
                run_benchmark $estimated_max
                if check_stop; then
                    last_ok=$estimated_max
                else
                    failed_batch=$estimated_max
                fi
            fi
        fi

        refine_thresholds
        read last_ok failed_batch <<< "$(find_threshold_bounds "$E2E_LIMIT_SEC")"
        if [ "$failed_batch" -gt 0 ]; then
            echo "  Max batch candidate with E2E<=${E2E_LIMIT_SEC}s: ${last_ok} (first failed batch=${failed_batch})"
        else
            echo "  Max batch candidate with E2E<=${E2E_LIMIT_SEC}s: ${last_ok} (WARNING: reached safety cap without exceeding E2E limit)"
        fi
    fi
fi

echo "All benchmark runs finished."
if [ -z "$KEEP_SERVER_UP" ]; then
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    rm -f "$SERVER_PID_FILE"
    echo "Done."
else
    echo "Server kept running (KEEP_SERVER_UP=1, PID=${SERVER_PID})."
fi

# Parse log and generate CSV
echo "Parsing log and generating CSV..."
python3 "$(dirname "$0")/parse_log.py" "$LOG_FILE"
echo "CSV saved."
