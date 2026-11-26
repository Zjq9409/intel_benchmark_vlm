#!/bin/bash
set -Eeuo pipefail

# ================================================================
# 复现 vLLM 博客 "Scaling Multi-GPU Video Captioning with PyNvVideoCodec and vLLM"
#   https://vllm.ai/blog/2026-09-18-pynvvideocodec
# 博客配置: Qwen3-VL-8B, 一卡一个 vLLM 副本, CUDA MPS, 16 帧, 输出 100~200 tokens,
#           api-server-count 4, renderer-num-workers 4, mm-ipc-gpu-memory-gb 2。
# 这里用 "每副本一个客户端并行压测" 代替博客的反向代理, 吞吐按副本求和。
#
# 用法:
#   bash repro_blog_pynvvideocodec.sh                 # 1 卡 (GPU0), opencv vs pynvvideocodec
#   REPLICAS=4 bash repro_blog_pynvvideocodec.sh      # 4 副本 (GPU0-3)
#   BACKENDS=pynvvideocodec USE_MPS=0 bash ...        # 单测某后端 / 关闭 MPS
# 可调: MAX_MODEL_LEN QUANT MAX_NUM_SEQS MAX_BATCHED_TOKENS GPU_MEM_UTIL REPLICAS GPU_LIST CONCURRENCY NUM_PROMPTS OUTPUT_LEN FRAMES MM_IPC_GB
#       API_SERVER_COUNT BASE_PORT VIDEO_FILES
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
CONTAINER="${VLLM_CONTAINER:-vllm-video-nv}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
MODEL_SRC="${MODEL_SRC:-/DISK0/weights/$MODEL_NAME}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-$HOME/weights/$MODEL_NAME}"   # 容器内 /llm/models/$MODEL_NAME

REPLICAS="${REPLICAS:-1}"
GPU_LIST="${GPU_LIST:-$(seq -s, 0 $((REPLICAS - 1)))}"
BACKENDS="${BACKENDS:-opencv pynvvideocodec}"
# 博客用 80GB H100 (bf16, 1024 seqs); 24GB 的 4090D 放不下, 默认 fp8 + 更小的批规模
QUANT="${QUANT:-fp8}"                       # none 则为博客的 bf16
MAX_NUM_SEQS="${MAX_NUM_SEQS:-120}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"      # 博客为 32768; 24GB 卡扣掉 NVDEC 预留后 KV 放不下
USE_MPS="${USE_MPS:-1}"
FRAMES="${FRAMES:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-150}"
CONCURRENCY="${CONCURRENCY:-64}"            # 每个副本的并发
NUM_PROMPTS="${NUM_PROMPTS:-256}"           # 每个副本的请求数
MM_IPC_GB="${MM_IPC_GB:-2}"
HW_DECODERS="${HW_DECODERS:-1}"            # 解码器缓存约 1GiB x hw_decoders x api-server-count, 24GB 卡用 1
API_SERVER_COUNT="${API_SERVER_COUNT:-2}"   # 每个 API server 的 NVDEC 预留约 2GiB, 24GB 卡用 2 (博客为 4)
BASE_PORT="${BASE_PORT:-8100}"
VIDEO_FILES="${VIDEO_FILES:-/llm/performance_benchmark/video/videos_h264_30fps/bears_1080p_h264.mp4 /llm/performance_benchmark/video/videos_h265_30fps/bears_1080p_h265.mp4}"
PROMPT="${PROMPT:-Analyze this front-facing dashcam video. Briefly describe the driving environment, relevant road users and traffic controls, and the ego vehicle actions. Report only clearly visible details and avoid speculation.}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${RUN_LABEL:-}"
RUN_SUFFIX="${RUN_LABEL:+_${RUN_LABEL}}"
if [ -n "${MATRIX_RUN_ROOT_HOST:-}" ] && [ -n "${MATRIX_RUN_ROOT_CTR:-}" ] && [ -n "$RUN_LABEL" ]; then
    OUT_HOST="$MATRIX_RUN_ROOT_HOST/$RUN_LABEL"
    OUT_CTR="$MATRIX_RUN_ROOT_CTR/$RUN_LABEL"
else
    OUT_HOST="$SCRIPT_DIR/blog_repro/${RUN_TS}${RUN_SUFFIX}_${REPLICAS}gpu"
    OUT_CTR="/llm/performance_benchmark/video/blog_repro/${RUN_TS}${RUN_SUFFIX}_${REPLICAS}gpu"
fi
mkdir -p "$OUT_HOST"
printf '{"run_label":"%s","video_files":"%s","replicas":%s,"gpu_list":"%s","cpu_profile":"%s","cpuset_cpus":"%s","frames":%s,"concurrency_per_replica":%s,"num_prompts_per_replica":%s,"output_tokens":%s}\n' \
    "$RUN_LABEL" "$VIDEO_FILES" "$REPLICAS" "$GPU_LIST" "${CPU_PROFILE:-unknown}" "${CPUSET_CPUS:-unknown}" "$FRAMES" "$CONCURRENCY" "$NUM_PROMPTS" "$OUTPUT_LEN" \
    > "$OUT_HOST/run_info.json"
: > "$OUT_HOST/startup.log"
: > "$OUT_HOST/benchmark.log"
printf '{"backends": {}}\n' > "$OUT_HOST/summary.json"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[ "${#GPUS[@]}" -eq "$REPLICAS" ] || { echo "GPU_LIST 数量 (${#GPUS[@]}) 与 REPLICAS ($REPLICAS) 不一致" >&2; exit 1; }

# 容器只挂载了 ~/weights -> /llm/models, 权重不在其中时先复制
if [ ! -f "$MODEL_HOST_DIR/config.json" ]; then
    echo "Copying $MODEL_SRC -> $MODEL_HOST_DIR"
    mkdir -p "$MODEL_HOST_DIR"; rsync -a "$MODEL_SRC/" "$MODEL_HOST_DIR/"
fi

read -r -a CONC_LIST <<< "${CONCURRENCY//,/ }"
read -r -a PROMPTS_LIST <<< "${NUM_PROMPTS//,/ }"
if [ "${#PROMPTS_LIST[@]}" -eq 1 ]; then
    for _ in "${CONC_LIST[@]:1}"; do PROMPTS_LIST+=("${PROMPTS_LIST[0]}"); done
fi
[ "${#CONC_LIST[@]}" -eq "${#PROMPTS_LIST[@]}" ] || { echo "CONCURRENCY ($CONCURRENCY) 与 NUM_PROMPTS ($NUM_PROMPTS) 数量不一致" >&2; exit 1; }
for v in "${CONC_LIST[@]}" "${PROMPTS_LIST[@]}"; do
    [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "并发/请求数必须是正整数: $v" >&2; exit 1; }
done

cexec() { docker exec "$CONTAINER" bash -lc "$1"; }
for video in $VIDEO_FILES; do
    cexec "test -f '$video'" || { echo "ERROR: video file not found inside container: $video" >&2; exit 1; }
done
MPS_ENV='export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log'

start_mps() {
    [ "$USE_MPS" = 1 ] || return 0
    cexec "$MPS_ENV; mkdir -p \$CUDA_MPS_PIPE_DIRECTORY \$CUDA_MPS_LOG_DIRECTORY; pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d" \
        || { echo "MPS 启动失败, 可用 USE_MPS=0 跳过" >&2; exit 1; }
    echo "CUDA MPS started"
}
stop_mps() {
    [ "$USE_MPS" = 1 ] || return 0
    cexec "$MPS_ENV; echo quit | nvidia-cuda-mps-control" >/dev/null 2>&1 || true
}

stop_servers() {
    cexec 'for f in /tmp/blog_repro_*.pid; do [ -s "$f" ] || continue; p=$(cat "$f"); kill -TERM -- -"$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true; rm -f "$f"; done; true' || true

    # 后端切换前必须确认旧进程与监听端口都已释放。
    for _ in $(seq 1 60); do
        local_active=0
        for i in $(seq 0 $((REPLICAS - 1))); do
            port=$((BASE_PORT + i))
            if cexec "ss -ltn | grep -q ':$port '" 2>/dev/null; then local_active=1; fi
            if curl -sf -m1 "http://localhost:$port/v1/models" >/dev/null 2>&1; then local_active=1; fi
        done
        [ "$local_active" -eq 0 ] && break
        sleep 1
    done

    # 若宽限期后端口仍占用，强制清理同一组进程和监听端口。
    cexec 'for f in /tmp/blog_repro_*.pid; do [ -s "$f" ] || continue; p=$(cat "$f"); kill -KILL -- -"$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true; rm -f "$f"; done; true' || true

    for _ in $(seq 1 60); do
        local_active=0
        for i in $(seq 0 $((REPLICAS - 1))); do
            port=$((BASE_PORT + i))
            if cexec "ss -ltn | grep -q ':$port '" 2>/dev/null; then local_active=1; fi
        done
        [ "$local_active" -eq 0 ] && break
        sleep 1
    done

    # 等 GPU 显存释放再启动下一个后端。
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_LIST" | awk '{s+=$1} END{print s+0}')
        [ "$used" -lt $((200 * REPLICAS)) ] && return 0
        sleep 1
    done
    echo "WARNING: GPU memory did not fall below expected idle threshold after stopping replicas" >&2
}
NVDEC_MONITOR_PID=""
CPU_MONITOR_PID=""
CURRENT_BACKEND=""
CURRENT_TAG=""
cleanup() {
    if [ -n "$NVDEC_MONITOR_PID" ]; then
        stop_nvdec_monitor "${CURRENT_TAG:-$CURRENT_BACKEND}" || true
    fi
    stop_servers || true
    if [ -n "$CURRENT_BACKEND" ]; then
        compact_backend_outputs "$CURRENT_BACKEND"
    fi
    stop_mps
}
trap cleanup EXIT

start_servers() {
    local backend="$1" media_kwargs extra=""
    if [ "$backend" = pynvvideocodec ]; then
        media_kwargs="{\"video\":{\"backend\":\"pynvvideocodec\",\"min_frames\":$FRAMES,\"max_frames\":$FRAMES,\"hw_decoders\":$HW_DECODERS}}"
        extra="--mm-ipc-gpu-memory-gb $MM_IPC_GB"
    else
        media_kwargs="{\"video\":{\"backend\":\"opencv\",\"num_frames\":$FRAMES,\"min_frames\":$FRAMES,\"max_frames\":$FRAMES}}"
    fi
    local quant_arg=""; [ "$QUANT" != none ] && quant_arg="--quantization $QUANT"
    for i in $(seq 0 $((REPLICAS - 1))); do
        local port=$((BASE_PORT + i)) gpu="${GPUS[$i]}"
        local launcher="$OUT_HOST/${backend}_launch_gpu${gpu}.sh"
        {
            echo '#!/bin/bash'
            [ "$USE_MPS" = 1 ] && echo "$MPS_ENV"
            echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$gpu"
            echo "echo \$\$ > /tmp/blog_repro_$port.pid"
            echo "exec vllm serve /llm/models/$MODEL_NAME --served-model-name $MODEL_NAME --host 0.0.0.0 --port $port \\"
            echo "  --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEM_UTIL $quant_arg \\"
            echo "  --api-server-count $API_SERVER_COUNT --renderer-num-workers $API_SERVER_COUNT --async-scheduling \\"
            echo "  --allowed-local-media-path /llm \\"
            echo "  --media-io-kwargs '$media_kwargs' \\"
            echo "  --mm-processor-kwargs '{\"size\":{\"shortest_edge\":65536,\"longest_edge\":9437184}}' $extra"
        } > "$launcher"
        docker exec -d "$CONTAINER" bash -c "setsid bash '$OUT_CTR/${backend}_launch_gpu${gpu}.sh' > '$OUT_CTR/${backend}_server_gpu${gpu}.log' 2>&1"
    done
    for i in $(seq 0 $((REPLICAS - 1))); do
        local port=$((BASE_PORT + i)) ok=0
        for _ in $(seq 1 180); do
            if curl -sf -m 2 "http://localhost:$port/v1/models" | grep -q "$MODEL_NAME"; then ok=1; break; fi
            grep -q "Engine core initialization failed" "$OUT_HOST/${backend}_server_gpu${GPUS[$i]}.log" 2>/dev/null && break
            sleep 5
        done
        [ "$ok" = 1 ] || { echo "replica on port $port failed to start; see $OUT_HOST/${backend}_server_gpu*.log" >&2; exit 2; }
    done
    echo "$backend: $REPLICAS replica(s) ready"
}

run_clients() {  # $1=label $2=concurrency $3=num_prompts
    local label="$1" conc="$2" n="$3" pids=()
    for i in $(seq 0 $((REPLICAS - 1))); do
        VIDEO_DIRS="$VIDEO_FILES" PROMPT="$PROMPT" OUTPUT_LEN="$OUTPUT_LEN" RESULT_DIR="$OUT_CTR" \
        BENCH_LABEL="${label}_r${i}" \
            bash "$SCRIPT_DIR/run_video_bench_serve.sh" "$MODEL_NAME" $((BASE_PORT + i)) "$conc" "$n" \
            > "$OUT_HOST/${label}_r${i}.client.log" 2>&1 &
        pids+=($!)
    done
    local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
}

NVDEC_MONITOR_PID=""
start_nvdec_monitor() {
    local backend="$1"
    local csv="$OUT_HOST/${backend}_nvdec.csv"
    printf 'timestamp,gpu,decoder_utilization_pct\n' > "$csv"
    nvidia-smi --query-gpu=timestamp,index,utilization.decoder \
        --format=csv,noheader,nounits -i "$GPU_LIST" -l 1 >> "$csv" &
    NVDEC_MONITOR_PID=$!

    local cpu_csv="$OUT_HOST/${backend}_cpu.csv"
    printf 'timestamp,cpu_util_pct,cpu_busy_cores\n' > "$cpu_csv"
    (
        prev=$(cpu_snapshot)
        while sleep 1; do
            cur=$(cpu_snapshot)
            echo "$(date '+%Y/%m/%d %H:%M:%S.%3N'),$(awk -v a="$prev" -v b="$cur" -v n="$(nproc)" 'BEGIN{split(a,x," "); split(b,y," "); d=y[2]-x[2]; u=(d>0)?100*(1-(y[1]-x[1])/d):0; printf "%.1f,%.1f", u, u/100*n}')" >> "$cpu_csv"
            prev=$cur
        done
    ) &
    CPU_MONITOR_PID=$!
}

stop_nvdec_monitor() {
    local backend="$1"
    local csv="$OUT_HOST/${backend}_nvdec.csv"
    [ -n "$NVDEC_MONITOR_PID" ] || return 0
    kill "$NVDEC_MONITOR_PID" 2>/dev/null || true
    wait "$NVDEC_MONITOR_PID" 2>/dev/null || true
    if [ -n "$CPU_MONITOR_PID" ]; then
        kill "$CPU_MONITOR_PID" 2>/dev/null || true
        wait "$CPU_MONITOR_PID" 2>/dev/null || true
        CPU_MONITOR_PID=""
    fi
    NVDEC_MONITOR_PID=""

    python3 - "$csv" "$OUT_HOST/${backend}_nvdec_summary.txt" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

csv_path, summary_path = sys.argv[1:]
values = defaultdict(list)
with open(csv_path, newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source, skipinitialspace=True):
        try:
            values[row["gpu"].strip()].append(float(row["decoder_utilization_pct"]))
        except (KeyError, ValueError):
            continue
with open(summary_path, "w", encoding="utf-8") as destination:
    destination.write("NVDEC decoder utilization during measured benchmark (%)\n")
    for gpu, samples in sorted(values.items()):
        destination.write(
            f"GPU {gpu}: mean={statistics.mean(samples):.2f}% "
            f"max={max(samples):.2f}% min={min(samples):.2f}% samples={len(samples)}\n"
        )
print(open(summary_path, encoding="utf-8").read(), end="")
PY
}

cpu_snapshot() { awk '/^cpu /{idle=$5+$6; t=0; for(i=2;i<=NF;i++) t+=$i; print idle, t}' /proc/stat; }

summarize() {  # $1=backend $2=label $3=cpu_before $4=cpu_after $5=concurrency
    python3 - "$OUT_HOST" "$1" "$2" "$5" "$REPLICAS" "$3" "$4" "$(nproc)" <<'PY'
import json, os, sys
out, backend, label, conc, replicas, before, after, cores = sys.argv[1:]
paths = [f"{out}/{label}_r{i}-concurrency{conc}.json" for i in range(int(replicas))]
missing = [path for path in paths if not os.path.isfile(path)]
if missing:
    sys.exit("ERROR: benchmark result missing: " + ", ".join(missing) + "; inspect benchmark.log for client errors")
rows = [json.load(open(path)) for path in paths]
i0, t0 = map(float, before.split()); i1, t1 = map(float, after.split())
cpu = 100 * (1 - (i1 - i0) / (t1 - t0))
s = {
    "backend": backend, "replicas": int(replicas), "concurrency_per_replica": int(conc),
    "num_prompts_per_replica": rows[0]["num_prompts"],
    "completed": sum(r["completed"] for r in rows), "failed": sum(r.get("failed", 0) for r in rows),
    "request_throughput": sum(r["request_throughput"] for r in rows),
    "output_throughput": sum(r["output_throughput"] for r in rows),
    "mean_ttft_ms": sum(r["mean_ttft_ms"] for r in rows) / len(rows),
    "total_input_tokens": sum(r["total_input_tokens"] for r in rows),
    "total_output_tokens": sum(r["total_output_tokens"] for r in rows),
    "input_tokens_per_request": round(sum(r["total_input_tokens"] for r in rows) / sum(r["completed"] for r in rows), 1),
    "output_tokens_per_request": round(sum(r["total_output_tokens"] for r in rows) / sum(r["completed"] for r in rows), 1),
    "host_cpu_util_pct": round(cpu, 1), "host_cpu_busy_cores": round(cpu / 100 * int(cores), 1),
}
json.dump(s, open(f"{out}/{backend}_c{conc}_summary.json", "w"), indent=2)
print(f"[{backend} c={conc}] req/s={s['request_throughput']:.2f} out_tok/s={s['output_throughput']:.0f} "
      f"in_tok/req={s['input_tokens_per_request']:.0f} out_tok/req={s['output_tokens_per_request']:.0f} "
      f"in_tok_total={s['total_input_tokens']} out_tok_total={s['total_output_tokens']} "
      f"TTFT={s['mean_ttft_ms']:.0f}ms completed={s['completed']} failed={s['failed']} "
      f"CPU={s['host_cpu_util_pct']}% (~{s['host_cpu_busy_cores']} cores)")
PY
}


compact_backend_outputs() {
    local backend="$1" file tag
    local benchmark_log="$OUT_HOST/benchmark.log"
    local server_log="$OUT_HOST/startup.log"
    local benchmark_logs=("$OUT_HOST/${backend}"_r*_bench_serve.log "$OUT_HOST/${backend}"_warmup_r*_bench_serve.log "$OUT_HOST/${backend}"_r*.client.log "$OUT_HOST/${backend}"_warmup_r*.client.log)
    local launchers=("$OUT_HOST/${backend}"_launch_gpu*.sh)
    local server_logs=("$OUT_HOST/${backend}"_server_gpu*.log)

    printf '\n===== %s benchmark =====\n' "$backend" >> "$benchmark_log"
    for file in "${benchmark_logs[@]}"; do
        [ -f "$file" ] || continue
        printf '\n===== %s =====\n' "$(basename "$file")" >> "$benchmark_log"
        cat "$file" >> "$benchmark_log"
        rm -f "$file"
    done

    printf '\n===== %s server startup =====\n' "$backend" >> "$server_log"
    for file in "${launchers[@]}"; do
        [ -f "$file" ] || continue
        printf '\n===== %s (launch script) =====\n' "$(basename "$file")" >> "$server_log"
        cat "$file" >> "$server_log"
    done
    for file in "${server_logs[@]}"; do
        [ -f "$file" ] || continue
        printf '\n===== %s =====\n' "$(basename "$file")" >> "$server_log"
        cat "$file" >> "$server_log"
        rm -f "$file"
    done

    for file in "$OUT_HOST/${backend}"_c*_nvdec_summary.txt; do
        [ -f "$file" ] || continue
        tag="$(basename "$file" _nvdec_summary.txt)"
        printf '\n===== NVDEC utilization (%s) =====\n' "$tag" >> "$benchmark_log"
        cat "$file" >> "$benchmark_log"
    done

    for file in "$OUT_HOST/${backend}"_c*_nvdec.csv; do
        [ -f "$file" ] || continue
        tag="$(basename "$file" _nvdec.csv)"
        python3 - "$backend" "${tag##*_c}" "$file" "$OUT_HOST/${tag}_cpu.csv" "$OUT_HOST/utilization.csv" <<'PYUTIL'
import csv
import os
import sys
from datetime import datetime

backend, conc, nvdec_path, cpu_path, out_path = sys.argv[1:]

def ts(text):
    return datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()

ticks = []
if os.path.exists(nvdec_path):
    with open(nvdec_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source, skipinitialspace=True):
            try:
                t = ts(row["timestamp"])
                gpu = row["gpu"].strip()
                value = row["decoder_utilization_pct"].strip()
            except (KeyError, ValueError):
                continue
            if not ticks or t - ticks[-1][0] > 0.5:
                ticks.append([t, row["timestamp"].strip(), {}])
            ticks[-1][2][gpu] = value

cpu = []
if os.path.exists(cpu_path):
    with open(cpu_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                cpu.append((ts(row["timestamp"]), row["timestamp"], row["cpu_util_pct"], row["cpu_busy_cores"]))
            except (KeyError, ValueError):
                continue

if not ticks:
    ticks = [[t, text, {}] for t, text, _, _ in cpu]

gpus = sorted({g for _, _, d in ticks for g in d}, key=int)
header = ["backend", "concurrency_per_replica", "timestamp", "cpu_util_pct", "cpu_busy_cores"] + [f"nvdec_gpu{g}_pct" for g in gpus]
write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
with open(out_path, "a", newline="", encoding="utf-8") as destination:
    writer = csv.writer(destination)
    if write_header:
        writer.writerow(header)
    for t, text, values in ticks:
        best = min(cpu, key=lambda c: abs(c[0] - t), default=None)
        match = best if best is not None and abs(best[0] - t) <= 0.6 else None
        writer.writerow([backend, conc, text, match[2] if match else "", match[3] if match else ""] + [values.get(g, "") for g in gpus])
PYUTIL
    done

    for file in "$OUT_HOST/${backend}"_c*_summary.json; do
        [ -f "$file" ] || continue
        tag="$(basename "$file" _summary.json)"
        python3 - "$OUT_HOST/summary.json" "$backend" "${tag##*_c}" "$file" "$OUT_HOST/${tag}_nvdec_summary.txt" <<'PYJSON'
import json
import os
import re
import sys

combined_path, backend, conc, source_path, nvdec_path = sys.argv[1:]
with open(combined_path, encoding="utf-8") as source:
    combined = json.load(source)
with open(source_path, encoding="utf-8") as source:
    entry = json.load(source)
if os.path.exists(nvdec_path):
    per_gpu = {}
    with open(nvdec_path, encoding="utf-8") as source:
        for line in source:
            m = re.match(r"GPU (\d+): mean=([\d.]+)% max=([\d.]+)%", line)
            if m:
                per_gpu[m.group(1)] = {"mean_pct": float(m.group(2)), "max_pct": float(m.group(3))}
    if per_gpu:
        entry["nvdec_mean_util_pct"] = round(sum(g["mean_pct"] for g in per_gpu.values()) / len(per_gpu), 2)
        entry["nvdec_max_util_pct"] = max(g["max_pct"] for g in per_gpu.values())
        entry["nvdec_util_per_gpu"] = per_gpu
combined["backends"].setdefault(backend, {})[conc] = entry
with open(combined_path, "w", encoding="utf-8") as destination:
    json.dump(combined, destination, indent=2)
    destination.write("\n")
PYJSON
    done

    # Remove all per-replica and intermediate files; leave the combined outputs.
    rm -f "$OUT_HOST/${backend}"*.json "$OUT_HOST/${backend}"*.log \
        "$OUT_HOST/${backend}"*.csv "$OUT_HOST/${backend}"*.txt \
        "$OUT_HOST/${backend}"_launch_gpu*.sh
}

echo "Replicas=$REPLICAS GPUs=$GPU_LIST frames=$FRAMES out=$OUTPUT_LEN conc/replica=$CONCURRENCY prompts/replica=$NUM_PROMPTS MPS=$USE_MPS"
echo "Results: $OUT_HOST"
start_mps
for backend in $BACKENDS; do
    CURRENT_BACKEND="$backend"
    start_servers "$backend"
    run_clients "${backend}_warmup" 4 $((8)) >/dev/null || echo "warmup had failures (see logs)"
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); n=d["completed"]; print("[%s] Input length: %.0f tokens/request, output length: %.0f tokens/request" % (sys.argv[2], d["total_input_tokens"]/n, d["total_output_tokens"]/n))' "$OUT_HOST/${backend}_warmup_r0-concurrency4.json" "$backend" 2>/dev/null || true
    for idx in "${!CONC_LIST[@]}"; do
        conc="${CONC_LIST[$idx]}"
        n="${PROMPTS_LIST[$idx]}"
        tag="${backend}_c${conc}"
        CURRENT_TAG="$tag"
        before=$(cpu_snapshot)
        start_nvdec_monitor "$tag"
        run_clients "$backend" "$conc" "$n" || echo "benchmark clients reported errors (concurrency=$conc)" >&2
        stop_nvdec_monitor "$tag"
        after=$(cpu_snapshot)
        summarize "$backend" "$backend" "$before" "$after" "$conc"
        CURRENT_TAG=""
    done
    stop_servers
    compact_backend_outputs "$backend"
    CURRENT_BACKEND=""
done
echo "Done: $OUT_HOST"
