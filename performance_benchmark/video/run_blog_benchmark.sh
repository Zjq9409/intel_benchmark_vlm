#!/bin/bash
set -Eeuo pipefail

# Run codec x resolution x duration independently; append all results to one JSON.
# Usage: run_blog_benchmark.sh [replicas] [cores|auto] [backend] [codec] [resolution]
# CPUSET can override affinity; CPU_PROFILE labels the result. CORES=auto preserves
# the container's existing cpuset (normally all host logical CPUs).

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"


# Matrix mode replaces the former run_all_test.sh entrypoint.
if [[ "${1:-}" == --matrix || "${1:-}" == matrix ]]; then
CONTAINER="${VLLM_CONTAINER:-vllm-video-nv}"
GPU_TOTAL="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
GPU_COUNTS=(1 2 4 8)
GOLD6530_SOCKETS="${GOLD6530_SOCKETS:-2}"
MATRIX_ROOT="${MATRIX_ROOT:-$SCRIPT_DIR/blog_repro/$(date +%Y%m%d_%H%M%S)_gpu_cpu_matrix}"
mkdir -p "$MATRIX_ROOT"

if [[ "$GOLD6530_SOCKETS" != 1 && "$GOLD6530_SOCKETS" != 2 ]]; then
    echo "GOLD6530_SOCKETS must be 1 or 2" >&2
    exit 1
fi

# Select 32 physical cores per simulated socket, balanced by NUMA node, and include
# both SMT siblings for each core. This avoids treating 128 leading logical IDs as
# 64 physical cores, which would be false on this host's CPU numbering.
gold6530_cpuset() {
    python3 - "$1" <<'PYCPU'
import glob, os, sys
from collections import defaultdict
socket_count = int(sys.argv[1])
cores = defaultdict(lambda: defaultdict(dict))
for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
    cpu = int(path.rsplit("cpu", 1)[1])
    topo = os.path.join(path, "topology")
    with open(os.path.join(topo, "physical_package_id"), encoding="utf-8") as f: socket = int(f.read())
    with open(os.path.join(topo, "core_id"), encoding="utf-8") as f: core = int(f.read())
    nodes = glob.glob(os.path.join(path, "node[0-9]*"))
    node = int(nodes[0].rsplit("node", 1)[1]) if nodes else 0
    text = open(os.path.join(topo, "thread_siblings_list"), encoding="utf-8").read().strip()
    siblings = []
    for part in text.split(","):
        bounds = [int(value) for value in part.split("-", 1)]
        siblings.extend(range(bounds[0], bounds[-1] + 1))
    cores[socket][node][core] = siblings
sockets = sorted(cores)
if len(sockets) < socket_count:
    raise SystemExit(f"Host has {len(sockets)} sockets; cannot simulate {socket_count}")
selected = []
for socket in sockets[:socket_count]:
    by_node = {node: sorted(items.values(), key=lambda ids: ids[0]) for node, items in cores[socket].items()}
    offsets = {node: 0 for node in by_node}
    chosen = 0
    while chosen < 32:
        progress = False
        for node in sorted(by_node):
            index = offsets[node]
            if index < len(by_node[node]) and chosen < 32:
                siblings = by_node[node][index]
                if len(siblings) != 2:
                    raise SystemExit(f"Expected SMT2; found {len(siblings)} threads for a core")
                selected.extend(siblings)
                offsets[node] += 1
                chosen += 1
                progress = True
        if not progress and chosen < 32:
            raise SystemExit(f"Socket {socket} has fewer than 32 physical cores")
def compact(values):
    values = sorted(set(values))
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
        else:
            ranges.append((start, previous))
            start = previous = value
    ranges.append((start, previous))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)
print(compact(selected))
PYCPU
}

GOLD6530_CPUSET="$(gold6530_cpuset "$GOLD6530_SOCKETS")"
ORIGINAL_CPUSET="$(docker inspect --format '{{.HostConfig.CpusetCpus}}' "$CONTAINER")"
echo "Detected GPUs: $GPU_TOTAL"
echo "All-host CPUSET: ${ORIGINAL_CPUSET:-unrestricted}"
echo "Gold6530 sockets=$GOLD6530_SOCKETS CPUSET=$GOLD6530_CPUSET"
echo "Combined JSON: $MATRIX_ROOT/summary.json"

for profile in host_all gold6530; do
    if [[ "$profile" == host_all ]]; then profile_cpuset=auto; else profile_cpuset="$GOLD6530_CPUSET"; fi
    for replicas in "${GPU_COUNTS[@]}"; do
        if (( replicas > GPU_TOTAL )); then continue; fi
        echo "===== CPU profile=$profile GPUs=$replicas ====="
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            echo "DRY_RUN: CPU_PROFILE=$profile CPUSET=$profile_cpuset REPLICAS=$replicas CODEC=both RESOLUTION=all"
            continue
        fi
        CPU_PROFILE="$profile" CPUSET="$profile_cpuset" MATRIX_ROOT="$MATRIX_ROOT" \
            bash "$SCRIPT_DIR/run_blog_benchmark.sh" "$replicas" auto both both all
    done
done
echo "Complete: $MATRIX_ROOT/summary.json"
    exit 0
fi
CONTAINER="${VLLM_CONTAINER:-vllm-video-nv}"
REPLICAS="${1:-${REPLICAS:-8}}"
CORES="${2:-${CORES:-auto}}"
BACKEND="${3:-${BACKEND:-both}}"
CODEC="${4:-${CODEC:-both}}"
RESOLUTION="${5:-${RESOLUTION:-all}}"
CPU_PROFILE="${CPU_PROFILE:-$CORES}"
CPUSET="${CPUSET:-auto}"
GPU_LIST="${GPU_LIST:-}"
MATRIX_ROOT="${MATRIX_ROOT:-}"

ORIG_CPUSET="$(docker inspect --format '{{.HostConfig.CpusetCpus}}' "$CONTAINER")"
restore_cpuset() {
    if [ -n "$ORIG_CPUSET" ]; then
        docker update --cpuset-cpus "$ORIG_CPUSET" "$CONTAINER" >/dev/null 2>&1 || true
    fi
}
trap restore_cpuset EXIT

VIDEO_DIR="/llm/performance_benchmark/video/videos_captioning"
case "$RESOLUTION" in
    all|both) RESOLUTION_LIST=(720p 1080p 1440p) ;;
    720p) RESOLUTION_LIST=(720p) ;;
    1080p) RESOLUTION_LIST=(1080p) ;;
    2k|2K|1440p|1440) RESOLUTION_LIST=(1440p) ;;
    *) echo "RESOLUTION must be 720p, 1080p, 2k, or all: $RESOLUTION" >&2; exit 1 ;;
esac
case "$CODEC" in
    h264|264) CODEC_LIST=(h264) ;;
    h265|265) CODEC_LIST=(h265) ;;
    both|mixed) CODEC_LIST=(h264 h265) ;;
    *) echo "CODEC must be h264, h265, or both: $CODEC" >&2; exit 1 ;;
esac
case "$BACKEND" in
    opencv) BACKENDS="opencv" ;;
    pynv|pynvvideocodec) BACKENDS="pynvvideocodec" ;;
    both) BACKENDS="opencv pynvvideocodec" ;;
    *) echo "BACKEND must be opencv, pynvvideocodec, or both: $BACKEND" >&2; exit 1 ;;
esac
[[ "$REPLICAS" =~ ^[1-9][0-9]*$ ]] || { echo "REPLICAS must be positive" >&2; exit 1; }
if [ "$CORES" != auto ] && [[ ! "$CORES" =~ ^[1-9][0-9]*$ ]]; then
    echo "CORES must be auto or a positive logical-CPU count" >&2
    exit 1
fi

if [ "$CPUSET" = auto ]; then
    if [ "$CORES" = auto ]; then EFFECTIVE_CPUSET="$ORIG_CPUSET"; else EFFECTIVE_CPUSET="0-$((CORES - 1))"; fi
else
    EFFECTIVE_CPUSET="$CPUSET"
fi
if [ -n "$EFFECTIVE_CPUSET" ] && [ "$EFFECTIVE_CPUSET" != "$ORIG_CPUSET" ]; then
    docker update --cpuset-cpus "$EFFECTIVE_CPUSET" "$CONTAINER" >/dev/null
fi
LOGICAL_CPUS="$(python3 - "$EFFECTIVE_CPUSET" <<'PYCPU'
import sys
spec = sys.argv[1]
count = 0
for part in spec.split(",") if spec else []:
    bounds = [int(value) for value in part.split("-", 1)]
    count += bounds[-1] - bounds[0] + 1
print(count)
PYCPU
)"
[ "$LOGICAL_CPUS" -gt 0 ] || LOGICAL_CPUS="$(nproc --all)"
[ -n "$GPU_LIST" ] || GPU_LIST="$(seq -s, 0 $((REPLICAS - 1)))"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
[ "${#GPU_ARRAY[@]}" -eq "$REPLICAS" ] || { echo "GPU_LIST and REPLICAS mismatch" >&2; exit 1; }

RUN_TS="$(date +%Y%m%d_%H%M%S)"
if [ -n "$MATRIX_ROOT" ]; then
    MATRIX_ROOT="$(realpath -m "$MATRIX_ROOT")"
else
    MATRIX_ROOT="$SCRIPT_DIR/blog_repro/${RUN_TS}_${CPU_PROFILE}_${REPLICAS}gpu"
fi
MATRIX_ROOT_CTR="/llm/performance_benchmark/video/blog_repro/$(basename "$MATRIX_ROOT")"
RUN_ROOT_HOST="$MATRIX_ROOT/${CPU_PROFILE}_${REPLICAS}gpu"
RUN_ROOT_CTR="$MATRIX_ROOT_CTR/${CPU_PROFILE}_${REPLICAS}gpu"
mkdir -p "$RUN_ROOT_HOST"
SUMMARY_JSON="$MATRIX_ROOT/summary.json"

# Create once for run_all_test; subsequent GPU/CPU configurations append to it.
python3 - "$SUMMARY_JSON" "$REPLICAS" "$BACKENDS" "$CPU_PROFILE" "${EFFECTIVE_CPUSET:-all}" "$LOGICAL_CPUS" "$GPU_LIST" <<'PYINIT'
import json, os, sys
path, replicas, backends, profile, cpuset, logical_cpus, gpus = sys.argv[1:]
summary = {"configurations": [], "results": []}
if os.path.isfile(path):
    with open(path, encoding="utf-8") as source:
        summary = json.load(source)
config = {
    "replicas": int(replicas), "gpu_list": [int(gpu) for gpu in gpus.split(",")],
    "backends": backends.split(), "cpu_profile": profile,
    "cpuset_cpus": cpuset, "logical_cpus": int(logical_cpus),
    "cpu_metrics_scope": "host_global_proc_stat",
}
if config not in summary["configurations"]:
    summary["configurations"].append(config)
with open(path, "w", encoding="utf-8") as destination:
    json.dump(summary, destination, indent=2)
    destination.write("\n")
PYINIT

echo "GPU replicas=$REPLICAS ($GPU_LIST), CPU profile=$CPU_PROFILE, cpuset=${EFFECTIVE_CPUSET:-all} ($LOGICAL_CPUS logical CPUs)"
echo "Backend(s)=$BACKENDS Codec(s)=${CODEC_LIST[*]} Resolution(s)=${RESOLUTION_LIST[*]}"
echo "Aggregate JSON: $SUMMARY_JSON"

for codec in "${CODEC_LIST[@]}"; do
    for resolution in "${RESOLUTION_LIST[@]}"; do
        for duration in 10 30; do
            case "$codec/$resolution/$duration" in
                h264/720p/10) VIDEO_FILE="$VIDEO_DIR/bears_720p_h264_10s_8M.mp4" ;;
                h264/720p/30) VIDEO_FILE="$VIDEO_DIR/bears_720p_h264_30s_8M.mp4" ;;
                h264/1080p/10) VIDEO_FILE="$VIDEO_DIR/bears_1080p_h264_10s_8M.mp4" ;;
                h264/1080p/30) VIDEO_FILE="$VIDEO_DIR/bears_1080p_h264_30s_8M.mp4" ;;
                h264/1440p/10) VIDEO_FILE="$VIDEO_DIR/bears_1440p_h264_10s_8M.mp4" ;;
                h264/1440p/30) VIDEO_FILE="$VIDEO_DIR/bears_1440p_h264_30s_8M.mp4" ;;
                h265/720p/10) VIDEO_FILE="$VIDEO_DIR/bears_720p_h265_10s_4M.mp4" ;;
                h265/720p/30) VIDEO_FILE="$VIDEO_DIR/bears_720p_h265_30s_4M.mp4" ;;
                h265/1080p/10) VIDEO_FILE="$VIDEO_DIR/bears_1080p_h265_10s_4M_from60s.mp4" ;;
                h265/1080p/30) VIDEO_FILE="$VIDEO_DIR/bears_1080p_h265_30s_4M_from60s.mp4" ;;
                h265/1440p/10) VIDEO_FILE="$VIDEO_DIR/bears_1440p_h265_10s_4M.mp4" ;;
                h265/1440p/30) VIDEO_FILE="$VIDEO_DIR/bears_1440p_h265_30s_4M.mp4" ;;
            esac
            RUN_LABEL="${codec}_${resolution}_${duration}s"
            echo "Starting $RUN_LABEL: $VIDEO_FILE"
            MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}" \
            MODEL_SRC="${MODEL_SRC:-/DISK0/Qwen3.5-9B}" \
            REPLICAS="$REPLICAS" GPU_LIST="$GPU_LIST" \
            MATRIX_RUN_ROOT_HOST="$RUN_ROOT_HOST" MATRIX_RUN_ROOT_CTR="$RUN_ROOT_CTR" \
            QUANT="${QUANT:-fp8}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" \
            MAX_NUM_SEQS="${MAX_NUM_SEQS:-120}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}" \
            API_SERVER_COUNT="${API_SERVER_COUNT:-2}" HW_DECODERS="${HW_DECODERS:-2}" \
            MM_IPC_GB="${MM_IPC_GB:-2}" CONCURRENCY="${CONCURRENCY:-32}" \
            NUM_PROMPTS="${NUM_PROMPTS:-100}" VIDEO_FILES="$VIDEO_FILE" BACKENDS="$BACKENDS" \
            FRAMES="${FRAMES:-16}" OUTPUT_LEN="${OUTPUT_LEN:-100}" USE_MPS="${USE_MPS:-1}" \
            RUN_LABEL="$RUN_LABEL" CPU_PROFILE="$CPU_PROFILE" CPUSET_CPUS="${EFFECTIVE_CPUSET:-all}" \
            bash "$SCRIPT_DIR/repro_blog_pynvvideocodec.sh"

            python3 - "$SUMMARY_JSON" "$RUN_ROOT_HOST/$RUN_LABEL/summary.json" "$codec" "$resolution" "$duration" "$VIDEO_FILE" "$CPU_PROFILE" "$REPLICAS" "$GPU_LIST" "${EFFECTIVE_CPUSET:-all}" "$LOGICAL_CPUS" <<'PYAPPEND'
import json, os, sys
summary_path, result_path, codec, resolution, duration, video, profile, replicas, gpus, cpuset, logical_cpus = sys.argv[1:]
with open(summary_path, encoding="utf-8") as source:
    summary = json.load(source)
with open(result_path, encoding="utf-8") as source:
    result = json.load(source)
summary["results"].append({
    "codec": codec, "resolution": resolution, "duration_seconds": int(duration),
    "video_file": os.path.basename(video), "cpu_profile": profile,
    "replicas": int(replicas), "gpu_list": [int(gpu) for gpu in gpus.split(",")],
    "cpuset_cpus": cpuset, "logical_cpus": int(logical_cpus),
    "cpu_metrics_scope": "host_global_proc_stat",
    "backends": result.get("backends", {}),
})
with open(summary_path, "w", encoding="utf-8") as destination:
    json.dump(summary, destination, indent=2)
    destination.write("\n")
PYAPPEND
            rm -f "$RUN_ROOT_HOST/$RUN_LABEL/summary.json" "$RUN_ROOT_HOST/$RUN_LABEL/run_info.json"
        done
    done
done
