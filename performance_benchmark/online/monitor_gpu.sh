#!/bin/bash
# monitor_gpu.sh — Continuously sample Intel XPU metrics via xpu-smi
# Metrics: GPU Util, Power, Frequency, Temp, Mem Used, Mem Util
# Output: timestamped lines to stdout (redirect to a log file from caller)
#
# Usage: bash monitor_gpu.sh [interval_seconds]
#   interval_seconds: sampling interval (default 5)

INTERVAL="${1:-5}"

echo "# $(date '+%Y-%m-%d %H:%M:%S') GPU monitor started (interval=${INTERVAL}s)"
echo "# Fields: timestamp | device | tile | gpu_util(%) | power(W) | freq(MHz) | temp(C) | mem_used(MiB) | mem_util(%)"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    sudo xpu-smi dump -m 0,1,2,3,18,22 -d -1 -n 1 2>/dev/null \
        | awk -v ts="$TS" 'NR>1 && NF { print ts " | " $0 }'
    sleep "$INTERVAL"
done
