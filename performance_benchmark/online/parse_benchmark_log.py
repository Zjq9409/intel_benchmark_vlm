#!/usr/bin/env python3
"""
Parse vllm benchmark log files and export the 2nd run (actual, skip 1st warm-up)
metrics per batch size to CSV.

Usage:
    python parse_benchmark_log.py <log_file> [output_csv]

If output_csv is omitted, it defaults to <log_file_basename>.csv
"""

import re
import csv
import sys
from pathlib import Path

# Metrics to extract, in output order
METRIC_PATTERNS = {
    "Maximum request concurrency":     r"Maximum request concurrency:\s+([\d.]+)",
    "Mean TTFT (ms)":                  r"Mean TTFT \(ms\):\s+([\d.]+)",
    "Mean TPOT (ms)":                  r"Mean TPOT \(ms\):\s+([\d.]+)",
    "Output token throughput (tok/s)": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "Request throughput (req/s)":      r"Request throughput \(req/s\):\s+([\d.]+)",
    "Benchmark duration (s)":          r"Benchmark duration \(s\):\s+([\d.]+)",
}

RESULT_HEADER = "============ Serving Benchmark Result ============"
RESULT_FOOTER = "=================================================="
BATCH_TRIGGER = r">>> Running vllm bench serve with --num-prompts=(\d+)"


def parse_log(log_path: str):
    """
    Returns a list of dicts, one per batch size, using the 2nd result block
    (1st block = warm-up, 2nd block = actual measurement).
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sections by the ">>> Running ..." marker
    sections = re.split(r"(>>> Running vllm bench serve with --num-prompts=\d+)", content)

    # sections[0]  = text before first marker (ignored)
    # sections[1]  = marker string
    # sections[2]  = body of that batch
    # sections[3]  = next marker, ...

    batches = {}  # num_prompts -> list of result blocks (in order)

    i = 1
    while i < len(sections) - 1:
        marker = sections[i]
        body   = sections[i + 1]
        i += 2

        m = re.search(BATCH_TRIGGER, marker)
        if not m:
            continue
        num_prompts = int(m.group(1))

        # Extract all result blocks within this body
        blocks = []
        start = 0
        while True:
            h = body.find(RESULT_HEADER, start)
            if h == -1:
                break
            f_pos = body.find(RESULT_FOOTER, h + len(RESULT_HEADER))
            if f_pos == -1:
                break
            block = body[h: f_pos + len(RESULT_FOOTER)]
            blocks.append(block)
            start = f_pos + len(RESULT_FOOTER)

        if num_prompts not in batches:
            batches[num_prompts] = []
        batches[num_prompts].extend(blocks)

    rows = []
    for num_prompts in sorted(batches.keys()):
        blocks = batches[num_prompts]
        if len(blocks) < 2:
            print(f"[WARN] num-prompts={num_prompts}: only {len(blocks)} result block(s) found, "
                  f"skipping (need 2).")
            continue

        # Use the 2nd block (index 1), skip 1st warm-up
        target_block = blocks[1]
        row = {"num_prompts": num_prompts}
        for col, pattern in METRIC_PATTERNS.items():
            m = re.search(pattern, target_block)
            row[col] = float(m.group(1)) if m else ""

        rows.append(row)

    return rows


def write_csv(rows, output_path: str):
    if not rows:
        print("No data to write.")
        return

    fieldnames = ["num_prompts"] + list(METRIC_PATTERNS.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows -> {output_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = str(Path(log_path).with_suffix(".csv"))

    rows = parse_log(log_path)
    write_csv(rows, output_path)


if __name__ == "__main__":
    main()
