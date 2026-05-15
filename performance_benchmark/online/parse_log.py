import sys
import re
import os
import unicodedata

rawdatafile = sys.argv[1]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def get_num(line: str, index: int = 0) -> float:
    candidates = [float(i) if '.' in i else int(i) for i in line.split() if is_number(i)]
    return candidates[index] if index < len(candidates) else 0

def get_field(line: str) -> str:
    parts = line.split(':', 1)
    return parts[1].strip() if len(parts) > 1 else ''

if __name__ == '__main__':
    results = []
    current_group = {}
    ctx = {}
    previous_line = None

    with open(rawdatafile, encoding="UTF-8") as f:
        for dataline in f:
            dataline = dataline.strip()
            if dataline == previous_line:
                continue
            previous_line = dataline

            # ── Header block ─────────────────────────────────────────────
            if dataline.startswith('GPU Type:'):
                ctx['Device'] = get_field(dataline)
            elif dataline.startswith('Server Model Name:'):
                # Qwen3-VL-4B-Instruct → Qwen3-VL-4B
                name = get_field(dataline)
                name = re.sub(r'-Instruct$', '', name)
                ctx['Model'] = name
            elif dataline.startswith('Quantization:'):
                ctx['Precision'] = get_field(dataline).upper()
            elif dataline.startswith('TP:'):
                ctx['TP'] = int(get_num(dataline))
            elif dataline.startswith('Images per request:'):
                ctx['Images per request'] = int(get_num(dataline))

            # ── BENCHMARK marker line ─────────────────────────────────────
            # [20260514_120104] === BENCHMARK batch=1 imgs=1 res=1280x720 in=512 out=128 quant=fp8 ===
            elif '=== BENCHMARK' in dataline:
                m = re.search(r'imgs=(\d+)', dataline)
                if m:
                    ctx['Images per request'] = int(m.group(1))
                m = re.search(r'res=(\d+)x(\d+)', dataline)
                if m:
                    ctx['Image Size'] = f"{m.group(1)}*{m.group(2)}"
                m = re.search(r'\bin=(\d+)', dataline)
                if m:
                    ctx['Input Text Length'] = int(m.group(1))
                m = re.search(r'\bout=(\d+)', dataline)
                if m:
                    ctx['Output Length'] = int(m.group(1))

            # ── Start of a benchmark run ──────────────────────────────────
            elif dataline.startswith('>>> Running vllm bench serve with --num-prompts='):
                if current_group:
                    results.append(current_group)
                bsize = int(dataline.split('--num-prompts=')[1].split()[0])
                current_group = {**ctx, 'batch_size': bsize, 'DP': 1, 'PP': 1}

            # ── Result metrics ────────────────────────────────────────────
            elif dataline.startswith('Benchmark duration (s):'):
                current_group['E2E Latency (s)'] = get_num(dataline)
            elif dataline.startswith('Request throughput (req/s):'):
                current_group['QPS (req/s)'] = get_num(dataline)
            elif dataline.startswith('Output token throughput (tok/s):'):
                current_group['TPS (tokens/s)'] = get_num(dataline)
            elif dataline.startswith('Mean TTFT (ms)'):
                current_group['TTFT (ms)'] = get_num(dataline)
            elif dataline.startswith('Mean TPOT (ms):'):
                current_group['TPOT (ms)'] = get_num(dataline)
            elif dataline.startswith('Mean ITL (ms):'):
                results.append(current_group)
                current_group = {}

    if current_group:
        results.append(current_group)

    # ── Output ───────────────────────────────────────────────────────────
    base_filename = os.path.splitext(os.path.basename(rawdatafile))[0]
    output_dir = os.path.dirname(os.path.abspath(rawdatafile))

    headers = [
        "Device", "Model", "Precision", "TP", "DP", "PP",
        "Image Size", "Input Text Length", "Output Length",
        "Images per request", "batch_size", "TTFT (ms)", "TPOT (ms)", "TPS (tokens/s)", "QPS (req/s)", "E2E Latency (s)"
    ]

    outpath = os.path.join(output_dir, f'{base_filename}.csv')
    with open(outpath, 'w', encoding="utf-8") as f:
        f.write(", ".join(headers) + '\n')
        for result_dict in results:
            if not result_dict:
                continue
            row_data = []
            for header in headers:
                value = result_dict.get(header, '')
                if isinstance(value, float):
                    row_data.append(f'{value:.2f}')
                else:
                    row_data.append(str(value))
            f.write(", ".join(row_data) + '\n')

    print(f"CSV saved: {outpath}")
