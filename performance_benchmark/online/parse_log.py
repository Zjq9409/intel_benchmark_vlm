import sys
import os
import json
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
    # 提取所有数字
    candidates = [float(i) if '.' in i else int(i) for i in line.split() if is_number(i)]
    return candidates[index] if index < len(candidates) else 0

if __name__ == '__main__':
    results = []
    current_group = {}
    previous_line = None  # 用于检查重复

    with open(rawdatafile, encoding="UTF-8") as f:
        for dataline in f:
            dataline = dataline.strip()
            
            # 检查重复数据行
            if dataline == previous_line:
                continue  # 跳过重复行
            previous_line = dataline

            if dataline.startswith('>>> Running vllm bench serve with --num-prompts='):
                # 如果遇到新的 Batch Size 开头，保存上一组数据并开始新组
                if current_group:
                    results.append(current_group)
                bsize = int(dataline.split('--num-prompts=')[1].split()[0])
                current_group = {'Batch Size': bsize}
            elif dataline.strip().lower().startswith('successful requests:'):
                current_group['Batch Size'] = get_num(dataline)
            elif dataline.startswith('Benchmark duration (s):'):
                current_group['E2E Latency (s)'] = get_num(dataline)
            elif dataline.startswith('Request throughput (req/s):'):
                current_group['Request Throughput (req/s)'] = get_num(dataline)
            elif dataline.startswith('Output token throughput (tok/s):'):
                current_group['Output Token Throughput (tok/s)'] = get_num(dataline)
            elif dataline.startswith('Mean TTFT (ms)'):
                current_group['Mean TTFT (ms)'] = get_num(dataline)
            elif dataline.startswith('Mean TPOT (ms):'):
                current_group['Mean TPOT (ms)'] = get_num(dataline)
            elif dataline.startswith('Mean ITL (ms):'):
                # 遇到 ITL 时，将完整组追加到结果中
                results.append(current_group)
                current_group = {}  # 清空当前组

    # 确保最后一组数据被追加
    if current_group:
        results.append(current_group)

    # 输出文件名
    base_filename = os.path.splitext(os.path.basename(rawdatafile))[0]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(rawdatafile)), 'LOG')
    os.makedirs(output_dir, exist_ok=True)

    # 定义目标列头顺序
    headers = [
        "Batch Size", "Mean TTFT (ms)", "Mean TPOT (ms)",
        "Output Token Throughput (tok/s)", "Request Throughput (req/s)",
        "E2E Latency (s)"
    ]

    # 写入四舍五入的 CSV 文件
    with open(os.path.join(output_dir, f'{base_filename}.csv'), 'w', encoding="utf-8") as f:
        f.write(", ".join(headers) + '\n')
        for result_dict in results:
            if not result_dict:
                continue
            
            row_data = []
            for header in headers:
                value = result_dict.get(header, 0)
                if isinstance(value, float):
                    row_data.append(f'{value:.2f}')
                else:
                    row_data.append(str(value))
            
            f.write(", ".join(row_data) + '\n')

    print(f"Data successfully processed! Outputs saved as {os.path.join(output_dir, base_filename)}.csv")
