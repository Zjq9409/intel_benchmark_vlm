import re
import csv

def extract_log_metrics(log_file, output_csv):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 拆分多个 benchmark 结果块
    results = content.split('============ Serving Benchmark Result ============')
    metrics_list = []

    for result in results[1:]:  # 跳过第一个前缀部分
        metrics = {}
        metrics['Total input tokens'] = re.search(r'Total input tokens:\s+(\d+)', result)
        metrics['Total image and input tokens'] = re.search(r'Total image and input tokens:\s+(\d+)', result)
        metrics['Total generated tokens'] = re.search(r'Total generated tokens:\s+(\d+)', result)
        metrics['Successful requests'] = re.search(r'Successful requests:\s+(\d+)', result)
        metrics['Mean TTFT (ms)'] = re.search(r'Mean TTFT \(ms\):\s+([\d.]+)', result)
        metrics['Mean TPOT (ms)'] = re.search(r'Mean TPOT \(ms\):\s+([\d.]+)', result)
        metrics['Mean ITL (ms)'] = re.search(r'Mean ITL \(ms\):\s+([\d.]+)', result)
        metrics['Request throughput (req/s)'] = re.search(r'Request throughput \(req/s\):\s+([\d.]+)', result)
        metrics['Output token throughput (tok/s)'] = re.search(r'Output token throughput \(tok/s\):\s+([\d.]+)', result)
        metrics['Benchmark duration (s)'] = re.search(r'Benchmark duration \(s\):\s+([\d.]+)', result)

        # 按指定顺序提取数值
        ordered_values = [
            metrics['Total input tokens'].group(1) if metrics['Total input tokens'] else '',
            metrics['Total image and input tokens'].group(1) if metrics['Total image and input tokens'] else '',
            metrics['Total generated tokens'].group(1) if metrics['Total generated tokens'] else '',
            metrics['Successful requests'].group(1) if metrics['Successful requests'] else '',
            metrics['Mean TTFT (ms)'].group(1) if metrics['Mean TTFT (ms)'] else '',
            metrics['Mean TPOT (ms)'].group(1) if metrics['Mean TPOT (ms)'] else '',
            metrics['Mean ITL (ms)'].group(1) if metrics['Mean ITL (ms)'] else '',
            metrics['Request throughput (req/s)'].group(1) if metrics['Request throughput (req/s)'] else '',
            metrics['Output token throughput (tok/s)'].group(1) if metrics['Output token throughput (tok/s)'] else '',
            metrics['Benchmark duration (s)'].group(1) if metrics['Benchmark duration (s)'] else '',
        ]
        metrics_list.append(ordered_values)

    # 写入CSV文件
    header = [
        'Total input tokens',
        'Total image and input tokens',
        'Total generated tokens',
        'Successful requests',
        'Mean TTFT (ms)',
        'Mean TPOT (ms)',
        'Mean ITL (ms)',
        'Request throughput (req/s)',
        'Output token throughput (tok/s)',
        'Benchmark duration (s)'
    ]
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(metrics_list)

    print(f"成功提取并保存到: {output_csv}")

# 用法示例
extract_log_metrics('personal_log', 'output_metrics.csv')
