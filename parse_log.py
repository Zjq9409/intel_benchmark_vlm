import re
import csv

def extract_accuracy_log(log_file):

    # 读取本地 log 文件
    with open(log_file, 'r', encoding='utf-8') as file:
        log = file.read()
    
    
    import re
    
    # 按记录块拆分（每条记录以 "Processing:" 开头）
    blocks = re.split(r'(?=Processing:)', log)
    
    # 存储提取结果
    results = []
    
    # 遍历每条记录进行提取
    for block in blocks:
        image_match = re.search(r'Image\s*:\s*(\S+)', block)
        output_match = re.search(r'输出结果：\s*(.+?)(?:={5,}|Processing:|$)', block, re.DOTALL)
    
        if image_match and output_match:
            image_path = image_match.group(1).strip()
            output_text = output_match.group(1).strip().replace('\n', ' ')
            results.append((image_path, output_text))
    
    # 输出结果示例（可改为写入 CSV / JSON）
    for i, (img, out) in enumerate(results, 1):
        print(f"[{i}] Image Path: {img}")
        print(f"    输出结果：{out}\n")

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
       
        # 获取 Mean TTFT 并判断是否超过5秒（5000毫秒）
        mean_ttft = float(metrics['Mean TTFT (ms)'].group(1)) if metrics['Mean TTFT (ms)'] else None
        if mean_ttft is not None and mean_ttft > 5000:
            continue  # 跳过该结果

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

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="提取日志指标并保存到CSV")
    parser.add_argument("log_file", type=str, help="日志文件名")
    parser.add_argument(
       "--performance",
       type=lambda x: x.lower() in ['true', '1', 'yes'],
       default=True,
       help="是否输出性能数据（默认：True）"
    )
    args = parser.parse_args()

    base_name = os.path.basename(args.log_file)
    output_file = f"{base_name}.csv"
    if args.performance: 
       extract_log_metrics(args.log_file, output_file)
    else:
       extract_accuracy_log(args.log_file)
