import re

# 读取本地 log 文件
with open('arc-qwen2-vl-7b-1-fp8-res', 'r', encoding='utf-8') as file:
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

