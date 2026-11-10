# COCO 数据集下载工具

用于下载 COCO 图像数据集及生成对应的元数据 JSON 文件，供多模态性能测试使用。

## 功能特点

- 自动从 HuggingFace 下载 COCO 数据集
- 生成标准格式的元数据 JSON
- 支持自定义下载数量
- 支持图片统一尺寸调整（可选，默认1080P）
- 内置数据集验证功能

## 使用方法

### 1. 下载数据集（默认 400 张图像）

```bash
cd /home/intel/llm_test/intel_benchmark_vlm/performance_benchmark

# 下载 400 张图像（默认）
python3 download_dataset.py

# 自定义数量
python3 download_dataset.py --num-images 320

# 指定输出目录
python3 download_dataset.py --output-dir ./my_dataset --num-images 500
```

**交互式图片尺寸调整：**

下载完成后，脚本会询问是否将所有图片调整到统一尺寸：

```
是否将所有图片调整到统一尺寸？(y/N): y
请输入目标宽度 (默认1920): 
请输入目标高度 (默认1080): 
```

- 直接回车使用默认尺寸 1920x1080（1080P）
- 输入自定义宽高
- 输入 N 跳过尺寸调整
- 调整时保持原始宽高比，使用高质量算法

### 2. 验证数据集

```bash
# 验证现有数据集
python3 download_dataset.py --verify

# 验证指定目录的数据集
python3 download_dataset.py --verify --output-dir ./dataset
```

## 输出结构

下载完成后，数据集目录结构如下：

```
dataset/
├── images/                    # 图像文件夹
│   ├── coco_000000000009.jpg
│   ├── coco_000000000025.jpg
│   └── ...
└── metadata.json             # 元数据文件
```

## 元数据格式

`metadata.json` 格式示例：

```json
[
  {
    "image": "coco_000000000009.jpg",
    "image_id": 9,
    "conversations": [
      {
        "from": "human",
        "value": "请详细描述这张图片的内容。"
      },
      {
        "from": "gpt",
        "value": ""
      }
    ]
  }
]
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 输出目录 | `./dataset` |
| `--num-images` | 下载图像数量 | `400` |
| `--verify` | 仅验证不下载 | - |

## 使用示例

### 场景 1: 首次下载

```bash
# 下载 400 张图像到 ./dataset 目录
python3 download_dataset.py
```

### 场景 2: 下载特定数量

```bash
# 下载 320 张图像
python3 download_dataset.py --num-images 320
```

### 场景 3: 指定输出目录

```bash
# 下载到自定义目录
python3 download_dataset.py --output-dir ../test_data --num-images 200
```

### 场景 4: 验证数据集

```bash
# 验证数据集完整性
python3 download_dataset.py --verify
```

### 场景 5: 下载并统一图片尺寸

```bash
# 下载 400 张图像
python3 download_dataset.py

# 在提示时选择调整尺寸
是否将所有图片调整到统一尺寸？(y/N): y
请输入目标宽度 (默认1920): 1280
请输入目标高度 (默认1080): 720

# 所有图片将被调整为 1280x720 (保持宽高比)
```

## 依赖要求

需要安装以下 Python 包：

```bash
pip install datasets tqdm pillow
```

## 注意事项

1. **网络连接**：首次下载需要从 HuggingFace 拉取数据集，请确保网络通畅
2. **磁盘空间**：400 张图像约占用 50-100 MB 空间
3. **下载时间**：根据网络速度，400 张图像约需 2-5 分钟
4. **重复下载**：如果目标目录已存在，会覆盖原有数据
5. **图片调整**：调整尺寸时保持宽高比，不会变形。使用 LANCZOS 算法保证质量
6. **尺寸建议**：
   - 1080P: 1920x1080 (默认)
   - 720P: 1280x720
   - 4K: 3840x2160

## 故障排查

### 问题 1: ModuleNotFoundError: No module named 'datasets'

**解决方案**：
```bash
pip install datasets
```

### 问题 2: 下载速度慢

**解决方案**：
- 使用代理或 VPN
- 减少下载数量
- 使用国内镜像源

### 问题 3: 图像和元数据不匹配

**解决方案**：
```bash
# 重新下载
rm -rf dataset
python3 download_dataset.py --num-images 400
```

## 集成到测试流程

下载完成后，可以在测试脚本中使用：

```python
import json
from pathlib import Path

# 加载元数据
with open('dataset/metadata.json', 'r') as f:
    dataset = json.load(f)

# 遍历数据集
for item in dataset:
    image_path = Path('dataset/images') / item['image']
    question = item['conversations'][0]['value']
    # 进行测试...
```

## 更新日志

- **v1.1** - 新增图片尺寸统一调整功能，支持自定义尺寸（默认1080P）
- **v1.0** - 初始版本，支持 COCO 数据集下载和元数据生成
