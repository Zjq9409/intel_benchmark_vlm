# Performance测试

## 参数与容器路径映射

| 参数 | 宿主机路径 | 容器内路径 | 说明 |
|------|-----------|-----------|------|
| `--script-dir` | 指定路径（默认：脚本所在目录） | `/llm` | benchmark 脚本、dataset 等均从此路径访问 |
| `--weights-dir` | 指定路径（默认：`../weights`） | `/llm/models` | 模型权重目录，容器内通过 `/llm/models/<model_name>` 引用 |
| `--container-name` | — | — | 自定义容器名（NV 默认 `vllm-nv-container`，XPU 默认 `lsv-container-<suffix>`） |
| `--nv-image` | — | — | NV Docker 完整镜像名（默认 `vllm/vllm-openai:v0.19.1-cu130`） |
| `--intel-image` | — | — | Intel Docker 完整镜像名（默认 `intel/llm-scaler-vllm:0.17.0-xpu`） |

## 快速开始

### 1. 初始化环境

```bash

# 指定 Intel Docker 镜像（完整镜像名）
bash setup_env.sh --weights-dir /data/weights/ --script-dir /data/intel_benchmark_vlm/ --co --intel-image intel/llm-scaler-vllm:0.18.0-xpu

# 指定 NV Docker 镜像
bash setup_env.sh --weights-dir /data/weights/ --script-dir /data/intel_benchmark_vlm/ --co --nv-image vllm/vllm-openai:v0.19.1-cu130

# 可通过 --container-name 自定义容器名（默认：NV=vllm-nv-container，XPU=lsv-container-<suffix>）
bash setup_env.sh --weights-dir /data/weights/ --script-dir /data/intel_benchmark_vlm/ --container-name my-container
```

### 2. 运行在线性能测试

详见 [performance_benchmark/online/online.md](performance_benchmark/online/online.md)  

---

## 工具：compute_input_token.py

计算 VLM 推理时图片输入实际占用的 token 数量，支持以下两种模式：

### 模式一：指定图片文件路径

```bash
python3 compute_input_token.py /path/to/image.jpg
```

自动读取图片尺寸，依次输出 smart_resize、patch 切分、PatchMerger 合并、input_ids 组装各步骤的中间结果。

### 模式二：直接指定图片尺寸（无需图片文件）

```bash
python3 compute_input_token.py 1920x1080
python3 compute_input_token.py 3840x2160
```

格式为 `WxH`（宽×高），无需实际图片，适合快速估算不同分辨率下的 token 数。

### 输出示例

```
Step 0  图片尺寸 (手动指定):  1920x1080 (WxH),  2,073,600 像素
Step 1  smart_resize:
        H: 1080 -> round(1080/32)*32 = 34*32 = 1088
        W: 1920 -> round(1920/32)*32 = 60*32 = 1920
        面积 2,088,960 in [65,536, 16,777,216] -> 无需缩放
        -> resize 1920x1080 -> 1920x1088
Step 2  切 patch:
        grid_h = 1088/16 = 68,  grid_w = 1920/16 = 120
        N_patches = 68x120 = 8,160
Step 3  PatchMerger 2x2:
        N_visual_tokens = 8160 / 4 = 2,040
Step 4  组装 input_ids (tokenizer 精确计算):
        文本合计:        14
        视觉 token:      2,040
        ─────────────────────────────────────────
        总计:            2,054  -> torch.Size([1, 2054])
```

### 说明

- 默认使用 `Qwen3-VL-2B-Instruct` tokenizer（路径 `/data1/models/Qwen3-VL-2B-Instruct`），可在文件头修改 `MODEL_PATH`
- 默认 prompt 为 `"Describe this image."`，token 数 = 视觉 token + 14（chat 模板结构 8 + prompt 词 4 + vision_start/end 2）
