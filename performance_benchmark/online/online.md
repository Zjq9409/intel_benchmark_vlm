# vLLM 在线性能测试

## 运行性能测试

无需下载数据集，直接运行 benchmark 脚本即可。脚本会自动启动 vllm 服务端并依次测试各并发量：

```bash
cd performance_benchmark/online

# 测试 30B 模型（默认），图片 1280×720（默认）
bash vllm_random_benchmark_server.sh

# 测试 4B 模型，图片 1280×720（默认）
bash vllm_random_benchmark_server.sh 4b

# 测试 30B 模型，图片 512×512
bash vllm_random_benchmark_server.sh 30b 512 512

# 测试 4B 模型，图片 512×512
bash vllm_random_benchmark_server.sh 4b 512 512

# 批量运行多个组合（见 run_both.sh）
bash run_both.sh
```

### 脚本参数

| 位置参数 | 可选值 | 默认值 | 说明 |
|---------|--------|--------|------|
| `$1` 模型规格 | `4b` / `30b` | `30b` | 选择测试模型；`4b` → `Qwen3-VL-4B-Instruct`（TP=1），`30b` → `Qwen3-VL-30B-A3B-Instruct`（TP=4） |
| `$2` 图片宽度 | 任意整数 | `1280` | 随机生成图片的宽度（像素） |
| `$3` 图片高度 | 任意整数 | `720` | 随机生成图片的高度（像素） |

### 服务端固定配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8006` | 服务端口 |
| `MAX_BATCHED_TOKENS` | `8192` | 最大批处理 token 数 |
| `MAX_MODEL_LEN` | `16384` | 最大模型上下文长度 |
| `GPU_MEM_UTIL` | `0.8` | GPU 显存利用率 |
| `INPUT_LEN` | `1024` | 随机输入 token 长度 |
| `OUTPUT_LEN` | `1024` | 随机输出 token 长度 |
| `MAX_BSIZE` | `200` | num-prompts 最大值 |
| TTFT 阈值 | `6000 ms` | 超过此值时自动停止测试并保存结果 |

测试循环范围：num-prompts = 1, 2, 4, 6, …（步长 2），最大 200，或 Mean TTFT > 6000ms 时提前终止。

### 日志文件命名规则

日志保存在以模型名命名的子目录下，文件名格式为：

```
<MODEL_NAME>/<YYYYMMDD_HHMMSS>_client_tp<TP>_mbt<MBT>_<W>x<H>_in<IN>_out<OUT>_<GPU_TYPE>.log
<MODEL_NAME>/<YYYYMMDD_HHMMSS>_server_tp<TP>_mbt<MBT>_<W>x<H>_in<IN>_out<OUT>_<GPU_TYPE>.log
```

例如：`Qwen3-VL-30B-A3B-Instruct/20260320_153000_client_tp4_mbt8192_1280x720_in1024_out1024_H100.log`

## 测试结果

测试脚本会输出以下性能指标：
- **Request throughput**: 请求吞吐量（req/s）
- **Output token throughput**: 输出token吞吐量（tok/s）
- **TTFT** (Time to First Token): 首token延迟
- **TPOT** (Time per Output Token): 每个输出token的平均时间
- **ITL** (Inter-token Latency): token间延迟
- **Total Token throughput**: 总token吞吐量（包含输入+输出）

## 解析测试日志

测试完成后脚本会自动调用 `parse_log.py` 生成 CSV，也可手动解析：

```bash
# 解析性能测试日志
python3 parse_log.py <MODEL_NAME>/xxx_client_xxx.log

# 会生成同名 .csv 文件，包含以下指标：
# - Total input tokens
# - Total image and input tokens
# - Total generated tokens
# - Successful requests
# - Mean TTFT (ms)
# - Mean TPOT (ms)
# - Mean ITL (ms)
# - Request throughput (req/s)
# - Output token throughput (tok/s)
# - Benchmark duration (s)
```

## 测试其他模型

如需测试非内置的模型，修改 `vllm_random_benchmark_server.sh` 脚本顶部的模型选择逻辑：

```bash
# 在 if/else 分支中添加新模型或修改已有配置
if [ "$MODEL_SELECT" = "4b" ]; then
    SERVER_MODEL="/llm/models/Qwen3-VL-4B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-4B-Instruct"
    TP=1
elif [ "$MODEL_SELECT" = "8b" ]; then          # ← 新增示例
    SERVER_MODEL="/llm/models/Qwen3-VL-8B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-8B-Instruct"
    TP=2
else
    SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
    SERVER_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"
    TP=4
fi
```

同时确认：
1. **模型权重**已放置在宿主机 `weights/` 目录下（容器内映射为 `/llm/models/`）
2. **TP**（tensor parallel）值与可用 GPU 卡数匹配
3. 若需要不同的量化策略，修改 `VLLM_SERVER_ARGS` 中的 `--quantization` 参数
