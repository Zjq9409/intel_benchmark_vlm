# 视频解码后端性能测试

## 执行流程

```text
30 FPS H.264/H.265 视频
        ↓
setup_container.sh：创建/启动容器，仓库根目录挂载为 /llm
        ↓
run_video_backend_test.sh：一个命令完成全部流程
  ├─ 依次启动 PyNvVideoCodec、OpenCV 服务
  ├─ 内部调用 run_video_bench_serve.sh 发送视频请求并记录指标
  └─ 每轮结束后停止本轮启动的服务，结果保存到时间戳 bench 目录
```

## 运行

在仓库根目录运行一次容器初始化（已有可用容器时跳过）：

```bash
bash performance_benchmark/video/setup_container.sh
```

然后运行对比脚本：

```bash
bash performance_benchmark/video/run_video_backend_test.sh
```

脚本无须传参。默认使用 `Qwen3.5-2B`、FP8、32 帧及 `videos_h264_30fps/` 和 `videos_h265_30fps/` 两个目录，依次测试 PyNvVideoCodec 与 OpenCV。每个后端并发测试 `1 4`；请求数自动设为最高并发的 8 倍（当前为每档 32 个），每次输出 32 tokens。

测试配置集中在 `run_video_backend_test.sh` 文件开头。需要调整模型、GPU、帧数、并发档位或视频目录时，直接修改对应常量。两个测试服务分别使用 8013 和 8014 端口；若端口已被占用，先停止占用服务再运行。

结果位于 `performance_benchmark/video/<模型名>/<时间戳>_bench/`，包含 `bench_serve.log`、生成的 ShareGPT 请求数据及 vLLM 保存的指标 JSON。
