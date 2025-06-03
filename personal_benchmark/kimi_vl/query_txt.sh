curl http://localhost:8008/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/weights/Kimi-VL-A3B-Thinking",
    "prompt": "请基于OpenCV写一段代码，加载指定mp4视频文件，并均匀抽取100帧，保存为.npz格式的numpy数组",
    "max_tokens": 24,
    "temperature": 0.7,
    "stream": true
  }'
