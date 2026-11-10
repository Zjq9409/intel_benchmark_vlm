curl http://localhost:8008/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/weights/Kimi-VL-A3B-Thinking",
    "image_url": "/weights/workspace/test.png",
    "prompt": "请描述该图片:",
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": false
  }'
