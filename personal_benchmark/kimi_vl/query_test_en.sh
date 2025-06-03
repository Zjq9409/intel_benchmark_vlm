curl http://localhost:8008/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/weights/Kimi-VL-A3B-Thinking",
    "image_url": "/weights/workspace/test.png",
    "prompt": "Please describe this image",
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": false
  }'


#    "image_path": "/weights/workspace/test.png",
#    "prompt": "Please describe this image",
#    "max_tokens": 128,
#    "temperature": 0.7,
#    "stream": false
