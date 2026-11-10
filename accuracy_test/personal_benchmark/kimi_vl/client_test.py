import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8008/v1",
    api_key='token-abc123',
)

image_path = "/weights/workspace/test.png"
image = Image.open(image_path).convert("RGB")

buffered = BytesIO()
image.save(buffered, format="PNG")
img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
base64_image_url = f"data:image/png;base64,{img_b64_str}"
prompt="Please describe this picture"
#prompt="请描述该图片"

messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": base64_image_url}}, {"type": "text", "text": prompt}], "stream":True}
]

completion = client.chat.completions.create(
  model="/weights/Kimi-VL-A3B-Thinking",
  messages=messages
)

print(completion.choices[0].message)
