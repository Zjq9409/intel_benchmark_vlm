import time
import httpx


def read_prompt(f_prompt):
    with open(f_prompt, 'r') as h:
        prompt = h.read().rstrip(' \t\r\n')
    return prompt


url = "http://localhost:8008/v1/completions"
f_prompt = '/weights/workspace/prompt_zto.txt'
prompt = read_prompt(f_prompt)
headers = {"Content-Type": "application/json"}
data = {
    "seed": 0,
    "model": "/weights/Kimi-VL-A3B-Thinking",
    "image_path": "/weights/workspace/shangmen673.jpg",
    "prompt": prompt,
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": True,
}

start_time = time.time()
first_token_time = None
tokens = []
answer = ''
with httpx.stream("POST", url, headers=headers, json=data, timeout=60) as response:
    for chunk in response.iter_text():
        if chunk.strip() == "": continue
        if chunk.startswith("data:"):
            try:
              answer += chunk.split('{')[2].split('"')[5]
            except:
              pass
            # 提取 token
            tokens.append(1)  # 每个 token 计为 1
            if first_token_time is None:
                first_token_time = time.time()

# 计算指标
total_time = time.time() - start_time
first_token_latency = first_token_time - start_time if first_token_time else None
tpot = (total_time - first_token_latency) / (len(tokens) - 1) if len(tokens) > 1 else None
print(answer)
print(f"Total time: {total_time:.4f}s")
print(f"First token latency: {first_token_latency}s")
print(f"TPOT: {tpot}s/token")
