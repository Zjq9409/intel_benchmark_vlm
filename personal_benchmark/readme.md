首先启动服务端脚本：
```
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Qwen2-VL-7B-Instruct \
    --chat-template ./vllm/examples/template_chatml.jinja \
    --tokenizer /data/models/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --max_num_seqs 64 \
    --max-model-len 16384 \
    --port 8001 \
    --host 127.0.0.1

```
测试脚本
```
python vlm_benchmark.py \
--image_path ./image.jpg  \
--prompt "请描述图片"      \
--model /data/models/Qwen2-VL-7B-Instruct \
--batch_size 1 \
--port 8001 \
--host 127.0.0.1
```

存在server_name的测试脚本
```
python vlm_benchmark.py  \
--image_path ./image.jpg  \
--prompt "请描述图片" \
--model /data/models/Qwen2-VL-7B-Instruct \
--served-model-name Qwen2-VL-7B-Instruct \
--batch_size 1 \
--port 8001 \
--host 127.0.0.1
```

输出结果
```
image size:  (1080, 810)
input prompt len:  3
输出结果： 图中是一辆黄色的卡车，卡车的货箱正在倾倒，卡车的后面有一台黄色的机器，机器的后面有一个人穿着红色的工作服，戴着黄色的安全帽，站在黑色的柏油路上。卡车的后面有几栋红色的建筑，建筑的后面是一片绿色的树林。
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  0.94      
Total input tokens:                      3         
Total generated tokens:                  67        
Request throughput (req/s):              1.06      
Output token throughput (tok/s):         70.93     
Total Token throughput (tok/s):          74.10     
---------------Time to First Token----------------
Mean TTFT (ms):                          180.99    
Median TTFT (ms):                        180.99    
P99 TTFT (ms):                           180.99    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.56     
Median TPOT (ms):                        11.56     
P99 TPOT (ms):                           11.56     
---------------Inter-token Latency----------------
Mean ITL (ms):                           11.22     
Median ITL (ms):                         9.60      
P99 ITL (ms):                            49.37     
==================================================
```