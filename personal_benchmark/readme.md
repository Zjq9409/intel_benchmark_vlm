- 首先启动服务端脚本：
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
- 准确率测试脚本:
```
python vlm_benchmark.py \
--image_path ./test.png  \
--prompt "请描述图片"      \
--model /data/models/Qwen2-VL-7B-Instruct \
--batch_size 1 \
--port 8001 \
--host 127.0.0.1
```

准确率测试输出结果（仅供参考，以不同平台实际输出为准）
```
image size:  (1080, 810)
input prompt len:  3
image and prompt len:  1145
输出结果： 图中是一辆黄色的卡车，卡车的货箱正在倾倒，卡车的后面有一台黄色的机器，机器的后面有一个人穿着红色的工作服，戴着黄色的安全帽，站在黑色的柏油路上。卡车的后面有几栋红色的建筑，建筑的后面是一片绿色的树林。
image and prompt len:  1145
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  0.95      
Total input tokens:                      3         
Total generated tokens:                  67        
Request throughput (req/s):              1.06      
Output token throughput (tok/s):         70.81     
Total Token throughput (tok/s):          73.98     
---------------Time to First Token----------------
Mean TTFT (ms):                          195.25    
Median TTFT (ms):                        195.25    
P99 TTFT (ms):                           195.25    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.37     
Median TPOT (ms):                        11.37     
P99 TPOT (ms):                           11.37     
---------------Inter-token Latency----------------
Mean ITL (ms):                           11.04     
Median ITL (ms):                         9.59      
P99 ITL (ms):                            45.22     
==================================================
```

- 性能测试脚本，强制输出指定的`output_len`长度，可以统一不同平台输出长度来对比性能
```
python vlm_benchmark.py \
--image_path ./test.png  \
--prompt "请描述图片"      \
--model /data/models/Qwen2-VL-7B-Instruct \
--batch_size 1 \
--port 8001 \
--ignore-eos \
--output_len 512 \
--host 127.0.0.1
```
性能测试结果输出，`Total generated tokens:`输出为指定的`--output_len`长度。（仅供参考，以不同平台实际输出为准）
```
image size:  (1080, 810)
input prompt len:  3
image and prompt len:  1145
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  5.23      
Total input tokens:                      3         
Total generated tokens:                  512       
Request throughput (req/s):              0.19      
Output token throughput (tok/s):         97.88     
Total Token throughput (tok/s):          98.46     
---------------Time to First Token----------------
Mean TTFT (ms):                          209.90    
Median TTFT (ms):                        209.90    
P99 TTFT (ms):                           209.90    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.82      
Median TPOT (ms):                        9.82      
P99 TPOT (ms):                           9.82      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.79      
Median ITL (ms):                         9.60      
P99 ITL (ms):                            10.23     
==================================================
```

- 存在server_name的测试脚本（可选）
```
python vlm_benchmark.py  \
--image_path ./test.png  \
--prompt "请描述图片" \
--model /data/models/Qwen2-VL-7B-Instruct \
--served-model-name Qwen2-VL-7B-Instruct \
--batch_size 1 \
--port 8001 \
--host 127.0.0.1
```
