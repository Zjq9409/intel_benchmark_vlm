- 首先启动服务端脚本：(NV 上 添加export VLLM_USE_V1=1环境变量)
```
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Qwen2-VL-7B-Instruct \
    --chat-template ./vllm/examples/template_chatml.jinja \
    --tokenizer /data/models/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --max_num_seqs 256 \
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
输出结果： 这张图片展示了一辆黄色的自卸卡车正在倾倒货物。卡车停在一条新铺好的柏油路上，背景中可以看到一些树木和建筑物。卡车的货箱已经打开，显示出里面装有碎石或沙子等建筑材料。在卡车旁边，有一个穿着橙色工作服和黄色安全帽的工人，他似乎在监督卡车的操作。天空晴朗，阳光明媚，整个场景看起来像是在一个建筑工地或道路施工现场。
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  1.66      
Total input tokens:                      3         
Total image and input tokens:            1145      
Total generated tokens:                  97        
Request throughput (req/s):              0.60      
Output token throughput (tok/s):         58.49     
Total Token throughput (tok/s):          60.30     
---------------Time to First Token----------------
Mean TTFT (ms):                          81.17     
Median TTFT (ms):                        81.17     
P99 TTFT (ms):                           81.17     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.42     
Median TPOT (ms):                        16.42     
P99 TPOT (ms):                           16.42     
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.25     
Median ITL (ms):                         16.41     
P99 ITL (ms):                            16.63     
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
--host 127.0.0.1 \
--ignore-eos \
--output_len 512 
```
性能测试结果输出，`Total generated tokens:`输出为指定的`--output_len`长度。（仅供参考，以不同平台实际输出为准）
```
image size:  (1080, 810)
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  8.48      
Total input tokens:                      3         
Total image and input tokens:            1145      
Total generated tokens:                  512       
Request throughput (req/s):              0.12      
Output token throughput (tok/s):         60.34     
Total Token throughput (tok/s):          60.70     
---------------Time to First Token----------------
Mean TTFT (ms):                          82.05     
Median TTFT (ms):                        82.05     
P99 TTFT (ms):                           82.05     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.44     
Median TPOT (ms):                        16.44     
P99 TPOT (ms):                           16.44     
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.41     
Median ITL (ms):                         16.44     
P99 ITL (ms):                            16.59     
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
