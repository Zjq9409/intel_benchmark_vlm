多模态服务服务端启动脚本：
```
export VLLM_USE_V1=1
python -m vllm.entrypoints.openai.api_server \
    --model /home/intel/zjq/models/Qwen2-VL-7B-Instruct \
    --chat-template ./vllm/examples/template_chatml.jinja \
    --tokenizer /home/intel/zjq/models/Qwen2-VL-7B-Instruct \
    --dtype bfloat16 \
    --max_num_seqs 256 \
    --max-model-len 16384 \
    --port 8001 \
    --gpu_memory_utilization 0.95 \
    --host 127.0.0.1
```

多模态服务客户端测试脚本（避免每次执行服务都需要 下载`VisionArena-Chat` 数据集，建议使用离线的 `VisionArena-Chat` ，数据集存放在17层跳板机的 ```/DISK3/home/llm/multimodalLLM/ VisionArena-Chat``` 文件夹，
 打下需要修改的patch包： `git apply benchmark.patch`
```
python3 vllm/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model /home/intel/zjq/models/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --hf-split train \
  --num-prompts 1 \
  --max-concurrency 1 \
  --port 8001 \
  --host 127.0.0.1 \
  --seed 0 \
  --hf-output-len 100 \
  --ignore-eos  \
  --dataset-path /home/intel/zjq/VisionArena-Chat/ \
```

输出：
```
INFO 04-15 10:58:42 [__init__.py:239] Automatically detected platform cuda.
Namespace(backend='openai-chat', base_url=None, host='127.0.0.1', port=8001, endpoint='/v1/chat/completions', dataset_name='hf', dataset_path='/home/intel/zjq/VisionArena-Chat/', max_concurrency=1, model='/home/intel/zjq/models/Qwen2-VL-7B-Instruct', tokenizer=None, use_beam_search=False, num_prompts=1, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, hf_subset=None, hf_split='train', hf_output_len=100, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None)
Resolving data files: 100%|███████████████████████████████| 43/43 [00:00<00:00, 329115.09it/s]
Resolving data files: 100%|████████████████████████████████| 43/43 [00:00<00:00, 38611.66it/s]
image:  (1024, 640)
prompt_len:  11
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|███████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.59s/it]
============ Serving Benchmark Result ============
Successful requests:                     1         
Benchmark duration (s):                  2.12      
Total input tokens:                      11        
Total image and input tokens:            873       
Total generated tokens:                  128       
Request throughput (req/s):              0.47      
Output token throughput (tok/s):         60.47     
Total Token throughput (tok/s):          65.67     
---------------Time to First Token----------------
Mean TTFT (ms):                          36.83     
Median TTFT (ms):                        36.83     
P99 TTFT (ms):                           36.83     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.37     
Median TPOT (ms):                        16.37     
P99 TPOT (ms):                           16.37     
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.24     
Median ITL (ms):                         16.36     
P99 ITL (ms):                            16.50     
==================================================
```
