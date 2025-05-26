# Performance测试
## 1. 测试输入的文本的输入与输出长度
1. 128/128
2. 1024/512， 1024/1024
3. 2048/512， 2048/1024， 10220384/2048

128输入的prompt参看文件prompt_128.txt
1024输入的prompt参看文件prompt_1024.txt
2048输入的prompt参看文件prompt_2048.txt
## 2. 测试图像输入
https://github.com/Zjq9409/intel_benchmark_vlm/blob/master/accuracy/zto/shangmen/40_20240903093021_78830772282673.jpg

## 3. 测试batch size选择
1. 对于enable_prefix_caching为1的情况，如果TTFT不大于5秒，batch size最大为32.
2. 对于enable_prefix_caching为0的情况，batch size从1，2，4，8，10，16，32，64增加，当TTFT大于5秒，测试结束。
   
## 4. 测试代码示例
```python
python vlm_benchmark.py \
--image_path ./test.jpg  \
--prompt `cat prompt_128.txt` \
--model /llm/gemma3-12b \
--served-model-name gemma3-12b \
--batch_size 1 \
--port 8000 \
--host 127.0.0.1 \
--ignore-eos \
--output_len 128 
```