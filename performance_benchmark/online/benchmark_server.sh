
python vlm_benchmark.py \
--prompt "$(cat /home/intel/llm_test/intel_benchmark_vlm/performance_benchmark/prompt_128.txt)" \
--model  /home/intel/llm_test/weights/Qwen3-VL-4B-Instruct/ \
--served-model-name Qwen3-VL-4B-Instruct \
--batch_size 3 \
--port 8000 \
--host 127.0.0.1 \
--ignore-eos \
--output_len 128 
