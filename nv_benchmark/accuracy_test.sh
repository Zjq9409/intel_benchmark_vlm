python /root/jane/intel_benchmark_vlm/personal_benchmark/vlm_benchmark.py \
--image_path /root/jane/intel_benchmark_vlm/accuracy/zto/shangmen/40_20240903093021_78830772282673.jpg \
--prompt "Please utilize the visual large model to analyze the image and determine the presence of the following elements: 1. Entrance door (Y/N), 2. Package (Y/N). Output the results concatenated by a comma, for example: Y,N."      \
--model  /root/jane/Qwen2-VL-7B-Instruct/ \
--batch_size 1 \
--port 8000 \
--host 127.0.0.1