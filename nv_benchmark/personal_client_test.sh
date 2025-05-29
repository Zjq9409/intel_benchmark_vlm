MODEL="/root/jane/Qwen2-VL-7B-Instruct/"
#MODEL="/disk0/LLM/Qwen2-VL-72B-Instruct-FP8-Dynamic/"
for bs in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32; do
for out_lens in 128 512 1024; do
    python ../personal_benchmark/vlm_benchmark.py \
    --image_path ../accuracy/zto/shangmen/40_20240903093021_78830772282673.jpg  \
    --prompt "Please utilize the visual large model to analyze the image and determine the presence of the following elements: 1. Entrance door (Y/N), 2. Package (Y/N). Output the results concatenated by a comma, for example: Y,N."      \
    --model  ${MODEL} \
    --batch_size ${bs} \
    --port 8000 \
    --host 127.0.0.1 \
    --ignore-eos \
    --output_len ${out_lens} 
done
done
