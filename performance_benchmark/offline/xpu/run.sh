export TORCH_LLM_ALLREDUCE=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export  VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 vllm_demo.py \
  --model-path /llm/models/Qwen3-VL-4B-Instruct/ \
  --imgs-path /llm/work/intel_benchmark_vlm/performance_benchmark/dataset/images \
  --batch 32 \
  --dtype float16 \
  --max-model-len 14000 \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --block-size 64 \
  --quantization fp8
