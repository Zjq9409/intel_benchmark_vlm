bash ../../setup_env.sh \
    --container-name vllm-jane-xpu \
    --weights-dir /home/intel/jane/weights/ \
    --script-dir $(realpath ../..) \
    --intel-image intel/llm-scaler-vllm:0.14.0-b8.3
