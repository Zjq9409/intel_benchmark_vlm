#!/bin/bash
# ----------------------------------------------------------------
# Shared model configuration — source this file, then call resolve_model
# Usage: resolve_model <model_key>
# Sets: MODEL_DIR, SERVER_MODEL_NAME, SERVER_MODEL, TP
#
# To add a new model: add one case entry here only.
# ----------------------------------------------------------------
resolve_model() {
    local key="$1"
    case "$key" in
        4b)
            MODEL_DIR="Qwen3-VL-4B-Instruct"
            SERVER_MODEL="/llm/models/Qwen3-VL-4B-Instruct"
            TP=1
            ;;
        q35-4b)
            MODEL_DIR="Qwen3.5-4B"
            SERVER_MODEL="/llm/models/Qwen3.5-4B"
            TP=1
            ;;
        32b)
            MODEL_DIR="Qwen3-VL-32B-Instruct"
            SERVER_MODEL="/llm/models/Qwen3-VL-32B-Instruct"
            TP=4
            ;;
        q36-35b)
            MODEL_DIR="Qwen3.6-35B-A3B"
            SERVER_MODEL="/DISK0/Qwen3.6-35B-A3B"
            TP=4
            ;;
        *)
            MODEL_DIR="Qwen3-VL-30B-A3B-Instruct"
            SERVER_MODEL="/llm/models/Qwen3-VL-30B-A3B-Instruct"
            TP=4
            ;;
    esac
    SERVER_MODEL_NAME="$MODEL_DIR"
}
