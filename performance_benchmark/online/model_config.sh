#!/bin/bash
# ----------------------------------------------------------------
# Shared model configuration — source this file, then call resolve_model
# Usage: resolve_model <model_key_or_abs_path>
# Sets: MODEL_DIR, SERVER_MODEL_NAME, SERVER_MODEL, TP,
#       SERVER_MODEL_ROOT, SERVER_ALLOWED_MEDIA_PATH
#
# Policy: no auto-detection. Model path must be explicitly provided
# (via MODEL_PATH / absolute path arg / per-model mapping below).
# ----------------------------------------------------------------
resolve_model() {
    local key="$1"
    local model_path=""

    # Default named-model mapping (explicit paths)
    case "$key" in
        4b)
            MODEL_DIR="Qwen3-VL-4B-Instruct"
            model_path="/llm/models/Qwen3-VL-4B-Instruct"
            TP=1
            ;;
        q35-4b)
            MODEL_DIR="Qwen3.5-4B"
            model_path="/llm/models/Qwen3.5-4B"
            TP=1
            ;;
        32b)
            MODEL_DIR="Qwen3-VL-32B-Instruct"
            model_path="/llm/models/Qwen3-VL-32B-Instruct"
            TP=4
            ;;
        q36-35b)
            MODEL_DIR="Qwen3.6-35B-A3B"
            model_path="/DISK0/Qwen3.6-35B-A3B"
            TP=4
            ;;
        q30-a3b)
            MODEL_DIR="Qwen3-VL-30B-A3B-Instruct"
            model_path="/llm/models/Qwen3-VL-30B-A3B-Instruct"
            TP=4
            ;;
        q36-27b)
            MODEL_DIR="Qwen3.6-27B"
            model_path="/llm/models/Qwen3.6-27B"
            TP=4
            ;;
        *)
            MODEL_DIR="Qwen3-VL-30B-A3B-Instruct"
            model_path="/llm/models/Qwen3-VL-30B-A3B-Instruct"
            TP=4
            ;;
    esac

    # Explicit path override priority: MODEL_PATH env > absolute arg
    if [ -n "${MODEL_PATH:-}" ]; then
        model_path="${MODEL_PATH%/}"
        MODEL_DIR="$(basename "$model_path")"
    elif [[ "$key" == /* ]]; then
        model_path="${key%/}"
        MODEL_DIR="$(basename "$model_path")"
    fi

    # TP: always respect TP_OVERRIDE if set
    [ -n "${TP_OVERRIDE:-}" ] && TP="$TP_OVERRIDE"

    SERVER_MODEL="$model_path"
    SERVER_MODEL_ROOT="$(dirname "$model_path")"
    SERVER_ALLOWED_MEDIA_PATH="$SERVER_MODEL_ROOT"
    SERVER_MODEL_NAME="$MODEL_DIR"
}
