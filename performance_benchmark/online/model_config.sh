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
    local MODEL_PATH_OVERRIDE=""
    local EXPLICIT_MODEL_PATH=""
    local KEY_MATCHED=0

    # 1) default named-model mapping
    case "$key" in
        4b)
            KEY_MATCHED=1
            MODEL_DIR="Qwen3-VL-4B-Instruct"
            MODEL_PATH_OVERRIDE="/llm/models/Qwen3-VL-4B-Instruct"
            TP=1
            ;;
        q35-4b)
            KEY_MATCHED=1
            MODEL_DIR="Qwen3.5-4B"
            MODEL_PATH_OVERRIDE="/llm/models/Qwen3.5-4B"
            TP=1
            ;;
        32b)
            KEY_MATCHED=1
            MODEL_DIR="Qwen3-VL-32B-Instruct"
            MODEL_PATH_OVERRIDE="/llm/models/Qwen3-VL-32B-Instruct"
            TP=4
            ;;
        q36-35b)
            KEY_MATCHED=1
            MODEL_DIR="Qwen3.6-35B-A3B"
            MODEL_PATH_OVERRIDE="/llm/models/Qwen3.6-35B-A3B"
            TP=2
            ;;
        q30-a3b|30b)
            KEY_MATCHED=1
            MODEL_DIR="Qwen3-VL-30B-A3B-Instruct"
            MODEL_PATH_OVERRIDE="/llm/models/Qwen3-VL-30B-A3B-Instruct"
            TP=4
            ;;
        q36-27b)
            KEY_MATCHED=1
            MODEL_DIR="Qwen3.6-27B"
            MODEL_PATH_OVERRIDE="/llm/models/Qwen3.6-27B"
            TP=4
            ;;
        *)
            KEY_MATCHED=0
            ;;
    esac

    # 2) explicit path override (highest priority): MODEL_PATH env > absolute arg
    if [ -n "${MODEL_PATH:-}" ]; then
        EXPLICIT_MODEL_PATH="${MODEL_PATH%/}"
    elif [[ "$key" == /* ]]; then
        EXPLICIT_MODEL_PATH="${key%/}"
    fi

    if [ -z "$EXPLICIT_MODEL_PATH" ] && [ "$KEY_MATCHED" -eq 0 ]; then
        echo "ERROR: Unknown model key '$key'. Valid keys: 4b, q35-4b, 32b, q36-35b, q30-a3b, 30b, q36-27b" >&2
        echo "  Or pass an absolute path / set MODEL_PATH=/your/path." >&2
        return 1
    fi

    if [ -n "$EXPLICIT_MODEL_PATH" ]; then
        MODEL_PATH_OVERRIDE="$EXPLICIT_MODEL_PATH"
        MODEL_DIR="$(basename "$MODEL_PATH_OVERRIDE")"
        TP=4
    fi

    # 3) Validate model path — must be /llm/models/... and exist (inside container).
    case "$MODEL_PATH_OVERRIDE" in
        /llm/models/*) ;;
        *)
            echo "ERROR: Model path must be under /llm/models/, got: $MODEL_PATH_OVERRIDE" >&2
            echo "  Mount your weights dir to /llm/models via setup_env.sh --weights-dir." >&2
            return 1
            ;;
    esac

    _in_ctr=0
    { [ -f "/.dockerenv" ] || grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; } && _in_ctr=1
    if [ "$_in_ctr" = "1" ] && [ ! -d "$MODEL_PATH_OVERRIDE" ]; then
        echo "ERROR: Model path not found inside container: $MODEL_PATH_OVERRIDE" >&2
        echo "  Check that the weights dir is mounted to /llm/models and contains '$MODEL_DIR'." >&2
        return 1
    fi

    SERVER_MODEL="$MODEL_PATH_OVERRIDE"
    SERVER_MODEL_ROOT="$(dirname "$MODEL_PATH_OVERRIDE")"
    SERVER_ALLOWED_MEDIA_PATH="$SERVER_MODEL_ROOT"
    SERVER_MODEL_NAME="$MODEL_DIR"
}
