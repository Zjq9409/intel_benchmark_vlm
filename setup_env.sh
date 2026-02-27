#!/bin/bash

# ============================================================
# Environment Setup Script
# - NVIDIA GPU: pulls vllm/vllm-openai Docker image and starts container
#               weights dir auto-detected as ../weights, or pass as $1
# - Intel GPU:  pulls intel/llm-scaler-vllm Docker image and starts container
#               image version passed as $1 (default: current hardcoded ver)
# Usage (NV):    $0 [weights_dir]
#   Example:     $0
#   Example:     $0 /data/models
# Usage (Intel): $0 [image_version]
#   Example:     $0 0.11.1-b7
# ============================================================

# SCRIPT_DIR: defaults to the directory containing this script; override via env var
SCRIPT_DIR="${SCRIPT_DIR:-$(dirname "$(realpath "$0")")}" 
CONTAINER_NAME="lsv-container"
IMAGE_BASE="intel/llm-scaler-vllm"

echo "============================================================"
echo "  Environment Setup"
echo "============================================================"

# ----------------------------------------------------------------
# Detect GPU type
# ----------------------------------------------------------------
USE_NV=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    NV_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NV_COUNT" -gt 0 ]; then
        USE_NV=1
    fi
fi

if [ "$USE_NV" -eq 1 ]; then
    echo "  Detected NVIDIA GPU — using Docker image path."

    # ----------------------------------------------------------------
    # NV path: Docker vllm image
    # ----------------------------------------------------------------
    NV_IMAGE="vllm/vllm-openai:v0.15.1-cu130"
    NV_CONTAINER="vllm-nv-container"
    # WEIGHTS_DIR: env var > $1 arg > auto-detect as ../weights relative to SCRIPT_DIR
    WEIGHTS_DIR="${WEIGHTS_DIR:-${1:-$(dirname "$SCRIPT_DIR")/weights}}"

    echo "  Image:       $NV_IMAGE"
    echo "  Container:   $NV_CONTAINER"
    echo "  Weights dir: $WEIGHTS_DIR"

    # Step 1: Pull image if not already present
    echo ""
    echo "[1/2] Checking Docker image: $NV_IMAGE ..."
    if docker image inspect "$NV_IMAGE" &>/dev/null; then
        echo "  Image already exists locally, skipping pull."
    else
        echo "  Image not found locally. Pulling..."
        docker pull "$NV_IMAGE"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to pull image $NV_IMAGE"
            exit 1
        fi
        echo "  Image pulled successfully."
    fi

    # Step 2: Check container state and start if needed
    echo ""
    echo "[2/2] Checking for existing container: $NV_CONTAINER ..."
    RUNNING=$(docker ps --filter "name=^/${NV_CONTAINER}$" --format "{{.Names}}")
    EXISTING=$(docker ps -a --filter "name=^/${NV_CONTAINER}$" --format "{{.Names}}")

    if [ -n "$RUNNING" ]; then
        echo "  Container '$NV_CONTAINER' is already running. Entering container..."
        # docker exec -it "$NV_CONTAINER" /bin/bash
    else
        if [ -n "$EXISTING" ]; then
            echo "  Found stopped container '$NV_CONTAINER'. Removing..."
            docker rm "$NV_CONTAINER"
        fi

        echo "  Starting container: $NV_CONTAINER ..."
        docker run -td \
            --runtime nvidia --gpus all \
            --name "$NV_CONTAINER" \
            -v "$WEIGHTS_DIR":/llm/models \
            -v "$SCRIPT_DIR":/llm \
            -e no_proxy=localhost,127.0.0.1 \
            -e http_proxy="$http_proxy" \
            -e https_proxy="$https_proxy" \
            --ipc=host \
            --entrypoint /bin/bash \
            "$NV_IMAGE"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to start container."
            exit 1
        fi
        echo "  Container '$NV_CONTAINER' started successfully."
        # docker exec -it "$NV_CONTAINER" /bin/bash
    fi

    echo ""
    echo "============================================================"
    echo "NV environment setup complete."
    echo "  Container : $NV_CONTAINER"
    echo "  Weights   : $WEIGHTS_DIR -> /models (inside container)"
    echo "  Scripts   : $SCRIPT_DIR  -> /workspace (inside container)"
    echo "  Port      : 8000 (host) -> 8000 (container)"
    echo "============================================================"
    exit 0
fi

echo "  No NVIDIA GPU detected — using Intel Docker path."

# ----------------------------------------------------------------
# Intel path: Docker image + container
# ----------------------------------------------------------------
IMAGE_VERSION=${1:-0.11.1-b7}
# WEIGHTS_DIR: env var > auto-detect as ../weights relative to SCRIPT_DIR
WEIGHTS_DIR="${WEIGHTS_DIR:-$(dirname "$SCRIPT_DIR")/weights}"
FULL_IMAGE="${IMAGE_BASE}:${IMAGE_VERSION}"
echo "  Image: $FULL_IMAGE"

# Step 1: Pull the image (skip if already exists locally)
echo ""
echo "[1/2] Checking Docker image: $FULL_IMAGE ..."
if sudo docker image inspect "$FULL_IMAGE" &>/dev/null; then
    echo "Image already exists locally, skipping pull."
else
    echo "Image not found locally. Pulling..."
    sudo docker pull "$FULL_IMAGE"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to pull image $FULL_IMAGE"
        exit 1
    fi
    echo "Image pulled successfully."
fi

# Step 2: Check container state and start if needed
echo ""
echo "[2/2] Checking for existing container: $CONTAINER_NAME ..."
RUNNING=$(sudo docker ps --filter "name=^/${CONTAINER_NAME}$" --format "{{.Names}}")
EXISTING=$(sudo docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format "{{.Names}}")

if [ -n "$RUNNING" ]; then
    echo "Container '$CONTAINER_NAME' is already running. Entering container..."
    # sudo docker exec -it "$CONTAINER_NAME" /bin/bash
    exit 0
else
    if [ -n "$EXISTING" ]; then
        echo "Found stopped container '$CONTAINER_NAME'. Removing..."
        sudo docker rm "$CONTAINER_NAME" 2>/dev/null
        echo "Existing container removed."
    else
        echo "No existing container found."
    fi

    echo "Starting container: $CONTAINER_NAME ..."
    sudo docker run -td \
        --privileged \
        --net=host \
        --device=/dev/dri \
        --name="$CONTAINER_NAME" \
        -v "$WEIGHTS_DIR":/llm/models \
        -v "$SCRIPT_DIR":/llm \
        -e no_proxy=localhost,127.0.0.1 \
        -e http_proxy="$http_proxy" \
        -e https_proxy="$https_proxy" \
        --shm-size="32g" \
        --entrypoint /bin/bash \
        "$FULL_IMAGE"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to start container."
        exit 1
    fi
    echo "Container '$CONTAINER_NAME' started successfully."
    # docker exec -it "$CONTAINER_NAME" /bin/bash
fi

echo ""
echo "============================================================"
echo "Environment setup complete."
echo "============================================================"
