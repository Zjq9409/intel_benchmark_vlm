#!/bin/bash

# ============================================================
# Environment Setup Script
# - NVIDIA GPU: sets up uv venv (Python 3.12) and installs vllm
# - Intel GPU:  pulls Docker image and starts container
# Usage: $0 [image_version]   (image_version only needed for Intel path)
# Example (Intel): $0 0.11.1-b7
# Example (NV):    $0
# ============================================================

SCRIPT_DIR="$(dirname "$(realpath "$0")")" 
VENV_DIR="$SCRIPT_DIR/.venv"
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
    echo "  Detected NVIDIA GPU — using uv venv install path."

    # ----------------------------------------------------------------
    # NV path: uv + vllm
    # ----------------------------------------------------------------

    # Check / install uv
    echo ""
    echo "[1/2] Checking uv..."
    if ! command -v uv &>/dev/null; then
        echo "  uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env"
        if ! command -v uv &>/dev/null; then
            echo "ERROR: uv installation failed or not in PATH."
            exit 1
        fi
        echo "  uv installed: $(uv --version)"
    else
        echo "  uv found: $(uv --version)"
    fi

    # Create venv if needed
    echo ""
    echo "[2/2] Setting up Python 3.12 venv and installing vllm..."
    if [ ! -d "$VENV_DIR" ]; then
        echo "  Creating venv at $VENV_DIR ..."
        uv venv --python 3.12 --seed --managed-python "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create venv."
            exit 1
        fi
        echo "  Venv created."
    else
        echo "  Venv already exists at $VENV_DIR, skipping creation."
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate venv."
        exit 1
    fi
    echo "  Venv activated: $(which python) ($(python --version))"

    # Install vllm
    if python -c "import vllm" &>/dev/null; then
        VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
        echo "  vllm already installed (version: $VLLM_VER), skipping install."
    else
        echo "  Installing vllm (--torch-backend=auto)..."
        uv pip install vllm --torch-backend=auto
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install vllm."
            exit 1
        fi
        echo "  vllm installed: $(python -c 'import vllm; print(vllm.__version__)')"
    fi

    echo ""
    echo "============================================================"
    echo "NV environment setup complete."
    echo "Activate with: source $VENV_DIR/bin/activate"
    echo "============================================================"
    exit 0
fi

echo "  No NVIDIA GPU detected — using Intel Docker path."

# ----------------------------------------------------------------
# Intel path: Docker image + container
# ----------------------------------------------------------------
IMAGE_VERSION=${1:-0.11.1-b7}
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
    sudo docker exec -it "$CONTAINER_NAME" /bin/bash
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
        -v /home/intel/llm_test/weights/:/llm/models/ \
        -v /home/intel/llm_test/:/llm/ \
        -v "$(dirname "$(realpath "$0")"):/llm_test/" \
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
    docker exec -it "$CONTAINER_NAME" /bin/bash
fi

echo ""
echo "============================================================"
echo "Environment setup complete."
echo "============================================================"
