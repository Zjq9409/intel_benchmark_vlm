#!/bin/bash
# Install nsight-systems inside vllm-nv-container
# Usage: bash install_nsight_systems_cli.sh [container_name] [version]
#
# Package sources:
#   - nsight-systems <= 2025.5.2 : CUDA apt repo (pre-configured in the image)
#   - nsight-systems >= 2026.x.x  : NVIDIA devtools repo (added by this script)

CONTAINER=${1:-vllm-nv-container}
VERSION=${2:-2026.1.1}

echo "=== Installing nsight-systems-${VERSION} in container: $CONTAINER ==="

# Step 1: Fetch and register the NVIDIA devtools GPG key
echo "[1/3] Adding NVIDIA devtools GPG key..."
docker exec "$CONTAINER" bash -c "
  curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg
  echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64 /' \
    > /etc/apt/sources.list.d/nvidia-devtools.list
"

# Step 2: Update apt and install
echo "[2/3] Running apt-get update..."
docker exec "$CONTAINER" bash -c "apt-get update -qq"

echo "[3/3] Installing nsight-systems-${VERSION}..."
docker exec "$CONTAINER" bash -c "apt-get install -y nsight-systems-${VERSION} 2>&1 | tail -6"

# Verify
echo ""
echo "=== Verification ==="
docker exec "$CONTAINER" nsys --version
