#!/bin/bash
# Install nsight-systems-cli inside vllm-nv-container
# Usage: bash install_nsight_systems_cli.sh [container_name]

CONTAINER=${1:-vllm-nv-container}

echo "=== Installing nsight-systems-cli in container: $CONTAINER ==="

# Step 1: Install gnupg
echo "[1/4] Installing gnupg..."
docker exec "$CONTAINER" bash -c "apt-get update -qq && apt-get install -y --no-install-recommends gnupg"

# Step 2: Add NVIDIA devtools repository
# Note: The key F60F4B3D7FA2AF80 is already present in /usr/share/keyrings/nvidia.gpg
# inside NVIDIA CUDA base images. Referencing it via signed-by avoids needing apt-key.
# (Do NOT use apt-key adv --fetch-keys; the HTTP endpoint is unreachable in this env)
echo "[2/4] Adding NVIDIA devtools apt repository..."
docker exec "$CONTAINER" bash -c "
  echo 'deb [signed-by=/usr/share/keyrings/nvidia.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu\$(source /etc/lsb-release; echo \"\$DISTRIB_RELEASE\" | tr -d .)/\$(dpkg --print-architecture) /' \
  > /etc/apt/sources.list.d/nvidia-devtools.list
  cat /etc/apt/sources.list.d/nvidia-devtools.list
"

# Step 3: apt update
echo "[3/4] Running apt update..."
docker exec "$CONTAINER" bash -c "apt-get update 2>&1 | grep -E 'devtools|Err|error' || true"

# Step 4: Install nsight-systems-cli
echo "[4/4] Installing nsight-systems-cli..."
docker exec "$CONTAINER" bash -c "apt-get install -y nsight-systems-cli"

# Verify
echo ""
echo "=== Verification ==="
docker exec "$CONTAINER" nsys --version
